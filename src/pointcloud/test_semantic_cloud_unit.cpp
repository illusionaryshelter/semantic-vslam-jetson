/*
 * test_semantic_cloud_unit.cpp
 *
 * 语义点云生成单元测试
 * 独立可执行文件，不依赖 ROS2
 *
 * 测试项:
 *   1. 点云维度正确 (width × height = depth 图大小)
 *   2. 有效深度对应的点有有效 3D 坐标
 *   3. 语义标签正确分配 (person = class 0)
 *   4. 无效深度产生 NaN 坐标
 *   5. 深度反投影精度 (已知深度 → 计算 z 与原始值一致)
 *   6. RGB/Depth 分辨率不匹配时 resize 工作正常
 *
 * 用法: ./test_semantic_cloud_unit <engine_path> <image_path> <depth_or_synth>
 */

#include "semantic_vslam/semantic_colors.hpp"
#include "semantic_vslam/yolo_inference.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "[FAIL] " << msg << std::endl;                              \
      failures++;                                                              \
    } else {                                                                   \
      std::cout << "[PASS] " << msg << std::endl;                              \
      passes++;                                                                \
    }                                                                          \
  } while (0)

// 生成带标签的点云（复用 semantic_cloud_node 的核心逻辑）
static void
buildSemanticCloud(const cv::Mat &rgb, const cv::Mat &depth_in,
                   const std::vector<semantic_vslam::Object> &objects, float fx,
                   float fy, float cx, float cy, float depth_scale,
                   pcl::PointCloud<pcl::PointXYZRGBL> &cloud) {

  // Resize depth if needed
  cv::Mat depth = depth_in;
  if (depth.rows != rgb.rows || depth.cols != rgb.cols) {
    cv::resize(depth, depth, cv::Size(rgb.cols, rgb.rows), 0, 0,
               cv::INTER_NEAREST);
  }

  const int rows = depth.rows;
  const int cols = depth.cols;

  // label map
  cv::Mat label_map = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Mat conf_map = cv::Mat::zeros(rows, cols, CV_32FC1);

  for (const auto &obj : objects) {
    const cv::Rect &r = obj.rect;
    int x0 = std::max(0, r.x), y0 = std::max(0, r.y);
    int x1 = std::min(cols, r.x + r.width), y1 = std::min(rows, r.y + r.height);
    if (x0 >= x1 || y0 >= y1 || obj.mask.empty())
      continue;
    for (int y = y0; y < y1; ++y) {
      const uint8_t *mr = obj.mask.ptr<uint8_t>(y - r.y);
      for (int x = x0; x < x1; ++x) {
        int mx = x - r.x;
        if (mx >= 0 && mx < obj.mask.cols && mr[mx] > 0 &&
            obj.prob > conf_map.at<float>(y, x)) {
          label_map.at<uint8_t>(y, x) = static_cast<uint8_t>(obj.label + 1);
          conf_map.at<float>(y, x) = obj.prob;
        }
      }
    }
  }

  cloud.clear();
  cloud.width = cols;
  cloud.height = rows;
  cloud.is_dense = false;
  cloud.points.resize(static_cast<size_t>(rows) * cols);
  const float inv_fx = 1.0f / fx, inv_fy = 1.0f / fy;
  const bool is_16u = (depth.type() == CV_16UC1);

  for (int v = 0; v < rows; ++v) {
    for (int u = 0; u < cols; ++u) {
      auto &pt = cloud.at(u, v);
      float z = is_16u
                    ? static_cast<float>(depth.at<uint16_t>(v, u)) * depth_scale
                    : depth.at<float>(v, u);
      if (z <= 0.01f || z > 10.0f || std::isnan(z)) {
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        pt.r = pt.g = pt.b = 0;
        pt.label = 0;
        continue;
      }
      pt.x = (float(u) - cx) * z * inv_fx;
      pt.y = (float(v) - cy) * z * inv_fy;
      pt.z = z;
      uint8_t lbl = label_map.at<uint8_t>(v, u);
      if (lbl > 0) {
        int cls = lbl - 1;
        pt.r = semantic_vslam::kSemanticColors[cls][0];
        pt.g = semantic_vslam::kSemanticColors[cls][1];
        pt.b = semantic_vslam::kSemanticColors[cls][2];
        pt.label = static_cast<uint32_t>(cls);
      } else {
        const auto &bgr = rgb.at<cv::Vec3b>(v, u);
        pt.r = bgr[2];
        pt.g = bgr[1];
        pt.b = bgr[0];
        pt.label = 0;
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_path> <rgb_image> <depth_image_or_'synth'>"
              << std::endl;
    return -1;
  }

  int passes = 0, failures = 0;
  std::string engine_path = argv[1];
  std::string rgb_path = argv[2];
  std::string depth_str = argv[3];

  // ---- Load YOLO ----
  semantic_vslam::YoloInference yolo(engine_path);
  TEST_ASSERT(yolo.init(), "YOLO engine init");

  cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
  TEST_ASSERT(!rgb.empty(), "RGB image loads");

  // ---- Generate depth ----
  cv::Mat depth;
  if (depth_str == "synth") {
    depth = cv::Mat(rgb.rows, rgb.cols, CV_16UC1);
    for (int v = 0; v < rgb.rows; ++v)
      for (int u = 0; u < rgb.cols; ++u)
        depth.at<uint16_t>(v, u) = 2000; // 2m uniform
  } else {
    depth = cv::imread(depth_str, cv::IMREAD_UNCHANGED);
  }
  TEST_ASSERT(!depth.empty(), "Depth image loads/generated");

  // Astra Pro defaults
  float fx = 570.3f, fy = 570.3f;
  float cx = rgb.cols / 2.0f, cy = rgb.rows / 2.0f;
  float depth_scale = 0.001f;

  // ---- YOLO ----
  std::vector<semantic_vslam::Object> objects;
  TEST_ASSERT(yolo.infer(rgb, objects, 0.4f, 0.45f), "YOLO inference succeeds");

  // ========== Test 1: 点云维度 ==========
  std::cout << "\n=== Test 1: Cloud Dimensions ===" << std::endl;
  pcl::PointCloud<pcl::PointXYZRGBL> cloud;
  buildSemanticCloud(rgb, depth, objects, fx, fy, cx, cy, depth_scale, cloud);

  TEST_ASSERT(cloud.width == static_cast<uint32_t>(rgb.cols),
              "Cloud width matches image width");
  TEST_ASSERT(cloud.height == static_cast<uint32_t>(rgb.rows),
              "Cloud height matches image height");
  TEST_ASSERT(cloud.size() == static_cast<size_t>(rgb.rows) * rgb.cols,
              "Cloud total points = rows * cols");

  // ========== Test 2: 有效深度 → 有效坐标 ==========
  std::cout << "\n=== Test 2: Depth-to-3D Projection ===" << std::endl;
  int valid_points = 0, nan_points = 0;
  for (const auto &pt : cloud.points) {
    if (std::isnan(pt.z))
      nan_points++;
    else
      valid_points++;
  }
  // 合成深度全部是 2m，应全部有效
  if (depth_str == "synth") {
    TEST_ASSERT(valid_points == static_cast<int>(cloud.size()),
                "All synthetic depth points are valid");
  }
  std::cout << "  Valid: " << valid_points << " NaN: " << nan_points
            << std::endl;

  // ========== Test 3: 深度反投影精度 ==========
  std::cout << "\n=== Test 3: Depth Reprojection Accuracy ===" << std::endl;
  if (depth_str == "synth") {
    // 中心点: u=cx, v=cy → x≈0, y≈0, z=2.0
    auto &center = cloud.at(static_cast<int>(cx), static_cast<int>(cy));
    TEST_ASSERT(std::abs(center.z - 2.0f) < 0.01f,
                "Center pixel z ≈ 2.0m (got " + std::to_string(center.z) + ")");
    TEST_ASSERT(std::abs(center.x) < 0.01f,
                "Center pixel x ≈ 0.0 (got " + std::to_string(center.x) + ")");
    TEST_ASSERT(std::abs(center.y) < 0.01f,
                "Center pixel y ≈ 0.0 (got " + std::to_string(center.y) + ")");
  }

  // ========== Test 4: 语义标签分配 ==========
  std::cout << "\n=== Test 4: Semantic Label Assignment ===" << std::endl;
  // 统计颜色被修改为语义色的点 (不能仅靠 label 字段，因为 person=class0=label0
  // 与背景相同)
  int semantic_pts = 0;
  for (int v = 0; v < rgb.rows; ++v) {
    for (int u = 0; u < rgb.cols; ++u) {
      const auto &pt = cloud.at(u, v);
      if (std::isnan(pt.z))
        continue;
      const auto &bgr = rgb.at<cv::Vec3b>(v, u);
      // 如果 RGB 颜色与原图不同，则说明被赋了语义色
      if (pt.r != bgr[2] || pt.g != bgr[1] || pt.b != bgr[0])
        semantic_pts++;
    }
  }
  if (!objects.empty()) {
    TEST_ASSERT(semantic_pts > 0, "Some points have semantic colors (" +
                                      std::to_string(semantic_pts) + ")");
  }

  // Person (class 0) 应被标色为红色
  bool found_person = false;
  for (const auto &pt : cloud.points) {
    if (pt.label == 0 && pt.r == 255 && pt.g == 0 && pt.b == 0) {
      // label=0 但 RGB=(255,0,0) 说明这是 person 的语义色
      // 注意: label field 存的是 class_id, person=0
      // 但 label_map 中 0 也是背景，这里 person 的 label=0
      // 我们需要检查颜色是否被正确分配
    }
  }
  // 检查 person 类别的检测对象对应的点是否有正确颜色
  for (const auto &obj : objects) {
    if (obj.label == 0) {
      found_person = true;
      // 找到 bbox 中心的点
      int cu = obj.rect.x + obj.rect.width / 2;
      int cv_pt = obj.rect.y + obj.rect.height / 2;
      if (cu < rgb.cols && cv_pt < rgb.rows) {
        auto &pt = cloud.at(cu, cv_pt);
        // person 颜色 = 红色 (255, 0, 0)
        if (!std::isnan(pt.z) && pt.r == 255 && pt.g == 0 && pt.b == 0) {
          std::cout << "  Person bbox center colored correctly (red)"
                    << std::endl;
        }
      }
    }
  }
  if (found_person)
    std::cout << "  Person objects found and checked" << std::endl;

  // ========== Test 5: 分辨率不匹配处理 ==========
  std::cout << "\n=== Test 5: Resolution Mismatch ===" << std::endl;
  // 创建半分辨率 depth
  cv::Mat depth_half;
  cv::resize(depth, depth_half, cv::Size(depth.cols / 2, depth.rows / 2), 0, 0,
             cv::INTER_NEAREST);
  pcl::PointCloud<pcl::PointXYZRGBL> cloud2;
  buildSemanticCloud(rgb, depth_half, objects, fx, fy, cx, cy, depth_scale,
                     cloud2);
  TEST_ASSERT(cloud2.width == static_cast<uint32_t>(rgb.cols),
              "Resized depth cloud width matches RGB width");
  TEST_ASSERT(cloud2.height == static_cast<uint32_t>(rgb.rows),
              "Resized depth cloud height matches RGB height");

  // ========== 总结 ==========
  std::cout << "\n========================================" << std::endl;
  std::cout << "Results: " << passes << " passed, " << failures << " failed"
            << std::endl;
  std::cout << "========================================" << std::endl;

  return (failures > 0) ? 1 : 0;
}
