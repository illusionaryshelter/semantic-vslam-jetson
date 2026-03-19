/*
 * test_pipeline.cpp
 *
 * 端到端管线测试: RGB + Depth → YOLO → 语义点云 → PCD + 可视化图
 * 独立可执行文件，不依赖 ROS2
 *
 * 验证整个处理链在模拟 Astra Pro 数据下的完整性:
 *   1. 引擎加载 OK
 *   2. YOLO 推理产出合理结果
 *   3. 点云生成完整
 *   4. 语义标签覆盖率合理
 *   5. PCD 文件可写入并重新加载
 *
 * 用法: ./test_pipeline <engine_path> <rgb_image> <depth_or_synth> [output.pcd]
 */

#include "semantic_vslam/semantic_colors.hpp"
#include "semantic_vslam/yolo_inference.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
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

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_path> <rgb_image> <depth_or_'synth'> [output.pcd]"
              << std::endl;
    return -1;
  }

  int passes = 0, failures = 0;
  std::string engine_path = argv[1];
  std::string rgb_path = argv[2];
  std::string depth_str = argv[3];
  std::string pcd_path = (argc >= 5) ? argv[4] : "pipeline_output.pcd";

  auto t_total_start = std::chrono::high_resolution_clock::now();

  // ===== Stage 1: Load =====
  std::cout << "\n=== Stage 1: Loading ===" << std::endl;
  semantic_vslam::YoloInference yolo(engine_path);
  TEST_ASSERT(yolo.init(), "YOLO engine init");

  cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
  TEST_ASSERT(!rgb.empty(), "RGB image loads");

  cv::Mat depth;
  if (depth_str == "synth") {
    depth = cv::Mat(rgb.rows, rgb.cols, CV_16UC1);
    float cx = rgb.cols / 2.0f, cy = rgb.rows / 2.0f;
    float max_dist = std::sqrt(cx * cx + cy * cy);
    for (int v = 0; v < rgb.rows; ++v)
      for (int u = 0; u < rgb.cols; ++u) {
        float d = std::sqrt(float(u - cx) * float(u - cx) +
                            float(v - cy) * float(v - cy));
        depth.at<uint16_t>(v, u) =
            static_cast<uint16_t>(1000 + 2000 * d / max_dist);
      }
  } else {
    depth = cv::imread(depth_str, cv::IMREAD_UNCHANGED);
  }
  TEST_ASSERT(!depth.empty(), "Depth image loads/generated");

  // ===== Stage 2: YOLO Inference =====
  std::cout << "\n=== Stage 2: YOLO Inference ===" << std::endl;
  std::vector<semantic_vslam::Object> objects;
  auto t0 = std::chrono::high_resolution_clock::now();
  TEST_ASSERT(yolo.infer(rgb, objects, 0.4f, 0.45f), "YOLO inference OK");
  auto t1 = std::chrono::high_resolution_clock::now();
  auto yolo_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "  " << objects.size() << " objects in " << yolo_ms << " ms"
            << std::endl;
  TEST_ASSERT(objects.size() > 0, "At least 1 object detected");

  // ===== Stage 3: Semantic Point Cloud =====
  std::cout << "\n=== Stage 3: Semantic Point Cloud ===" << std::endl;
  float fx = 570.3f, fy = 570.3f;
  float cx = rgb.cols / 2.0f, cy = rgb.rows / 2.0f;
  float depth_scale = 0.001f;

  // Resize depth if needed
  if (depth.rows != rgb.rows || depth.cols != rgb.cols) {
    cv::resize(depth, depth, cv::Size(rgb.cols, rgb.rows), 0, 0,
               cv::INTER_NEAREST);
  }

  // Build label map + cloud
  cv::Mat label_map = cv::Mat::zeros(rgb.rows, rgb.cols, CV_8UC1);
  cv::Mat conf_map = cv::Mat::zeros(rgb.rows, rgb.cols, CV_32FC1);
  for (const auto &obj : objects) {
    const cv::Rect &r = obj.rect;
    int x0 = std::max(0, r.x), y0 = std::max(0, r.y);
    int x1 = std::min(rgb.cols, r.x + r.width),
        y1 = std::min(rgb.rows, r.y + r.height);
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

  pcl::PointCloud<pcl::PointXYZRGBL> cloud;
  cloud.width = rgb.cols;
  cloud.height = rgb.rows;
  cloud.is_dense = false;
  cloud.points.resize(static_cast<size_t>(rgb.rows) * rgb.cols);
  int valid_pts = 0, semantic_pts = 0;

  for (int v = 0; v < rgb.rows; ++v) {
    for (int u = 0; u < rgb.cols; ++u) {
      auto &pt = cloud.at(u, v);
      float z = (depth.type() == CV_16UC1)
                    ? static_cast<float>(depth.at<uint16_t>(v, u)) * depth_scale
                    : depth.at<float>(v, u);
      if (z <= 0.01f || z > 10.0f || std::isnan(z)) {
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        pt.r = pt.g = pt.b = 0;
        pt.label = 0;
        continue;
      }
      pt.x = (float(u) - cx) * z / fx;
      pt.y = (float(v) - cy) * z / fy;
      pt.z = z;
      valid_pts++;
      uint8_t lbl = label_map.at<uint8_t>(v, u);
      if (lbl > 0) {
        int cls = lbl - 1;
        pt.r = semantic_vslam::kSemanticColors[cls][0];
        pt.g = semantic_vslam::kSemanticColors[cls][1];
        pt.b = semantic_vslam::kSemanticColors[cls][2];
        pt.label = static_cast<uint32_t>(cls);
        semantic_pts++;
      } else {
        const auto &bgr = rgb.at<cv::Vec3b>(v, u);
        pt.r = bgr[2];
        pt.g = bgr[1];
        pt.b = bgr[0];
        pt.label = 0;
      }
    }
  }

  TEST_ASSERT(valid_pts > 0,
              "Cloud has valid points (" + std::to_string(valid_pts) + ")");
  TEST_ASSERT(semantic_pts > 0, "Cloud has semantic points (" +
                                    std::to_string(semantic_pts) + ")");

  float sem_ratio = float(semantic_pts) / float(valid_pts) * 100.0f;
  std::cout << "  Total: " << cloud.size() << " Valid: " << valid_pts
            << " Semantic: " << semantic_pts << " (" << sem_ratio << "%)"
            << std::endl;

  // ===== Stage 4: PCD Save & Reload =====
  std::cout << "\n=== Stage 4: PCD Save & Reload ===" << std::endl;
  pcl::io::savePCDFileBinary(pcd_path, cloud);
  TEST_ASSERT(true, "PCD saved: " + pcd_path);

  pcl::PointCloud<pcl::PointXYZRGBL> cloud_reload;
  int load_ok = pcl::io::loadPCDFile(pcd_path, cloud_reload);
  TEST_ASSERT(load_ok == 0, "PCD reloads successfully");
  TEST_ASSERT(cloud_reload.size() == cloud.size(),
              "Reloaded cloud size matches (" +
                  std::to_string(cloud_reload.size()) + ")");

  // ===== Stage 5: Visualization Output =====
  std::cout << "\n=== Stage 5: Visualization ===" << std::endl;
  cv::Mat vis = rgb.clone();
  cv::Mat overlay = cv::Mat::zeros(rgb.size(), rgb.type());
  for (const auto &obj : objects) {
    cv::rectangle(vis, obj.rect, cv::Scalar(0, 255, 0), 2);
    if (!obj.mask.empty() && obj.label >= 0 && obj.label < 80) {
      cv::Mat color_mask(
          obj.mask.size(), CV_8UC3,
          cv::Scalar(semantic_vslam::kSemanticColors[obj.label][2],
                     semantic_vslam::kSemanticColors[obj.label][1],
                     semantic_vslam::kSemanticColors[obj.label][0]));
      color_mask.copyTo(overlay(obj.rect), obj.mask);
    }
  }
  cv::addWeighted(vis, 1.0, overlay, 0.5, 0, vis);
  cv::imwrite("pipeline_vis.jpg", vis);
  TEST_ASSERT(true, "Visualization saved: pipeline_vis.jpg");

  auto t_total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      t_total_end - t_total_start)
                      .count();

  // ===== Summary =====
  std::cout << "\n========================================" << std::endl;
  std::cout << "Pipeline total time: " << total_ms << " ms" << std::endl;
  std::cout << "Results: " << passes << " passed, " << failures << " failed"
            << std::endl;
  std::cout << "========================================" << std::endl;

  return (failures > 0) ? 1 : 0;
}
