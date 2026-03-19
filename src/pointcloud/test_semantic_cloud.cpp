/*
 * test_semantic_cloud.cpp
 *
 * 独立单元测试: 读取 RGB + Depth 图片 → YOLOv8-seg 推理 → 生成语义点云 → 保存为
 * .pcd
 *
 * 用法:
 *   ./test_semantic_cloud <engine_path> <rgb_image> <depth_image> [output.pcd]
 *
 * 此测试不依赖 ROS2, 可独立运行。
 * 需要一对对齐的 RGB + Depth 图片（Depth 为 16UC1 毫米制 PNG）。
 *
 * 如果暂时没有 depth 图, 会用一个合成的虚拟 depth 来演示点云结构。
 */

#include "semantic_vslam/yolo_inference.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 复用 semantic_cloud_node 中定义的颜色表
static const std::array<std::array<uint8_t, 3>, 80> kSemanticColors = {{
    {{255, 0, 0}},    //  0 person
    {{0, 255, 0}},    //  1 bicycle
    {{0, 0, 255}},    //  2 car
    {{255, 255, 0}},  //  3 motorcycle
    {{255, 0, 255}},  //  4 airplane
    {{0, 255, 255}},  //  5 bus
    {{128, 0, 0}},    //  6 train
    {{0, 128, 0}},    //  7 truck
    {{0, 0, 128}},    //  8 boat
    {{128, 128, 0}},  //  9 traffic light
    {{128, 0, 128}},  // 10 fire hydrant
    {{0, 128, 128}},  // 11 stop sign
    {{64, 0, 0}},     // 12 parking meter
    {{0, 64, 0}},     // 13 bench
    {{0, 0, 64}},     // 14 bird
    {{64, 64, 0}},    // 15 cat
    {{64, 0, 64}},    // 16 dog
    {{0, 64, 64}},    // 17 horse
    {{192, 0, 0}},    // 18 sheep
    {{0, 192, 0}},    // 19 cow
    {{0, 0, 192}},    // 20 elephant
    {{192, 192, 0}},  // 21 bear
    {{192, 0, 192}},  // 22 zebra
    {{0, 192, 192}},  // 23 giraffe
    {{128, 64, 0}},   // 24 backpack
    {{0, 128, 64}},   // 25 umbrella
    {{64, 0, 128}},   // 26 handbag
    {{64, 128, 0}},   // 27 tie
    {{128, 0, 64}},   // 28 suitcase
    {{0, 64, 128}},   // 29 frisbee
    {{200, 100, 50}}, // 30-79: 其他类别
    {{50, 200, 100}},  {{100, 50, 200}}, {{150, 100, 50}}, {{50, 150, 100}},
    {{100, 50, 150}},  {{200, 150, 50}}, {{50, 200, 150}}, {{150, 50, 200}},
    {{100, 200, 50}},  {{50, 100, 200}}, {{200, 50, 100}}, {{80, 160, 240}},
    {{240, 160, 80}},  {{160, 80, 240}}, {{80, 240, 160}}, {{240, 80, 160}},
    {{160, 240, 80}},  {{120, 60, 180}}, {{60, 180, 120}}, {{180, 120, 60}},
    {{60, 120, 180}},  {{180, 60, 120}}, {{120, 180, 60}}, {{140, 70, 210}},
    {{70, 210, 140}},  {{210, 140, 70}}, {{70, 140, 210}}, {{210, 70, 140}},
    {{140, 210, 70}},  {{90, 45, 180}},  {{45, 180, 90}},  {{180, 90, 45}},
    {{45, 90, 180}},   {{180, 45, 90}},  {{90, 180, 45}},  {{110, 55, 165}},
    {{55, 165, 110}},  {{165, 110, 55}}, {{55, 110, 165}}, {{165, 55, 110}},
    {{110, 165, 55}},  {{130, 65, 195}}, {{65, 195, 130}}, {{195, 130, 65}},
    {{65, 130, 195}},  {{195, 65, 130}}, {{130, 195, 65}}, {{100, 150, 200}},
    {{200, 100, 150}},
}};

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " <engine_path> <rgb_image> <depth_image_or_'synth'> [output.pcd]"
        << std::endl;
    std::cerr << "  depth_image: 16UC1 PNG (mm), or 'synth' for synthetic depth"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string rgb_path = argv[2];
  std::string depth_path = argv[3];
  std::string output_path = (argc >= 5) ? argv[4] : "semantic_cloud.pcd";

  // --- 加载 YOLO ---
  semantic_vslam::YoloInference yolo(engine_path);
  if (!yolo.init()) {
    std::cerr << "[Error] Failed to init YOLO from: " << engine_path
              << std::endl;
    return -1;
  }
  std::cout << "YOLO loaded OK" << std::endl;

  // --- 读取 RGB ---
  cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
  if (rgb.empty()) {
    std::cerr << "[Error] Cannot read RGB: " << rgb_path << std::endl;
    return -1;
  }

  // --- 读取 Depth / 合成 ---
  cv::Mat depth;
  bool use_synth = (depth_path == "synth");
  if (use_synth) {
    // 合成虚拟深度: 离中心越远越"远"，范围 1~3 米
    std::cout << "[Info] Using synthetic depth map" << std::endl;
    depth = cv::Mat(rgb.rows, rgb.cols, CV_16UC1);
    float cx = rgb.cols / 2.0f, cy = rgb.rows / 2.0f;
    float max_dist = std::sqrt(cx * cx + cy * cy);
    for (int v = 0; v < rgb.rows; ++v) {
      for (int u = 0; u < rgb.cols; ++u) {
        float d = std::sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy));
        float z_m = 1.0f + 2.0f * d / max_dist; // 1~3 m
        depth.at<uint16_t>(v, u) = static_cast<uint16_t>(z_m * 1000);
      }
    }
  } else {
    depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
      std::cerr << "[Error] Cannot read depth: " << depth_path << std::endl;
      return -1;
    }
  }
  std::cout << "RGB: " << rgb.cols << "x" << rgb.rows
            << "  Depth: " << depth.cols << "x" << depth.rows
            << " type=" << depth.type() << std::endl;

  // --- YOLO 推理 ---
  std::vector<semantic_vslam::Object> objects;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (!yolo.infer(rgb, objects, 0.4f, 0.45f)) {
    std::cerr << "[Error] YOLO inference failed" << std::endl;
    return -1;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout
      << "YOLO: " << objects.size() << " objects, "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << " ms" << std::endl;

  for (const auto &obj : objects) {
    std::cout << "  class=" << obj.label << " prob=" << obj.prob << " rect=("
              << obj.rect.x << "," << obj.rect.y << "," << obj.rect.width << ","
              << obj.rect.height << ")" << std::endl;
  }

  // --- 构建语义标签图 ---
  cv::Mat label_map = cv::Mat::zeros(rgb.rows, rgb.cols, CV_8UC1);
  cv::Mat conf_map = cv::Mat::zeros(rgb.rows, rgb.cols, CV_32FC1);

  for (const auto &obj : objects) {
    const cv::Rect &r = obj.rect;
    int x0 = std::max(0, r.x);
    int y0 = std::max(0, r.y);
    int x1 = std::min(rgb.cols, r.x + r.width);
    int y1 = std::min(rgb.rows, r.y + r.height);

    if (x0 >= x1 || y0 >= y1 || obj.mask.empty())
      continue;

    for (int y = y0; y < y1; ++y) {
      const uint8_t *mask_row = obj.mask.ptr<uint8_t>(y - r.y);
      uint8_t *label_row = label_map.ptr<uint8_t>(y);
      float *conf_row = conf_map.ptr<float>(y);
      for (int x = x0; x < x1; ++x) {
        int mx = x - r.x;
        if (mx >= 0 && mx < obj.mask.cols && mask_row[mx] > 0) {
          if (obj.prob > conf_row[x]) {
            label_row[x] = static_cast<uint8_t>(obj.label + 1);
            conf_row[x] = obj.prob;
          }
        }
      }
    }
  }

  // --- 用默认内参 (Astra Pro 近似) 或用户可改 ---
  // Astra Pro 640x480 典型值
  float fx = 570.3f, fy = 570.3f;
  float cx_cam = rgb.cols / 2.0f;
  float cy_cam = rgb.rows / 2.0f;
  float depth_scale = 0.001f; // 16UC1 mm → m

  std::cout << "Using intrinsics: fx=" << fx << " fy=" << fy << " cx=" << cx_cam
            << " cy=" << cy_cam << std::endl;

  // --- 生成点云 ---
  pcl::PointCloud<pcl::PointXYZRGBL> cloud;
  cloud.width = rgb.cols;
  cloud.height = rgb.rows;
  cloud.is_dense = false;
  cloud.points.resize(static_cast<size_t>(rgb.rows) * rgb.cols);

  const float inv_fx = 1.0f / fx;
  const float inv_fy = 1.0f / fy;
  const bool is_16u = (depth.type() == CV_16UC1);
  int valid_pts = 0;
  int semantic_pts = 0;

  for (int v = 0; v < rgb.rows; ++v) {
    for (int u = 0; u < rgb.cols; ++u) {
      pcl::PointXYZRGBL &pt = cloud.at(u, v);

      float z = 0.0f;
      if (is_16u) {
        z = static_cast<float>(depth.at<uint16_t>(v, u)) * depth_scale;
      } else {
        z = depth.at<float>(v, u);
      }

      if (z <= 0.01f || z > 10.0f || std::isnan(z)) {
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        pt.r = pt.g = pt.b = 0;
        pt.label = 0;
        continue;
      }

      pt.x = (static_cast<float>(u) - cx_cam) * z * inv_fx;
      pt.y = (static_cast<float>(v) - cy_cam) * z * inv_fy;
      pt.z = z;
      valid_pts++;

      uint8_t lbl = label_map.at<uint8_t>(v, u);
      if (lbl > 0) {
        int cls = lbl - 1;
        if (cls >= 0 && cls < 80) {
          pt.r = kSemanticColors[cls][0];
          pt.g = kSemanticColors[cls][1];
          pt.b = kSemanticColors[cls][2];
        }
        pt.label = static_cast<uint32_t>(cls);
        semantic_pts++;
      } else {
        const cv::Vec3b &bgr = rgb.at<cv::Vec3b>(v, u);
        pt.r = bgr[2];
        pt.g = bgr[1];
        pt.b = bgr[0];
        pt.label = 0;
      }
    }
  }

  std::cout << "Point cloud: " << cloud.size() << " total, " << valid_pts
            << " valid, " << semantic_pts << " semantic" << std::endl;

  // --- 保存 PCD ---
  pcl::io::savePCDFileBinary(output_path, cloud);
  std::cout << "Saved: " << output_path << std::endl;

  // --- 同时保存一张语义可视化图 ---
  cv::Mat vis = rgb.clone();
  cv::Mat overlay = cv::Mat::zeros(rgb.size(), rgb.type());
  for (const auto &obj : objects) {
    cv::rectangle(vis, obj.rect, cv::Scalar(0, 255, 0), 2);
    if (!obj.mask.empty() && obj.label >= 0 && obj.label < 80) {
      cv::Mat color_mask(obj.mask.size(), CV_8UC3,
                         cv::Scalar(kSemanticColors[obj.label][2],
                                    kSemanticColors[obj.label][1],
                                    kSemanticColors[obj.label][0]));
      color_mask.copyTo(overlay(obj.rect), obj.mask);
    }
  }
  cv::addWeighted(vis, 1.0, overlay, 0.5, 0, vis);
  cv::imwrite("semantic_vis.jpg", vis);
  std::cout << "Saved: semantic_vis.jpg" << std::endl;

  return 0;
}
