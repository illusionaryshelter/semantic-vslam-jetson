/*
 * test_yolo_realtime.cpp
 *
 * 独立实时测试: Astra Pro UVC 摄像头 + YOLOv8-seg 推理
 * 不依赖 ROS2，使用 OpenCV VideoCapture 读取摄像头
 *
 * 功能:
 *   - 打开 Astra Pro 的 UVC 彩色摄像头 (/dev/video0 或指定设备)
 *   - 对每帧执行 YOLOv8-seg 推理
 *   - 显示 原始帧 | 语义分割叠加 的双窗口对比
 *   - 实时输出 FPS、检测数量、类别信息
 *
 * 用法:
 *   ./test_yolo_realtime <engine_path> [camera_index=0] [conf_thresh=0.4]
 *
 * 按 'q' 或 ESC 退出
 */

#include "semantic_vslam/semantic_colors.hpp"
#include "semantic_vslam/yolo_inference.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// COCO 80 类名称
static const char *kCOCO_NAMES[] = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair dryer",    "toothbrush"};

// ---------------------------------------------------------------------------
// 在图像上绘制分割掩码和边界框
// ---------------------------------------------------------------------------
static cv::Mat
drawSegmentation(const cv::Mat &frame,
                 const std::vector<semantic_vslam::Object> &objs) {
  cv::Mat vis = frame.clone();
  cv::Mat overlay = vis.clone();

  for (const auto &obj : objs) {
    // 掩码半透明叠加
    const auto &c = semantic_vslam::kSemanticColors[obj.label % 80];
    cv::Scalar color(c[2], c[1], c[0]); // BGR

    if (!obj.mask.empty() && obj.rect.width > 0 && obj.rect.height > 0) {
      // 安全裁剪 ROI
      cv::Rect roi = obj.rect & cv::Rect(0, 0, vis.cols, vis.rows);
      if (roi.area() > 0) {
        cv::Mat mask_roi;
        // mask 大小与 obj.rect 匹配，截取需要的部分
        int dx = roi.x - obj.rect.x;
        int dy = roi.y - obj.rect.y;
        cv::Rect mask_crop(dx, dy, roi.width, roi.height);
        mask_crop &= cv::Rect(0, 0, obj.mask.cols, obj.mask.rows);
        if (mask_crop.area() > 0) {
          mask_roi = obj.mask(mask_crop);
          overlay(roi).setTo(color, mask_roi);
        }
      }
    }

    // 边界框
    cv::rectangle(vis, obj.rect, color, 2);

    // 标签文字
    char label_buf[128];
    const char *name =
        (obj.label >= 0 && obj.label < 80) ? kCOCO_NAMES[obj.label] : "?";
    snprintf(label_buf, sizeof(label_buf), "%s %.0f%%", name, obj.prob * 100);

    int baseline = 0;
    cv::Size text_sz =
        cv::getTextSize(label_buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::Point text_org(obj.rect.x,
                       std::max(obj.rect.y - 4, text_sz.height + 2));

    // 背景条
    cv::rectangle(
        vis, cv::Point(text_org.x, text_org.y - text_sz.height - 2),
        cv::Point(text_org.x + text_sz.width, text_org.y + baseline + 2), color,
        -1);
    cv::putText(vis, label_buf, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
  }

  // 叠加掩码 (alpha=0.45)
  cv::addWeighted(overlay, 0.45, vis, 0.55, 0, vis);
  return vis;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0]
              << " <engine_path> [camera_index=0] [conf_thresh=0.4]\n"
              << "\n"
              << "示例:\n"
              << "  " << argv[0] << " models/yolov8n-seg.engine\n"
              << "  " << argv[0] << " models/yolov8n-seg.engine 0 0.5\n"
              << "\n"
              << "按 'q' 或 ESC 退出\n";
    return 1;
  }

  const std::string engine_path = argv[1];
  int cam_index = (argc >= 3) ? std::atoi(argv[2]) : 0;
  float conf_thresh = (argc >= 4) ? std::atof(argv[3]) : 0.4f;

  // ---- 1. 加载 YOLO 模型 ----
  std::cout << "[INFO] Loading YOLO engine: " << engine_path << std::endl;
  semantic_vslam::YoloInference yolo(engine_path);
  if (!yolo.init()) {
    std::cerr << "[FAIL] Cannot load YOLO engine: " << engine_path << std::endl;
    return 1;
  }
  std::cout << "[PASS] YOLO engine loaded\n";

  // ---- 2. 打开摄像头 ----
  std::cout << "[INFO] Opening camera index " << cam_index
            << " (Astra Pro UVC)..." << std::endl;

  cv::VideoCapture cap;
  // 尝试 V4L2 后端 (Astra Pro UVC 摄像头)
  cap.open(cam_index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    // 回退到默认后端
    cap.open(cam_index);
  }
  if (!cap.isOpened()) {
    std::cerr << "[FAIL] Cannot open camera " << cam_index << "\n"
              << "  提示: \n"
              << "  1. 确保 Astra Pro 已连接且 ROS2 camera driver 未运行\n"
              << "  2. 检查 /dev/video* 是否存在\n"
              << "  3. 尝试: sudo chmod 666 /dev/video" << cam_index << "\n";
    return 1;
  }

  // 设置分辨率 (与 Astra Pro 默认一致)
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps_cam = cap.get(cv::CAP_PROP_FPS);
  std::cout << "[PASS] Camera opened: " << w << "x" << h << " @ " << fps_cam
            << " fps\n";

  // ---- 3. 实时推理循环 ----
  std::cout << "\n"
            << "========================================\n"
            << "  实时 YOLOv8-seg 分割测试\n"
            << "  conf_thresh = " << conf_thresh << "\n"
            << "  按 'q' 或 ESC 退出\n"
            << "========================================\n\n";

  int frame_count = 0;
  int total_detections = 0;
  double total_infer_ms = 0.0;
  auto t_start_all = std::chrono::steady_clock::now();

  while (true) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "[WARN] Empty frame, retrying...\n";
      continue;
    }

    // 推理
    auto t0 = std::chrono::steady_clock::now();
    std::vector<semantic_vslam::Object> objects;
    bool ok = yolo.infer(frame, objects, conf_thresh);
    auto t1 = std::chrono::steady_clock::now();

    double infer_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_infer_ms += infer_ms;
    frame_count++;

    if (!ok) {
      std::cerr << "[WARN] Inference failed on frame " << frame_count << "\n";
      continue;
    }

    total_detections += static_cast<int>(objects.size());

    // 绘制分割结果
    cv::Mat vis = drawSegmentation(frame, objects);

    // 在画面上添加信息条
    double avg_fps = frame_count * 1000.0 / total_infer_ms;
    char info_buf[256];
    snprintf(info_buf, sizeof(info_buf),
             "Frame:%d | Det:%zu | Infer:%.1fms | FPS:%.1f", frame_count,
             objects.size(), infer_ms, avg_fps);
    cv::putText(vis, info_buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    // 拼接 原始 | 分割 并显示
    cv::Mat original_resized, seg_resized;
    cv::resize(frame, original_resized, cv::Size(480, 360));
    cv::resize(vis, seg_resized, cv::Size(480, 360));
    cv::Mat side_by_side;
    cv::hconcat(original_resized, seg_resized, side_by_side);

    // 添加左/右标签
    cv::putText(side_by_side, "Original", cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::putText(side_by_side, "YOLOv8-seg", cv::Point(490, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

    cv::imshow("YOLO Realtime Test (q=quit)", side_by_side);

    // 终端输出 (每 30 帧)
    if (frame_count % 30 == 0) {
      std::printf("[INFO] Frame %d: %zu objects, %.1f ms, avg %.1f FPS\n",
                  frame_count, objects.size(), infer_ms, avg_fps);
      // 打印检测到的类别
      for (const auto &obj : objects) {
        const char *name = (obj.label >= 0 && obj.label < 80)
                               ? kCOCO_NAMES[obj.label]
                               : "unknown";
        std::printf("       -> %s (%.0f%%) [%d,%d %dx%d]\n", name,
                    obj.prob * 100, obj.rect.x, obj.rect.y, obj.rect.width,
                    obj.rect.height);
      }
    }

    // 按键检测
    int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) { // 'q' or ESC
      std::cout << "\n[INFO] User pressed quit key\n";
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();

  // ---- 4. 总结 ----
  auto t_end_all = std::chrono::steady_clock::now();
  double total_sec =
      std::chrono::duration<double>(t_end_all - t_start_all).count();
  double avg_infer = (frame_count > 0) ? total_infer_ms / frame_count : 0;
  double throughput = (total_sec > 0) ? frame_count / total_sec : 0;

  std::cout << "\n"
            << "========================================\n"
            << "  实时测试总结\n"
            << "========================================\n"
            << "  总帧数:       " << frame_count << "\n"
            << "  总运行时间:   " << std::fixed << std::setprecision(1)
            << total_sec << " s\n"
            << "  平均推理耗时: " << avg_infer << " ms/frame\n"
            << "  平均吞吐:     " << throughput << " FPS\n"
            << "  总检测数:     " << total_detections << "\n"
            << "  平均检测/帧:  "
            << (frame_count > 0 ? (double)total_detections / frame_count : 0)
            << "\n========================================\n";

  // PASS/FAIL 判定
  bool passed = (frame_count >= 10 && avg_infer < 500.0);
  if (passed) {
    std::cout << "\n[PASS] 实时推理测试通过: " << frame_count << " 帧, 平均 "
              << avg_infer << " ms\n";
  } else {
    std::cout << "\n[FAIL] 实时推理测试失败: "
              << "frames=" << frame_count << " avg_ms=" << avg_infer << "\n";
  }

  return passed ? 0 : 1;
}
