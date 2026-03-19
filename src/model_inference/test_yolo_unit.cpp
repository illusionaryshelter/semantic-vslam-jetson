/*
 * test_yolo_unit.cpp
 *
 * YOLOv8-seg 单元测试
 * 独立可执行文件，不依赖 ROS2
 *
 * 测试项:
 *   1. 引擎加载
 *   2. 推理执行
 *   3. 检测结果: 数量、置信度、bbox 范围
 *   4. 掩码: 非空、尺寸匹配 bbox
 *
 * 用法: ./test_yolo_unit <engine_path> <image_path>
 */

#include "semantic_vslam/yolo_inference.hpp"
#include <cassert>
#include <chrono>
#include <iostream>

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
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>"
              << std::endl;
    return -1;
  }

  int passes = 0, failures = 0;

  std::string engine_path = argv[1];
  std::string image_path = argv[2];

  // ========== Test 1: 引擎加载 ==========
  std::cout << "\n=== Test 1: Engine Loading ===" << std::endl;
  semantic_vslam::YoloInference yolo(engine_path);
  bool init_ok = yolo.init();
  TEST_ASSERT(init_ok, "YOLO engine loads successfully");

  if (!init_ok) {
    std::cerr << "Cannot continue tests without engine." << std::endl;
    return 1;
  }

  // ========== Test 2: 图像读取 ==========
  std::cout << "\n=== Test 2: Image Loading ===" << std::endl;
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  TEST_ASSERT(!img.empty(), "Test image loads successfully");
  TEST_ASSERT(img.channels() == 3, "Image has 3 channels (BGR)");
  std::cout << "  Image size: " << img.cols << "x" << img.rows << std::endl;

  // ========== Test 3: 推理执行 ==========
  std::cout << "\n=== Test 3: Inference Execution ===" << std::endl;
  std::vector<semantic_vslam::Object> objects;
  auto t0 = std::chrono::high_resolution_clock::now();
  bool infer_ok = yolo.infer(img, objects, 0.3f, 0.45f);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  TEST_ASSERT(infer_ok, "Inference returns true");
  TEST_ASSERT(objects.size() > 0, "At least 1 detection found");
  std::cout << "  Detected: " << objects.size() << " objects in " << ms << " ms"
            << std::endl;

  // ========== Test 4: 检测结果验证 ==========
  std::cout << "\n=== Test 4: Detection Validation ===" << std::endl;
  for (size_t i = 0; i < objects.size(); ++i) {
    const auto &obj = objects[i];
    std::cout << "  Object " << i << ": class=" << obj.label
              << " prob=" << obj.prob << " rect=(" << obj.rect.x << ","
              << obj.rect.y << "," << obj.rect.width << "," << obj.rect.height
              << ")"
              << " mask=" << obj.mask.cols << "x" << obj.mask.rows << std::endl;

    // 置信度应在 (0, 1] 范围
    TEST_ASSERT(obj.prob > 0.0f && obj.prob <= 1.0f,
                "Object " + std::to_string(i) + " confidence in (0,1]");

    // 类别应在 [0, 79] 范围
    TEST_ASSERT(obj.label >= 0 && obj.label < 80,
                "Object " + std::to_string(i) + " class in [0,79]");

    // bbox 应在图像范围内
    TEST_ASSERT(obj.rect.x >= 0 && obj.rect.y >= 0,
                "Object " + std::to_string(i) + " bbox origin >= 0");
    TEST_ASSERT(obj.rect.x + obj.rect.width <= img.cols,
                "Object " + std::to_string(i) + " bbox right <= image width");
    TEST_ASSERT(obj.rect.y + obj.rect.height <= img.rows,
                "Object " + std::to_string(i) + " bbox bottom <= image height");
    TEST_ASSERT(obj.rect.width > 0 && obj.rect.height > 0,
                "Object " + std::to_string(i) + " bbox has positive size");

    // 掩码应非空且与 bbox 大小匹配
    TEST_ASSERT(!obj.mask.empty(),
                "Object " + std::to_string(i) + " mask is not empty");
    TEST_ASSERT(obj.mask.cols == obj.rect.width &&
                    obj.mask.rows == obj.rect.height,
                "Object " + std::to_string(i) + " mask size matches bbox");
    TEST_ASSERT(obj.mask.type() == CV_8UC1,
                "Object " + std::to_string(i) + " mask is CV_8UC1");

    // 掩码应至少包含一些非零像素
    int nz = cv::countNonZero(obj.mask);
    TEST_ASSERT(nz > 0, "Object " + std::to_string(i) +
                            " mask has non-zero pixels (" + std::to_string(nz) +
                            ")");
  }

  // ========== Test 5: 多次推理一致性 ==========
  std::cout << "\n=== Test 5: Inference Consistency ===" << std::endl;
  std::vector<semantic_vslam::Object> objects2;
  yolo.infer(img, objects2, 0.3f, 0.45f);
  TEST_ASSERT(objects.size() == objects2.size(),
              "Second inference produces same detection count");

  // ========== 总结 ==========
  std::cout << "\n========================================" << std::endl;
  std::cout << "Results: " << passes << " passed, " << failures << " failed"
            << std::endl;
  std::cout << "========================================" << std::endl;

  return (failures > 0) ? 1 : 0;
}
