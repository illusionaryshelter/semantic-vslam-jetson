#include "semantic_vslam/yolo_inference.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string image_path = argv[2];

  semantic_vslam::YoloInference yolo(engine_path);
  if (!yolo.init()) {
    std::cerr << "Failed to initialize YOLO model" << std::endl;
    return -1;
  }
  std::cout << "YOLO model initialized successfully" << std::endl;

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Failed to read image: " << image_path << std::endl;
    return -1;
  }

  std::vector<semantic_vslam::Object> objects;

  // Warm up
  yolo.infer(img, objects);

  auto start = std::chrono::high_resolution_clock::now();
  if (!yolo.infer(img, objects, 0.4f, 0.45f)) {
    std::cerr << "Inference failed" << std::endl;
    return -1;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Inference took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;

  std::cout << "Detected " << objects.size() << " objects" << std::endl;

  cv::Mat resultImg = img.clone();
  cv::Mat maskOverlay = cv::Mat::zeros(img.size(), img.type());

  for (const auto &obj : objects) {
    std::cout << "Object class: " << obj.label << " Prob: " << obj.prob
              << " Rect: (" << obj.rect.x << ", " << obj.rect.y << ", "
              << obj.rect.width << ", " << obj.rect.height << ")" << std::endl;

    cv::rectangle(resultImg, obj.rect, cv::Scalar(0, 255, 0), 2);

    if (!obj.mask.empty()) {
      cv::Mat coloredMask(obj.mask.size(), CV_8UC3, cv::Scalar(0, 0, 255));
      coloredMask.copyTo(maskOverlay(obj.rect), obj.mask);
    }
  }

  cv::addWeighted(resultImg, 1.0, maskOverlay, 0.5, 0, resultImg);
  cv::imwrite("test_output.jpg", resultImg);
  std::cout << "Saved test_output.jpg" << std::endl;

  return 0;
}
