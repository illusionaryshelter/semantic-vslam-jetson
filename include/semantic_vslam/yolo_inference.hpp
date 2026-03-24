#pragma once

#include <NvInfer.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace semantic_vslam {

// ---------------------------------------------------------------------------
// 张量/模型超参数 (与 yolov8n-seg 导出的 .engine / .onnx 对齐)
// ---------------------------------------------------------------------------
static constexpr int kINPUT_H = 640;
static constexpr int kINPUT_W = 640;
static constexpr int kNC = 80;        // COCO 类别数
static constexpr int kNM = 32;        // 掩码系数维度
static constexpr int kANCHORS = 8400; // 特征图候选框总数 (80^2 + 40^2 + 20^2)
static constexpr int kROWS = 4 + kNC + kNM; // 116
static constexpr int kMASK_H = 160;
static constexpr int kMASK_W = 160;

// ---------------------------------------------------------------------------
// 检测结果结构体
// ---------------------------------------------------------------------------
struct Object {
  cv::Rect rect; // 在原图坐标系中的边界框
  int label;     // 类别 id
  float prob;    // 置信度
  cv::Mat mask;  // CV_8UC1, 255/0, 与 rect 大小匹配
};

// ---------------------------------------------------------------------------
// TRT Logger
// ---------------------------------------------------------------------------
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override;
};

// ---------------------------------------------------------------------------
// YOLOv8-seg 推理类 (独显版: cudaMalloc + 显式 cudaMemcpy)
// ---------------------------------------------------------------------------
class YoloInference {
public:
  explicit YoloInference(const std::string &engine_path);
  ~YoloInference();

  // 不可拷贝
  YoloInference(const YoloInference &) = delete;
  YoloInference &operator=(const YoloInference &) = delete;

  /// 加载 .engine, 分配 CUDA 内存
  bool init();

  /// 对单张图像执行推理
  bool infer(const cv::Mat &img, std::vector<Object> &objects,
             float conf_thresh = 0.4f, float iou_thresh = 0.45f,
             bool is_rgb = false);

private:
  std::string engine_path_;
  Logger logger_;

  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // 独显模式: host (CPU, malloc) 和 device (GPU, cudaMalloc) 分开
  // TRT 输入/输出
  float *h_input_ = nullptr;     // host: CPU 写入预处理结果
  void  *d_input_ = nullptr;     // device: TRT/kernel 使用
  float *h_output0_ = nullptr;   // host: CPU 读取检测输出
  void  *d_output0_ = nullptr;   // device: TRT 写入
  float *h_output1_ = nullptr;   // host: CPU 读取 proto
  void  *d_output1_ = nullptr;   // device: TRT 写入

  // GPU 预处理: 原始 BGR 图像缓冲区
  uint8_t *h_img_src_ = nullptr;  // host: CPU 写入原始图像
  void    *d_img_src_ = nullptr;  // device: 预处理 kernel 读取
  size_t   img_buf_size_ = 0;

  // GPU 掩码解码
  float *h_mask_coeffs_ = nullptr;  // host: CPU 写入 NMS 后的 mask 系数
  void  *d_mask_coeffs_ = nullptr;  // device: kernel 读取
  float *h_mask_out_ = nullptr;     // host: CPU 读取解码后掩码
  void  *d_mask_out_ = nullptr;     // device: kernel 写入
  int    mask_buf_capacity_ = 0;

  cudaStream_t stream_ = nullptr;
};

} // namespace semantic_vslam
