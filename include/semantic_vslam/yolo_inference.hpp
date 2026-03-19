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
// YOLOv8-seg 推理类
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
  /// @param img         输入图像 (任意分辨率, BGR 或 RGB)
  /// @param objects     输出: 检测 + 分割结果
  /// @param conf_thresh 置信度阈值
  /// @param iou_thresh  NMS IoU 阈值
  /// @param is_rgb      true = 输入是 RGB (跳过 BGR→RGB); false = 输入是 BGR
  bool infer(const cv::Mat &img, std::vector<Object> &objects,
             float conf_thresh = 0.4f, float iou_thresh = 0.45f,
             bool is_rgb = false);

private:
  // GPU 预处理 (已弃用 CPU preProcess, 由 infer() 中 cudaPreprocess 替代)
  // void preProcess(const cv::Mat &img, float &out_scale, int &out_pad_x, int &out_pad_y);

  std::string engine_path_;
  Logger logger_;

  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // 零拷贝内存: CPU/GPU 共享同一块物理内存 (Jetson 统一内存架构)
  // mapped_* 为 CPU 端地址, dev_* 为 GPU 端地址, 指向同一物理页
  float *mapped_input_ = nullptr;   // CPU 端输入地址
  float *mapped_output0_ = nullptr; // CPU 端输出0地址 [116, 8400]
  float *mapped_output1_ = nullptr; // CPU 端输出1地址 [32, 160, 160]
  void  *dev_input_ = nullptr;      // GPU 端输入地址
  void  *dev_output0_ = nullptr;    // GPU 端输出0地址
  void  *dev_output1_ = nullptr;    // GPU 端输出1地址

  // GPU 预处理: 原始 BGR 图像的零拷贝缓冲区
  uint8_t *mapped_img_src_ = nullptr;  // CPU 端: 写入原始 BGR 数据
  void    *dev_img_src_ = nullptr;     // GPU 端: 预处理 kernel 读取
  size_t   img_buf_size_ = 0;          // 当前缓冲区大小 (字节)

  // GPU 掩码解码: 零拷贝缓冲区
  float *mapped_mask_coeffs_ = nullptr;  // CPU 写入 NMS 后的 mask 系数 [N, 32]
  void  *dev_mask_coeffs_ = nullptr;     // GPU 端用于 kernel 读取
  float *mapped_mask_out_ = nullptr;     // CPU 读取解码后的全分辨率掩码 [N, 160, 160]
  void  *dev_mask_out_ = nullptr;        // GPU 端用于 kernel 写入
  int    mask_buf_capacity_ = 0;         // 当前 mask 缓冲区可容纳的最大目标数

  cudaStream_t stream_ = nullptr;
};

} // namespace semantic_vslam
