/*
 * yolo_inference.cpp
 *
 * YOLOv8-seg TensorRT 推理实现
 *
 * 张量布局 (Ultralytics ONNX export):
 *   images:  [1, 3,   640, 640]  - 输入
 *   output0: [1, 116, 8400]      - 检测头:  [cx, cy, w, h, 80个类别分数,
 * 32个掩码系数] output1: [1, 32,  160, 160]  - Proto 掩码特征图
 *
 * 掩码解码方式 (参考 tensorrtx/yolov8):
 *   对每个检测框，在160x160 proto图上的对应ROI区域内逐像素计算
 *   mask_coeff (32维) 与 proto (32 x 25600) 的点积，sigmoid后得到软掩码，
 *   再resize到原始bbox大小。
 */

#include "semantic_vslam/yolo_inference.hpp"
#include "semantic_vslam/cuda_preprocess.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

namespace semantic_vslam {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
void Logger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TRT] " << msg << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
YoloInference::YoloInference(const std::string &engine_path)
    : engine_path_(engine_path) {}

YoloInference::~YoloInference() {
  if (stream_)
    cudaStreamDestroy(stream_);
  // 零拷贝: 只需释放 host 端 mapped 内存 (同时释放 GPU 映射)
  if (mapped_input_)
    cudaFreeHost(mapped_input_);
  if (mapped_output0_)
    cudaFreeHost(mapped_output0_);
  if (mapped_output1_)
    cudaFreeHost(mapped_output1_);
  if (mapped_img_src_)
    cudaFreeHost(mapped_img_src_);
  if (mapped_mask_coeffs_)
    cudaFreeHost(mapped_mask_coeffs_);
  if (mapped_mask_out_)
    cudaFreeHost(mapped_mask_out_);
  delete context_;
  delete engine_;
  delete runtime_;
}

// ---------------------------------------------------------------------------
// init: 加载 TRT engine，分配 CUDA 内存
// ---------------------------------------------------------------------------
bool YoloInference::init() {
  try {
    // 读取 .engine 文件
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
      std::cerr << "[Error] Cannot open engine file: " << engine_path_
                << std::endl;
      return false;
    }
    file.seekg(0, file.end);
    size_t fsize = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> blob(fsize);
    file.read(blob.data(), fsize);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
      std::cerr << "[Error] createInferRuntime failed" << std::endl;
      return false;
    }

    engine_ = runtime_->deserializeCudaEngine(blob.data(), fsize);
    if (!engine_) {
      std::cerr << "[Error] deserializeCudaEngine failed" << std::endl;
      return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
      std::cerr << "[Error] createExecutionContext failed" << std::endl;
      return false;
    }

    // 计算各缓冲区尺寸 (float32)
    const size_t in_sz = 1 * 3 * kINPUT_H * kINPUT_W;   // 1×3×640×640
    const size_t out0_sz = 1 * kROWS * kANCHORS;        // 1×116×8400
    const size_t out1_sz = 1 * kNM * kMASK_H * kMASK_W; // 1×32×160×160

    // ---- 零拷贝内存分配 ----
    // cudaHostAllocMapped: CPU/GPU 共享同一物理内存页
    // 在 Jetson 统一内存架构上完全消除 cudaMemcpy 开销
    const unsigned int flags = cudaHostAllocMapped;

    if (cudaHostAlloc((void **)&mapped_input_, in_sz * sizeof(float), flags) !=
        cudaSuccess)
      return false;
    if (cudaHostAlloc((void **)&mapped_output0_, out0_sz * sizeof(float),
                      flags) != cudaSuccess)
      return false;
    if (cudaHostAlloc((void **)&mapped_output1_, out1_sz * sizeof(float),
                      flags) != cudaSuccess)
      return false;

    // 获取 GPU 端地址 (指向同一物理页)
    if (cudaHostGetDevicePointer(&dev_input_, mapped_input_, 0) != cudaSuccess)
      return false;
    if (cudaHostGetDevicePointer(&dev_output0_, mapped_output0_, 0) !=
        cudaSuccess)
      return false;
    if (cudaHostGetDevicePointer(&dev_output1_, mapped_output1_, 0) !=
        cudaSuccess)
      return false;

    if (cudaStreamCreate(&stream_) != cudaSuccess)
      return false;

    // 设置张量地址 (TRT 10 API) — 直接使用 GPU 映射地址
    context_->setTensorAddress("images", dev_input_);
    context_->setTensorAddress("output0", dev_output0_);
    context_->setTensorAddress("output1", dev_output1_);

    // ---- GPU 预处理: 预分配默认图像缓冲区 (640×480×3) ----
    const size_t default_img_sz = 640 * 480 * 3;
    if (cudaHostAlloc((void **)&mapped_img_src_, default_img_sz,
                      cudaHostAllocMapped) != cudaSuccess)
      return false;
    if (cudaHostGetDevicePointer(&dev_img_src_, mapped_img_src_, 0) !=
        cudaSuccess)
      return false;
    img_buf_size_ = default_img_sz;

    // ---- GPU 掩码解码: 预分配缓冲区 (32 个目标) ----
    const int default_mask_cap = 32;
    const size_t coeff_sz = default_mask_cap * kNM * sizeof(float);
    const size_t mask_sz = default_mask_cap * kMASK_H * kMASK_W * sizeof(float);
    if (cudaHostAlloc((void **)&mapped_mask_coeffs_, coeff_sz,
                      cudaHostAllocMapped) != cudaSuccess)
      return false;
    if (cudaHostGetDevicePointer(&dev_mask_coeffs_, mapped_mask_coeffs_, 0) !=
        cudaSuccess)
      return false;
    if (cudaHostAlloc((void **)&mapped_mask_out_, mask_sz,
                      cudaHostAllocMapped) != cudaSuccess)
      return false;
    if (cudaHostGetDevicePointer(&dev_mask_out_, mapped_mask_out_, 0) !=
        cudaSuccess)
      return false;
    mask_buf_capacity_ = default_mask_cap;

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[Exception] YoloInference::init: " << e.what() << std::endl;
    return false;
  }
}

// ---------------------------------------------------------------------------
// sigmoid 内联辅助
// ---------------------------------------------------------------------------
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// decodeMask
//   对于单个检测框，在 160×160 proto 图上对应 ROI 逐像素点积 + sigmoid，
//   再 resize 到 bbox 的实际像素尺寸，最后阈值化为二值掩码。
// ---------------------------------------------------------------------------
static void decodeMask(const std::vector<float> &mask_coeff, const float *proto,
                       const cv::Rect &rect_in_img, float scale, int pad_x,
                       int pad_y, int mask_w, int mask_h, cv::Mat &out_mask) {
  if (rect_in_img.width <= 0 || rect_in_img.height <= 0) {
    out_mask = cv::Mat::zeros(rect_in_img.height, rect_in_img.width, CV_8UC1);
    return;
  }

  const float inv4 = 0.25f;
  int rx0 = static_cast<int>((rect_in_img.x * scale + pad_x) * inv4);
  int ry0 = static_cast<int>((rect_in_img.y * scale + pad_y) * inv4);
  int rx1 = static_cast<int>(
      ((rect_in_img.x + rect_in_img.width) * scale + pad_x) * inv4);
  int ry1 = static_cast<int>(
      ((rect_in_img.y + rect_in_img.height) * scale + pad_y) * inv4);

  rx0 = std::max(0, std::min(rx0, mask_w - 1));
  ry0 = std::max(0, std::min(ry0, mask_h - 1));
  rx1 = std::max(1, std::min(rx1, mask_w));
  ry1 = std::max(1, std::min(ry1, mask_h));

  int roi_w = rx1 - rx0;
  int roi_h = ry1 - ry0;

  cv::Mat roi_mask(roi_h, roi_w, CV_32FC1);
  const int proto_row = mask_w * mask_h;

  for (int y = 0; y < roi_h; ++y) {
    float *row_ptr = roi_mask.ptr<float>(y);
    for (int x = 0; x < roi_w; ++x) {
      int px = rx0 + x;
      int py = ry0 + y;
      float val = 0.0f;
      for (int j = 0; j < 32; ++j) {
        val += mask_coeff[j] * proto[j * proto_row + py * mask_w + px];
      }
      row_ptr[x] = sigmoid(val);
    }
  }

  cv::Mat mask_resized;
  cv::resize(roi_mask, mask_resized,
             cv::Size(rect_in_img.width, rect_in_img.height), 0, 0,
             cv::INTER_LINEAR);
  out_mask = (mask_resized > 0.5f);
}

// ---------------------------------------------------------------------------
// infer: 完整的推理流程
// ---------------------------------------------------------------------------
bool YoloInference::infer(const cv::Mat &img, std::vector<Object> &objects,
                          float conf_thresh, float iou_thresh, bool is_rgb) {
  try {
    if (!context_)
      return false;
    objects.clear();

    if (img.empty())
      throw std::invalid_argument("Input image is empty");

    // 1. GPU 预处理: 将原始 BGR 图像写入零拷贝缓冲区, GPU kernel 完成剩余操作
    const size_t img_bytes = (size_t)img.cols * img.rows * img.channels();

    // 动态扩展缓冲区 (首次或图像尺寸变大时)
    if (img_bytes > img_buf_size_) {
      if (mapped_img_src_)
        cudaFreeHost(mapped_img_src_);
      if (cudaHostAlloc((void **)&mapped_img_src_, img_bytes,
                        cudaHostAllocMapped) != cudaSuccess)
        return false;
      if (cudaHostGetDevicePointer(&dev_img_src_, mapped_img_src_, 0) !=
          cudaSuccess)
        return false;
      img_buf_size_ = img_bytes;
    }

    // 拷贝原始图像到零拷贝缓冲 (CPU memcpy, ~0.5ms for 640x480x3)
    // 确保连续内存
    if (img.isContinuous()) {
      std::memcpy(mapped_img_src_, img.data, img_bytes);
    } else {
      cv::Mat cont = img.clone();
      std::memcpy(mapped_img_src_, cont.data, img_bytes);
    }

    // GPU kernel: resize + BGR2RGB + normalize + HWC2CHW →
    // mapped_input_/dev_input_
    float scale;
    int pad_x, pad_y;
    cudaPreprocess((const uint8_t *)dev_img_src_, (float *)dev_input_, img.cols,
                   img.rows, kINPUT_W, kINPUT_H, scale, pad_x, pad_y, stream_,
                   is_rgb);

    // 2. TRT 推理 (enqueueV3 for TRT 10)
    if (!context_->enqueueV3(stream_)) {
      std::cerr << "[Error] enqueueV3 failed" << std::endl;
      return false;
    }

    // 4. 同步等待推理完成 — 零拷贝: 无需 cudaMemcpy D2H
    cudaStreamSynchronize(stream_);

    // 5. 解析 output0: shape [116, 8400], 行主序
    //    行 0~3   : cx, cy, w, h  (in 640x640 空间)
    //    行 4~83  : 80 类别得分
    //    行 84~115: 32 掩码系数
    const int orig_w = img.cols;
    const int orig_h = img.rows;
    const float *det = mapped_output0_;   // 零拷贝: 直接读 CPU 端地址
    const float *proto = mapped_output1_; // 零拷贝: 直接读 CPU 端地址

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    // --- 【性能优化核心】：预分配数组，用于连续内存访问 ---
    std::vector<float> max_scores(kANCHORS, 0.0f);
    std::vector<int> max_class_ids(kANCHORS, -1);

    // 1. 魔法就在这里：外层循环类别 (80次)，内层循环 Anchor (8400次)
    for (int c = 0; c < kNC; ++c) {
      // 指向第 c 个类别的行首
      const float *class_row = det + (4 + c) * kANCHORS;
      for (int i = 0; i < kANCHORS; ++i) {
        // 🚀 这里的 class_row[i] 是 100% 完美的顺序读取！CPU 会疯狂预取数据
        if (class_row[i] > max_scores[i]) {
          max_scores[i] = class_row[i];
          max_class_ids[i] = c;
        }
      }
    }

    // 2. 遍历找出的最高得分，只对大于阈值的 Anchor 提取边框和掩码系数
    for (int i = 0; i < kANCHORS; ++i) {
      if (max_scores[i] < conf_thresh)
        continue;

      // 💡 注意：执行到这里的 Anchor 通常只有几十个！
      // 所以即使下面的 cx/cy/w/h
      // 和掩码系数是跨步读取，由于次数极少，代价已经微乎其微。

      // 读取 cx, cy, w, h (640x640 坐标系)
      float cx = det[0 * kANCHORS + i];
      float cy = det[1 * kANCHORS + i];
      float bw = det[2 * kANCHORS + i];
      float bh = det[3 * kANCHORS + i];

      // 映射回原图坐标系
      float x1 = (cx - bw / 2.0f - pad_x) / scale;
      float y1 = (cy - bh / 2.0f - pad_y) / scale;
      float w = bw / scale;
      float h = bh / scale;

      // 裁剪到图像范围
      x1 = std::max(0.0f, x1);
      y1 = std::max(0.0f, y1);
      w = std::min(w, static_cast<float>(orig_w) - x1);
      h = std::min(h, static_cast<float>(orig_h) - y1);

      if (w <= 0 || h <= 0)
        continue;

      bboxes.push_back(cv::Rect(
          static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)),
          static_cast<int>(std::round(w)), static_cast<int>(std::round(h))));
      scores.push_back(max_scores[i]);
      class_ids.push_back(max_class_ids[i]);

      // 读取 32 个掩码系数
      std::vector<float> coeff(kNM);
      for (int m = 0; m < kNM; ++m) {
        coeff[m] = det[(4 + kNC + m) * kANCHORS + i];
      }
      mask_coeffs.push_back(std::move(coeff)); // 使用 std::move 减少拷贝开销
    }

    // 6. NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, scores, conf_thresh, iou_thresh, nms_indices);

    // 7. GPU 批量掩码解码
    const int N = static_cast<int>(nms_indices.size());
    if (N > 0) {
      // 动态扩展缓冲区
      if (N > mask_buf_capacity_) {
        if (mapped_mask_coeffs_)
          cudaFreeHost(mapped_mask_coeffs_);
        if (mapped_mask_out_)
          cudaFreeHost(mapped_mask_out_);
        const size_t coeff_sz = N * kNM * sizeof(float);
        const size_t mask_sz = N * kMASK_H * kMASK_W * sizeof(float);
        cudaHostAlloc((void **)&mapped_mask_coeffs_, coeff_sz,
                      cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dev_mask_coeffs_, mapped_mask_coeffs_, 0);
        cudaHostAlloc((void **)&mapped_mask_out_, mask_sz, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dev_mask_out_, mapped_mask_out_, 0);
        mask_buf_capacity_ = N;
      }

      // 收集 NMS 后的系数到零拷贝缓冲
      std::vector<int> valid_nms; // NMS indices that passed validation
      for (int idx : nms_indices) {
        if (idx < 0 || idx >= static_cast<int>(bboxes.size()))
          continue;
        if (bboxes[idx].width <= 0 || bboxes[idx].height <= 0)
          continue;
        valid_nms.push_back(idx);
      }
      const int Nv = static_cast<int>(valid_nms.size());

      for (int i = 0; i < Nv; ++i) {
        std::memcpy(mapped_mask_coeffs_ + i * kNM,
                    mask_coeffs[valid_nms[i]].data(), kNM * sizeof(float));
      }

      // GPU kernel: 批量 dot product + sigmoid
      cudaDecodeMasks(
          (const float *)dev_output1_, // proto [32, 160, 160] — 留在 GPU
          (const float *)dev_mask_coeffs_, (float *)dev_mask_out_, Nv, kMASK_W,
          kMASK_H, kNM, stream_);
      cudaStreamSynchronize(stream_);

      // CPU: 从全分辨率 160×160 裁剪 ROI 并 resize 到 bbox 大小
      const float inv4 = 0.25f;
      for (int i = 0; i < Nv; ++i) {
        const cv::Rect &r = bboxes[valid_nms[i]];
        Object obj;
        obj.rect = r;
        obj.label = class_ids[valid_nms[i]];
        obj.prob = scores[valid_nms[i]];

        // 从 mapped_mask_out_ 构建 160×160 cv::Mat
        cv::Mat full_mask(kMASK_H, kMASK_W, CV_32FC1,
                          mapped_mask_out_ + i * kMASK_H * kMASK_W);

        // 计算 ROI 在 proto 坐标系中的位置
        int rx0 = static_cast<int>((r.x * scale + pad_x) * inv4);
        int ry0 = static_cast<int>((r.y * scale + pad_y) * inv4);
        int rx1 = static_cast<int>(((r.x + r.width) * scale + pad_x) * inv4);
        int ry1 = static_cast<int>(((r.y + r.height) * scale + pad_y) * inv4);
        rx0 = std::max(0, std::min(rx0, kMASK_W - 1));
        ry0 = std::max(0, std::min(ry0, kMASK_H - 1));
        rx1 = std::max(1, std::min(rx1, kMASK_W));
        ry1 = std::max(1, std::min(ry1, kMASK_H));

        cv::Mat roi_mask = full_mask(cv::Rect(rx0, ry0, rx1 - rx0, ry1 - ry0));
        cv::Mat mask_resized;
        cv::resize(roi_mask, mask_resized, cv::Size(r.width, r.height), 0, 0,
                   cv::INTER_LINEAR);
        obj.mask = (mask_resized > 0.5f);
        objects.push_back(std::move(obj));
      }
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[Exception] YoloInference::infer: " << e.what() << std::endl;
    return false;
  }
}

} // namespace semantic_vslam
