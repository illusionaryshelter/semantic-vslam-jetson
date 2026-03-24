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
  // 独显: 分别释放 host (free) 和 device (cudaFree)
  if (h_input_) free(h_input_);
  if (d_input_) cudaFree(d_input_);
  if (h_output0_) free(h_output0_);
  if (d_output0_) cudaFree(d_output0_);
  if (h_output1_) free(h_output1_);
  if (d_output1_) cudaFree(d_output1_);
  if (h_img_src_) free(h_img_src_);
  if (d_img_src_) cudaFree(d_img_src_);
  if (h_mask_coeffs_) free(h_mask_coeffs_);
  if (d_mask_coeffs_) cudaFree(d_mask_coeffs_);
  if (h_mask_out_) free(h_mask_out_);
  if (d_mask_out_) cudaFree(d_mask_out_);
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

    // ---- 独显: malloc (host) + cudaMalloc (device) ----
    h_input_ = (float *)malloc(in_sz * sizeof(float));
    h_output0_ = (float *)malloc(out0_sz * sizeof(float));
    h_output1_ = (float *)malloc(out1_sz * sizeof(float));
    if (!h_input_ || !h_output0_ || !h_output1_) return false;

    if (cudaMalloc(&d_input_, in_sz * sizeof(float)) != cudaSuccess)
      return false;
    if (cudaMalloc(&d_output0_, out0_sz * sizeof(float)) != cudaSuccess)
      return false;
    if (cudaMalloc(&d_output1_, out1_sz * sizeof(float)) != cudaSuccess)
      return false;

    if (cudaStreamCreate(&stream_) != cudaSuccess)
      return false;

    // 设置张量地址 (TRT 10 API) — 使用 device 指针
    context_->setTensorAddress("images", d_input_);
    context_->setTensorAddress("output0", d_output0_);
    context_->setTensorAddress("output1", d_output1_);

    // ---- GPU 预处理: 预分配默认图像缓冲区 (640×480×3) ----
    const size_t default_img_sz = 640 * 480 * 3;
    h_img_src_ = (uint8_t *)malloc(default_img_sz);
    if (!h_img_src_) return false;
    if (cudaMalloc(&d_img_src_, default_img_sz) != cudaSuccess)
      return false;
    img_buf_size_ = default_img_sz;

    // ---- GPU 掩码解码: 预分配缓冲区 (32 个目标) ----
    const int default_mask_cap = 32;
    const size_t coeff_sz = default_mask_cap * kNM * sizeof(float);
    const size_t mask_sz = default_mask_cap * kMASK_H * kMASK_W * sizeof(float);
    h_mask_coeffs_ = (float *)malloc(coeff_sz);
    h_mask_out_ = (float *)malloc(mask_sz);
    if (!h_mask_coeffs_ || !h_mask_out_) return false;
    if (cudaMalloc(&d_mask_coeffs_, coeff_sz) != cudaSuccess)
      return false;
    if (cudaMalloc(&d_mask_out_, mask_sz) != cudaSuccess)
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

    // 1. GPU 预处理: 拷贝原始图像到 device, GPU kernel 完成剩余操作
    const size_t img_bytes = (size_t)img.cols * img.rows * img.channels();

    // 动态扩展缓冲区
    if (img_bytes > img_buf_size_) {
      if (h_img_src_) free(h_img_src_);
      if (d_img_src_) cudaFree(d_img_src_);
      h_img_src_ = (uint8_t *)malloc(img_bytes);
      if (!h_img_src_) return false;
      if (cudaMalloc(&d_img_src_, img_bytes) != cudaSuccess)
        return false;
      img_buf_size_ = img_bytes;
    }

    // 拷贝原始图像到 host 缓冲
    if (img.isContinuous()) {
      std::memcpy(h_img_src_, img.data, img_bytes);
    } else {
      cv::Mat cont = img.clone();
      std::memcpy(h_img_src_, cont.data, img_bytes);
    }

    // H2D: 拷贝图像到 device
    cudaMemcpyAsync(d_img_src_, h_img_src_, img_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // GPU kernel: resize + BGR2RGB + normalize + HWC2CHW → d_input_
    float scale;
    int pad_x, pad_y;
    cudaPreprocess((const uint8_t *)d_img_src_, (float *)d_input_, img.cols,
                   img.rows, kINPUT_W, kINPUT_H, scale, pad_x, pad_y, stream_,
                   is_rgb);

    // 2. TRT 推理 (enqueueV3 for TRT 10)
    if (!context_->enqueueV3(stream_)) {
      std::cerr << "[Error] enqueueV3 failed" << std::endl;
      return false;
    }

    // 同步 + D2H: 拷贝 TRT 输出回 host
    cudaStreamSynchronize(stream_);

    const size_t out0_sz = 1 * kROWS * kANCHORS;
    const size_t out1_sz = 1 * kNM * kMASK_H * kMASK_W;
    cudaMemcpy(h_output0_, d_output0_, out0_sz * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output1_, d_output1_, out1_sz * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 5. 解析 output0: shape [116, 8400], 行主序
    //    行 0~3   : cx, cy, w, h  (in 640x640 空间)
    //    行 4~83  : 80 类别得分
    //    行 84~115: 32 掩码系数
    const int orig_w = img.cols;
    const int orig_h = img.rows;
    const float *det = h_output0_;    // host: D2H 后直接读
    const float *proto = h_output1_;  // host: D2H 后直接读

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
        if (h_mask_coeffs_) free(h_mask_coeffs_);
        if (d_mask_coeffs_) cudaFree(d_mask_coeffs_);
        if (h_mask_out_) free(h_mask_out_);
        if (d_mask_out_) cudaFree(d_mask_out_);
        const size_t coeff_sz = N * kNM * sizeof(float);
        const size_t mask_sz = N * kMASK_H * kMASK_W * sizeof(float);
        h_mask_coeffs_ = (float *)malloc(coeff_sz);
        cudaMalloc(&d_mask_coeffs_, coeff_sz);
        h_mask_out_ = (float *)malloc(mask_sz);
        cudaMalloc(&d_mask_out_, mask_sz);
        mask_buf_capacity_ = N;
      }

      // 收集 NMS 后的系数到 host 缓冲
      std::vector<int> valid_nms;
      for (int idx : nms_indices) {
        if (idx < 0 || idx >= static_cast<int>(bboxes.size()))
          continue;
        if (bboxes[idx].width <= 0 || bboxes[idx].height <= 0)
          continue;
        valid_nms.push_back(idx);
      }
      const int Nv = static_cast<int>(valid_nms.size());

      for (int i = 0; i < Nv; ++i) {
        std::memcpy(h_mask_coeffs_ + i * kNM,
                    mask_coeffs[valid_nms[i]].data(), kNM * sizeof(float));
      }

      // H2D: 拷贝系数到 device
      cudaMemcpy(d_mask_coeffs_, h_mask_coeffs_, Nv * kNM * sizeof(float),
                 cudaMemcpyHostToDevice);

      // GPU kernel: 批量 dot product + sigmoid
      cudaDecodeMasks(
          (const float *)d_output1_, // proto [32, 160, 160] — 在 device
          (const float *)d_mask_coeffs_, (float *)d_mask_out_, Nv, kMASK_W,
          kMASK_H, kNM, stream_);
      cudaStreamSynchronize(stream_);

      // D2H: 拷贝掩码输出回 host
      cudaMemcpy(h_mask_out_, d_mask_out_,
                 Nv * kMASK_H * kMASK_W * sizeof(float),
                 cudaMemcpyDeviceToHost);

      // CPU: 从全分辨率 160×160 裁剪 ROI 并 resize 到 bbox 大小
      const float inv4 = 0.25f;
      for (int i = 0; i < Nv; ++i) {
        const cv::Rect &r = bboxes[valid_nms[i]];
        Object obj;
        obj.rect = r;
        obj.label = class_ids[valid_nms[i]];
        obj.prob = scores[valid_nms[i]];

        // 从 h_mask_out_ 构建 160×160 cv::Mat
        cv::Mat full_mask(kMASK_H, kMASK_W, CV_32FC1,
                          h_mask_out_ + i * kMASK_H * kMASK_W);

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
