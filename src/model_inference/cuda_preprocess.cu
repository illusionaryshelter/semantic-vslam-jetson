/*
 * cuda_preprocess.cu
 *
 * 融合 GPU 预处理 CUDA Kernel:
 *   1. Letterbox Resize (双线性插值 / 直接拷贝)
 *   2. BGR → RGB (通道交换, is_rgb=true 时跳过)
 *   3. uint8 → float32 + /255 归一化
 *   4. HWC → CHW 格式转换
 *
 * 优化 (640×480 → 640×640 特殊路径):
 *   scale=1.0 时 resized_w=640 resized_h=480, pad_y=80
 *   所有内容是 1:1 映射, 无需双线性插值
 *   使用 preprocess_kernel_identity 直接拷贝
 */

#include "semantic_vslam/cuda_preprocess.hpp"
#include <cstdio>

namespace semantic_vslam {

// ======================================================================
// 通用 kernel: 双线性插值 letterbox
// ======================================================================
__global__ void preprocess_kernel_bilinear(
    const uint8_t *__restrict__ src,
    float *__restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    int resized_w, int resized_h,
    int pad_x, int pad_y,
    float scale_x, float scale_y,
    bool swap_rb)  // true: BGR→RGB; false: 通道保持原序
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  float c0, c1, c2;

  int rx = x - pad_x;
  int ry = y - pad_y;

  if (rx >= 0 && rx < resized_w && ry >= 0 && ry < resized_h) {
    float src_xf = rx * scale_x;
    float src_yf = ry * scale_y;

    int x0 = (int)src_xf;
    int y0 = (int)src_yf;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = min(x0, src_w - 1);
    y0 = min(y0, src_h - 1);

    float dx = src_xf - x0;
    float dy = src_yf - y0;

    const uint8_t *p00 = src + (y0 * src_w + x0) * 3;
    const uint8_t *p01 = src + (y0 * src_w + x1) * 3;
    const uint8_t *p10 = src + (y1 * src_w + x0) * 3;
    const uint8_t *p11 = src + (y1 * src_w + x1) * 3;

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // ch0, ch1, ch2: 输入通道顺序 (BGR 或 RGB)
    c0 = (p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11) / 255.0f;
    c1 = (p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11) / 255.0f;
    c2 = (p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11) / 255.0f;
  } else {
    c0 = c1 = c2 = 114.0f / 255.0f;
  }

  // 输出 CHW: YOLO 需要 RGB
  const int area = dst_h * dst_w;
  if (swap_rb) {
    // BGR → RGB: c0=B, c1=G, c2=R → R, G, B
    dst[0 * area + y * dst_w + x] = c2;  // R
    dst[1 * area + y * dst_w + x] = c1;  // G
    dst[2 * area + y * dst_w + x] = c0;  // B
  } else {
    // 已经是 RGB: c0=R, c1=G, c2=B
    dst[0 * area + y * dst_w + x] = c0;
    dst[1 * area + y * dst_w + x] = c1;
    dst[2 * area + y * dst_w + x] = c2;
  }
}

// ======================================================================
// 快速路径 kernel: scale=1.0 时直接拷贝 (无双线性插值)
//
// 640×480 → 640×640 场景:
//   pad_x=0, pad_y=80, resized_w=640, resized_h=480
//   每个源像素直接映射到目标位置, 无需任何插值计算
//   省去: 4 次内存查找 + 4 次浮点乘法 + 3 次加法 / pixel
// ======================================================================
__global__ void preprocess_kernel_identity(
    const uint8_t *__restrict__ src,
    float *__restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    int pad_x, int pad_y,
    int resized_w, int resized_h,
    bool swap_rb)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  const int area = dst_h * dst_w;
  int rx = x - pad_x;
  int ry = y - pad_y;

  if (rx >= 0 && rx < resized_w && ry >= 0 && ry < resized_h) {
    // 直接 1:1 读取源像素 (无插值!)
    const uint8_t *p = src + (ry * src_w + rx) * 3;

    if (swap_rb) {
      dst[0 * area + y * dst_w + x] = p[2] / 255.0f;  // R
      dst[1 * area + y * dst_w + x] = p[1] / 255.0f;  // G
      dst[2 * area + y * dst_w + x] = p[0] / 255.0f;  // B
    } else {
      dst[0 * area + y * dst_w + x] = p[0] / 255.0f;
      dst[1 * area + y * dst_w + x] = p[1] / 255.0f;
      dst[2 * area + y * dst_w + x] = p[2] / 255.0f;
    }
  } else {
    // padding 区域
    dst[0 * area + y * dst_w + x] = 114.0f / 255.0f;
    dst[1 * area + y * dst_w + x] = 114.0f / 255.0f;
    dst[2 * area + y * dst_w + x] = 114.0f / 255.0f;
  }
}

// ======================================================================
// Host 接口
// ======================================================================
void cudaPreprocess(
    const uint8_t *d_src,
    float *d_dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float &out_scale, int &out_pad_x, int &out_pad_y,
    cudaStream_t stream,
    bool is_rgb)
{
  // 计算 letterbox 参数
  out_scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int resized_w = (int)roundf(src_w * out_scale);
  int resized_h = (int)roundf(src_h * out_scale);
  out_pad_x = (dst_w - resized_w) / 2;
  out_pad_y = (dst_h - resized_h) / 2;

  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y);

  bool swap_rb = !is_rgb;  // BGR 输入需要交换, RGB 输入不需要

  // 判断是否可用快速路径:
  //   scale ≈ 1.0 (宽度刚好匹配, 高度只加 padding)
  //   resized_w == src_w && resized_h == src_h
  if (resized_w == src_w && resized_h == src_h) {
    // 🚀 快速路径: 直接拷贝, 无双线性插值
    preprocess_kernel_identity<<<grid, block, 0, stream>>>(
        d_src, d_dst, src_w, src_h, dst_w, dst_h,
        out_pad_x, out_pad_y, resized_w, resized_h, swap_rb);
  } else {
    // 通用路径: 双线性插值 letterbox
    float scale_x = (float)src_w / resized_w;
    float scale_y = (float)src_h / resized_h;
    preprocess_kernel_bilinear<<<grid, block, 0, stream>>>(
        d_src, d_dst, src_w, src_h, dst_w, dst_h,
        resized_w, resized_h, out_pad_x, out_pad_y,
        scale_x, scale_y, swap_rb);
  }
}

// ======================================================================
// 批量掩码解码 kernel (不变)
// ======================================================================
__global__ void decode_masks_kernel(
    const float *__restrict__ proto,
    const float *__restrict__ coeffs,
    float *__restrict__ mask_out,
    int N, int mask_w, int mask_h, int nm)
{
  int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int obj_idx = blockIdx.y;

  const int total_pixels = mask_h * mask_w;
  if (pixel_idx >= total_pixels || obj_idx >= N) return;

  const float *coeff = coeffs + obj_idx * nm;
  float val = 0.0f;
  for (int j = 0; j < nm; ++j) {
    val += coeff[j] * proto[j * total_pixels + pixel_idx];
  }

  val = 1.0f / (1.0f + expf(-val));
  mask_out[obj_idx * total_pixels + pixel_idx] = val;
}

void cudaDecodeMasks(
    const float *d_proto,
    const float *d_coeffs,
    float *d_mask_out,
    int N, int mask_w, int mask_h, int nm,
    cudaStream_t stream)
{
  if (N <= 0) return;

  const int total_pixels = mask_h * mask_w;
  const int threads = 256;
  dim3 block(threads);
  dim3 grid((total_pixels + threads - 1) / threads, N);

  decode_masks_kernel<<<grid, block, 0, stream>>>(
      d_proto, d_coeffs, d_mask_out,
      N, mask_w, mask_h, nm);
}

} // namespace semantic_vslam
