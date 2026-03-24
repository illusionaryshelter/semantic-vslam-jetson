/**
 * cuda_colorspace.cu
 *
 * GPU 加速色彩空间转换 — 模块化设计
 *
 * 独显版: cudaMalloc (device) + cudaMemcpy H2D/D2H
 *
 * RGB8↔BGR8: 640x480 < 1ms
 */

#include "semantic_vslam/cuda_colorspace.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace semantic_vslam {
namespace cuda {

// ======================================================================
// Kernel: RGB8 ↔ BGR8 通道交换 (R,G,B → B,G,R)
// ======================================================================
__global__ void kernelSwapRB(const uint8_t *__restrict__ src,
                              uint8_t *__restrict__ dst,
                              int total_pixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;

  int offset = idx * 3;
  uint8_t r = src[offset + 0];
  uint8_t g = src[offset + 1];
  uint8_t b = src[offset + 2];
  dst[offset + 0] = b;   // R → B
  dst[offset + 1] = g;   // G → G
  dst[offset + 2] = r;   // B → R
}

// ======================================================================
// Kernel: YUYV (YUV422) → BGR8
// ======================================================================
__global__ void kernelYUYVtoBGR(const uint8_t *__restrict__ src,
                                 uint8_t *__restrict__ dst,
                                 int width, int height) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = (width * height) / 2;
  if (pair_idx >= total_pairs) return;

  int src_offset = pair_idx * 4;
  float y0 = static_cast<float>(src[src_offset + 0]);
  float u  = static_cast<float>(src[src_offset + 1]) - 128.0f;
  float y1 = static_cast<float>(src[src_offset + 2]);
  float v  = static_cast<float>(src[src_offset + 3]) - 128.0f;

  int dst_offset0 = pair_idx * 6;
  dst[dst_offset0 + 0] = static_cast<uint8_t>(fminf(fmaxf(y0 + 1.772f * u, 0.0f), 255.0f));
  dst[dst_offset0 + 1] = static_cast<uint8_t>(fminf(fmaxf(y0 - 0.344f * u - 0.714f * v, 0.0f), 255.0f));
  dst[dst_offset0 + 2] = static_cast<uint8_t>(fminf(fmaxf(y0 + 1.402f * v, 0.0f), 255.0f));
  dst[dst_offset0 + 3] = static_cast<uint8_t>(fminf(fmaxf(y1 + 1.772f * u, 0.0f), 255.0f));
  dst[dst_offset0 + 4] = static_cast<uint8_t>(fminf(fmaxf(y1 - 0.344f * u - 0.714f * v, 0.0f), 255.0f));
  dst[dst_offset0 + 5] = static_cast<uint8_t>(fminf(fmaxf(y1 + 1.402f * v, 0.0f), 255.0f));
}

// ======================================================================
// 独显: cudaMalloc (device) 缓冲区
// ======================================================================
static struct ColorspaceBuffers {
  uint8_t *d_src = nullptr;    // device input
  uint8_t *d_dst = nullptr;    // device output
  size_t   alloc_size = 0;

  void ensure(size_t needed) {
    if (alloc_size >= needed) return;
    release();
    cudaMalloc(&d_src, needed);
    cudaMalloc(&d_dst, needed);
    alloc_size = needed;
  }

  void release() {
    if (d_src) { cudaFree(d_src); d_src = nullptr; }
    if (d_dst) { cudaFree(d_dst); d_dst = nullptr; }
    alloc_size = 0;
  }

  ~ColorspaceBuffers() { release(); }
} g_bufs;

// ======================================================================
// Public API: gpuSwapRB
// ======================================================================
void gpuSwapRB(const cv::Mat &src, cv::Mat &dst, void *stream) {
  if (src.empty() || src.channels() != 3) return;

  int total_pixels = src.rows * src.cols;
  size_t byte_size = total_pixels * 3;

  g_bufs.ensure(byte_size);

  // H2D: 拷贝输入到 device
  cudaMemcpy(g_bufs.d_src, src.data, byte_size, cudaMemcpyHostToDevice);

  // 启动 kernel
  int threads = 256;
  int blocks = (total_pixels + threads - 1) / threads;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  kernelSwapRB<<<blocks, threads, 0, s>>>(g_bufs.d_src, g_bufs.d_dst, total_pixels);

  if (s) cudaStreamSynchronize(s);
  else   cudaDeviceSynchronize();

  // D2H: 拷贝输出回 host
  if (dst.empty() || dst.rows != src.rows || dst.cols != src.cols || dst.type() != src.type()) {
    dst.create(src.rows, src.cols, src.type());
  }
  cudaMemcpy(dst.data, g_bufs.d_dst, byte_size, cudaMemcpyDeviceToHost);
}

// ======================================================================
// Public API: gpuYUYVtoBGR
// ======================================================================
void gpuYUYVtoBGR(const uint8_t *src, cv::Mat &dst,
                   int width, int height, void *stream) {
  if (!src || width <= 0 || height <= 0) return;

  size_t yuyv_size = width * height * 2;
  size_t bgr_size  = width * height * 3;
  g_bufs.ensure(std::max(yuyv_size, bgr_size));

  // H2D
  cudaMemcpy(g_bufs.d_src, src, yuyv_size, cudaMemcpyHostToDevice);

  int total_pairs = (width * height) / 2;
  int threads = 256;
  int blocks = (total_pairs + threads - 1) / threads;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  kernelYUYVtoBGR<<<blocks, threads, 0, s>>>(g_bufs.d_src, g_bufs.d_dst, width, height);

  if (s) cudaStreamSynchronize(s);
  else   cudaDeviceSynchronize();

  // D2H
  dst.create(height, width, CV_8UC3);
  cudaMemcpy(dst.data, g_bufs.d_dst, bgr_size, cudaMemcpyDeviceToHost);
}

} // namespace cuda
} // namespace semantic_vslam
