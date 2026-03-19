#pragma once
/**
 * cuda_colorspace.hpp
 *
 * 模块化 CUDA 色彩空间转换工具
 *
 * 功能:
 *   - RGB8 ↔ BGR8 通道交换 (Jetson zero-copy)
 *   - YUYV → BGR8 转换 (如需要)
 *
 * 设计:
 *   解耦模块，可被任何节点调用。
 *   在 Jetson 上利用 zero-copy 内存避免 CPU-GPU 拷贝。
 */

#include <opencv2/core.hpp>

namespace semantic_vslam {
namespace cuda {

/**
 * GPU 加速的 RGB8 ↔ BGR8 通道交换
 *
 * @param src     输入图像 (3 通道, continuous)
 * @param dst     输出图像 (预分配, 与 src 同尺寸同类型)
 * @param stream  CUDA stream (0 = 默认 stream)
 *
 * 如果 dst 为空会自动分配; 如果 src == dst 支持原地操作。
 * 在 Jetson Orin Nano 上 640x480 耗时 <1ms。
 */
void gpuSwapRB(const cv::Mat &src, cv::Mat &dst, void *stream = nullptr);

/**
 * GPU 加速的 YUYV (YUV422) → BGR8 转换
 *
 * @param src     输入 YUYV 图像 (width x height x 2 bytes)
 * @param dst     输出 BGR8 图像
 * @param width   图像宽度
 * @param height  图像高度
 * @param stream  CUDA stream
 */
void gpuYUYVtoBGR(const uint8_t *src, cv::Mat &dst,
                   int width, int height, void *stream = nullptr);

} // namespace cuda
} // namespace semantic_vslam
