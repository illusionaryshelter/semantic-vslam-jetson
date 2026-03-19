#pragma once

/*
 * cuda_preprocess.hpp
 *
 * GPU 预处理接口: letterbox resize + BGR→RGB + uint8→float32/255 + HWC→CHW
 * 单个 CUDA kernel 调用替代多次 OpenCV CPU 操作
 *
 * 优化:
 *   - is_rgb=true 时跳过 BGR→RGB 通道交换 (避免双重交换)
 *   - scale=1.0 时使用直接拷贝路径 (跳过双线性插值)
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace semantic_vslam {

/// GPU 融合预处理: 将原始图像转换为 TRT 输入格式
/// @param d_src      GPU 端源图像指针 (uint8 HWC, 连续内存)
/// @param d_dst      GPU 端 TRT 输入指针 (float32 CHW, [3, dst_h, dst_w])
/// @param src_w      源图像宽度
/// @param src_h      源图像高度
/// @param dst_w      目标宽度 (640)
/// @param dst_h      目标高度 (640)
/// @param out_scale  输出: letterbox 缩放比例
/// @param out_pad_x  输出: letterbox x 偏移
/// @param out_pad_y  输出: letterbox y 偏移
/// @param stream     CUDA stream
/// @param is_rgb     true = 输入已是 RGB (跳过通道交换); false = 输入是 BGR
void cudaPreprocess(
    const uint8_t *d_src,
    float *d_dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float &out_scale, int &out_pad_x, int &out_pad_y,
    cudaStream_t stream,
    bool is_rgb = false);

/// GPU 批量掩码解码
void cudaDecodeMasks(
    const float *d_proto,
    const float *d_coeffs,
    float *d_mask_out,
    int N, int mask_w, int mask_h, int nm,
    cudaStream_t stream);

} // namespace semantic_vslam
