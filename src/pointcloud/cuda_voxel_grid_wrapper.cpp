/**
 * cuda_voxel_grid_wrapper.cpp
 *
 * PCL ↔ VoxelPoint 转换封装
 * 此文件由 g++ 编译 (可用 PCL/Eigen), 调用 .cu 中的 CUDA 接口
 *
 * Jetson UMA 零拷贝:
 *   h_input / h_output 使用 cudaMallocManaged 分配,
 *   CPU 写入 → GPU 直接读取, 无 H2D/D2H 拷贝
 *   持久缓冲区: 仅在容量不足时重新分配
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

// 持久 managed 内存缓冲区 (进程生命周期)
static VoxelPoint* g_managed_input  = nullptr;
static VoxelPoint* g_managed_output = nullptr;
static int g_managed_capacity = 0;

// 确保 managed 缓冲区容量
static void ensureManagedBuffers(int N) {
  if (N <= g_managed_capacity) return;

  int new_cap = std::max(N, static_cast<int>(g_managed_capacity * 1.5));
  if (g_managed_input)  cudaVoxelGridFreeManaged(g_managed_input);
  if (g_managed_output) cudaVoxelGridFreeManaged(g_managed_output);

  g_managed_input  = cudaVoxelGridAllocManaged(new_cap);
  g_managed_output = cudaVoxelGridAllocManaged(new_cap);
  g_managed_capacity = new_cap;
}

void cudaVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZRGB> &input,
                         pcl::PointCloud<pcl::PointXYZRGB> &output,
                         float voxel_size) {
  const int N = static_cast<int>(input.size());
  if (N == 0)
    return;
  if (voxel_size <= 0.0f) {
    output = input;
    return;
  }

  // 确保 managed 缓冲区足够
  ensureManagedBuffers(N);

  // ---- PCL → VoxelPoint (直接写入 managed memory, GPU 可零拷贝读取) ----
  for (int i = 0; i < N; ++i) {
    const auto &pt = input.points[i];
    g_managed_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }

  // ---- 调用 CUDA 底层 (managed 指针, Jetson UMA 无 H2D/D2H) ----
  int num_out = cudaVoxelGridFilterRaw(
      g_managed_input, N, g_managed_output, N, voxel_size);

  // ---- VoxelPoint → PCL (直接从 managed memory 读, 无 D2H) ----
  output.resize(num_out);
  output.width = num_out;
  output.height = 1;
  output.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto &op = output.points[i];
    op.x = g_managed_output[i].x;
    op.y = g_managed_output[i].y;
    op.z = g_managed_output[i].z;
    op.r = g_managed_output[i].r;
    op.g = g_managed_output[i].g;
    op.b = g_managed_output[i].b;
  }
}

} // namespace semantic_vslam
