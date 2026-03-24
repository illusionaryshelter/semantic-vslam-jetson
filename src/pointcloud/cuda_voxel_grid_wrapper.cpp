/**
 * cuda_voxel_grid_wrapper.cpp
 *
 * PCL ↔ VoxelPoint 转换封装
 * 此文件由 g++ 编译 (可用 PCL/Eigen), 调用 .cu 中的 CUDA 接口
 *
 * 独显模式: 使用 std::vector 作为 host 缓冲区
 *   FilterRaw 内部做 cudaMemcpy H2D/D2H
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

// 持久 host 缓冲区 (避免每帧 alloc/free)
static std::vector<VoxelPoint> g_h_input;
static std::vector<VoxelPoint> g_h_output;

void cudaVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZRGB> &input,
                         pcl::PointCloud<pcl::PointXYZRGB> &output,
                         float voxel_size) {
  const int N = static_cast<int>(input.size());
  if (N == 0) return;
  if (voxel_size <= 0.0f) {
    output = input;
    return;
  }

  g_h_input.resize(N);
  g_h_output.resize(N);

  // ---- PCL → VoxelPoint ----
  for (int i = 0; i < N; ++i) {
    const auto &pt = input.points[i];
    g_h_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }

  // ---- CUDA (FilterRaw 内部做 H2D/D2H) ----
  int num_out = cudaVoxelGridFilterRaw(
      g_h_input.data(), N, g_h_output.data(), N, voxel_size);

  // ---- VoxelPoint → PCL ----
  output.resize(num_out);
  output.width = num_out;
  output.height = 1;
  output.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto &op = output.points[i];
    op.x = g_h_output[i].x;
    op.y = g_h_output[i].y;
    op.z = g_h_output[i].z;
    op.r = g_h_output[i].r;
    op.g = g_h_output[i].g;
    op.b = g_h_output[i].b;
  }
}

// ============================================================================
// CudaIncrementalVoxelGrid 实现
// ============================================================================

CudaIncrementalVoxelGrid::CudaIncrementalVoxelGrid(float voxel_size)
    : voxel_size_(voxel_size) {}

void CudaIncrementalVoxelGrid::addCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>& new_cloud) {
  if (new_cloud.empty()) return;

  const int M = static_cast<int>(global_map_.size());
  const int N = static_cast<int>(new_cloud.size());
  const int total = M + N;

  g_h_input.resize(total);
  g_h_output.resize(total);

  // ---- 拼接: 先放现有地图 (sort 稳定性下现有点优先) ----
  for (int i = 0; i < M; ++i) {
    const auto& pt = global_map_.points[i];
    g_h_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }
  for (int i = 0; i < N; ++i) {
    const auto& pt = new_cloud.points[i];
    g_h_input[M + i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }

  // ---- CUDA VoxelGrid (FilterRaw 内部做 H2D/D2H) ----
  int num_out = cudaVoxelGridFilterRaw(
      g_h_input.data(), total, g_h_output.data(), total, voxel_size_);

  // ---- 更新全局地图 ----
  global_map_.resize(num_out);
  global_map_.width = num_out;
  global_map_.height = 1;
  global_map_.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto& op = global_map_.points[i];
    op.x = g_h_output[i].x;
    op.y = g_h_output[i].y;
    op.z = g_h_output[i].z;
    op.r = g_h_output[i].r;
    op.g = g_h_output[i].g;
    op.b = g_h_output[i].b;
  }
}

void CudaIncrementalVoxelGrid::clear() {
  global_map_.clear();
}

} // namespace semantic_vslam
