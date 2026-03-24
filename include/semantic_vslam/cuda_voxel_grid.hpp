/**
 * cuda_voxel_grid.hpp
 *
 * GPU 加速的 VoxelGrid 下采样, 替换 pcl::VoxelGrid
 *
 * 架构: .cu (CUDA kernel, 无 PCL) + .cpp (PCL 封装)
 * 颜色策略: first-point (语义颜色不能平均)
 *
 * 内存策略 (独显 PCIe):
 *   - 全部 cudaMalloc (device memory)
 *   - FilterRaw 接收 host 指针, 内部做 H2D/D2H
 *   - 持久 GPU 池: 一次分配跨调用复用
 */

#pragma once

#include <cstdint>

namespace semantic_vslam {

// GPU 端点结构 (紧凑 16 字节, 无 Eigen/PCL 依赖)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

/**
 * CUDA 底层实现 (纯 C 接口, 在 .cu 中定义)
 *
 * 独显模式: 接收 host 指针, 内部做 cudaMemcpy H2D/D2H
 *
 * @param input       输入点数组 (host 指针)
 * @param num_points  输入点数
 * @param h_output    输出缓冲区 (host 指针, 由内部 D2H 填充)
 * @param max_output  输出缓冲区最大容量
 * @param voxel_size  体素尺寸 (m)
 * @return 输出点数
 */
int cudaVoxelGridFilterRaw(
    const VoxelPoint* input, int num_points,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float min_x = 0, float max_x = 0,
    float min_y = 0, float max_y = 0,
    float min_z = 0, float max_z = 0);

}  // namespace semantic_vslam

// ---- PCL 高层封装 (仅在 C++ 编译器可见, NVCC 不编译) ----
#ifndef __CUDACC__
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

/**
 * CUDA VoxelGrid 下采样 (PCL 接口, 单帧)
 */
void cudaVoxelGridFilter(
    const pcl::PointCloud<pcl::PointXYZRGB>& input,
    pcl::PointCloud<pcl::PointXYZRGB>& output,
    float voxel_size);

/**
 * 增量式 CUDA VoxelGrid (全局地图)
 *
 * 维护持久化全局点云地图, 增量融合新帧:
 *   global_map (50K) + new_frame (5K) → cudaVoxelGrid → 50K
 */
class CudaIncrementalVoxelGrid {
public:
  explicit CudaIncrementalVoxelGrid(float voxel_size);

  /// 将新帧融合到全局地图
  void addCloud(const pcl::PointCloud<pcl::PointXYZRGB>& new_cloud);

  /// 获取当前全局地图 (只读引用)
  const pcl::PointCloud<pcl::PointXYZRGB>& getMap() const { return global_map_; }

  /// 获取当前地图点数
  size_t size() const { return global_map_.size(); }

  /// 清空全局地图
  void clear();

private:
  float voxel_size_;
  pcl::PointCloud<pcl::PointXYZRGB> global_map_;
};

}  // namespace semantic_vslam
#endif
