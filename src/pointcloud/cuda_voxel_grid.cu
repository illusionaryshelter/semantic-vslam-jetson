/**
 * cuda_voxel_grid.cu
 *
 * GPU 加速 VoxelGrid 下采样 (底层 CUDA 实现)
 *
 * 算法: calcVoxelIndex → thrust::sort_by_key → markBoundary → inclusive_scan → gatherFirstPoint
 * 颜色策略: first-point (语义颜色不可平均)
 *
 * 空间哈希: uint64_t bit-packing (21位/轴, 无碰撞, ±20km @ 0.02m)
 *
 * 内存策略 (独显 PCIe):
 *   1. GPUPool: cudaMalloc (纯 device memory, GPU L2 cached)
 *   2. I/O: cudaMalloc (device) + cudaMemcpy H2D/D2H (显式拷贝)
 *   3. 持久分配: 一次分配跨调用复用, 1.5x 增长
 *
 * 注意: 此文件仅包含 CUDA Runtime / Thrust, 不包含 PCL / Eigen
 *       PCL 封装在 cuda_voxel_grid_wrapper.cpp 中
 */

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>

namespace semantic_vslam {

struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

// ---- 空间哈希常量 ----
static constexpr uint64_t VOXEL_BITS = 21;
static constexpr uint64_t VOXEL_OFFSET = (1ULL << (VOXEL_BITS - 1));

// ============================================================================
// 持久 GPU 内存池: 全部 cudaMalloc (纯 device memory)
//
// 包含中间缓冲区 + I/O 缓冲区, 一次分配跨调用复用
// ============================================================================
struct GPUPool {
  // 中间缓冲区 (仅 GPU 读写)
  uint64_t*   voxel_keys = nullptr;
  uint32_t*   point_idx  = nullptr;
  uint32_t*   boundary   = nullptr;
  uint32_t*   prefix     = nullptr;

  // I/O 缓冲区 (device memory, 需要显式 H2D/D2H)
  VoxelPoint* d_input    = nullptr;
  VoxelPoint* d_output   = nullptr;

  int capacity = 0;

  void ensure(int N) {
    if (N <= capacity) return;
    int new_cap = std::max(N, static_cast<int>(capacity * 1.5));
    free_all();

    cudaMalloc(&voxel_keys, new_cap * sizeof(uint64_t));
    cudaMalloc(&point_idx,  new_cap * sizeof(uint32_t));
    cudaMalloc(&boundary,   new_cap * sizeof(uint32_t));
    cudaMalloc(&prefix,     new_cap * sizeof(uint32_t));
    cudaMalloc(&d_input,    new_cap * sizeof(VoxelPoint));
    cudaMalloc(&d_output,   new_cap * sizeof(VoxelPoint));
    capacity = new_cap;
  }

  void free_all() {
    if (voxel_keys) { cudaFree(voxel_keys); voxel_keys = nullptr; }
    if (point_idx)  { cudaFree(point_idx);  point_idx  = nullptr; }
    if (boundary)   { cudaFree(boundary);   boundary   = nullptr; }
    if (prefix)     { cudaFree(prefix);     prefix     = nullptr; }
    if (d_input)    { cudaFree(d_input);    d_input    = nullptr; }
    if (d_output)   { cudaFree(d_output);   d_output   = nullptr; }
    capacity = 0;
  }

  ~GPUPool() { free_all(); }
};

static GPUPool g_pool;

// ---- Kernel 1: 计算每个点的 voxel key (uint64_t bit-packing) ----
__global__ void calcVoxelIndexKernel(
    const VoxelPoint* __restrict__ points,
    int num_points,
    float inv_voxel,
    uint64_t* __restrict__ voxel_keys,
    uint32_t* __restrict__ point_indices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  const VoxelPoint& p = points[idx];

  int64_t vx = static_cast<int64_t>(floorf(p.x * inv_voxel));
  int64_t vy = static_cast<int64_t>(floorf(p.y * inv_voxel));
  int64_t vz = static_cast<int64_t>(floorf(p.z * inv_voxel));

  uint64_t ux = static_cast<uint64_t>(vx + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uy = static_cast<uint64_t>(vy + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uz = static_cast<uint64_t>(vz + static_cast<int64_t>(VOXEL_OFFSET));

  voxel_keys[idx] = (ux << (VOXEL_BITS * 2)) | (uy << VOXEL_BITS) | uz;
  point_indices[idx] = static_cast<uint32_t>(idx);
}

// ---- Kernel 2: 标记 voxel 边界 ----
__global__ void markBoundaryKernel(
    const uint64_t* __restrict__ sorted_voxel_keys,
    int num_points,
    uint32_t* __restrict__ boundary)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  if (idx == 0) {
    boundary[idx] = 1;
  } else {
    boundary[idx] = (sorted_voxel_keys[idx] != sorted_voxel_keys[idx - 1]) ? 1 : 0;
  }
}

// ---- Kernel 3: 提取每个 voxel 的第一个点 ----
__global__ void gatherFirstPointKernel(
    const VoxelPoint* __restrict__ points,
    const uint32_t* __restrict__ sorted_point_indices,
    const uint32_t* __restrict__ boundary,
    const uint32_t* __restrict__ prefix_sum,
    int num_points,
    VoxelPoint* __restrict__ output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  if (boundary[idx] == 1) {
    uint32_t out_idx = prefix_sum[idx] - 1;
    uint32_t pt_idx = sorted_point_indices[idx];
    output[out_idx] = points[pt_idx];
  }
}

// ============================================================================
// 底层接口: 接收 host 指针, 内部做 H2D/D2H
//
// 独显: CPU 和 GPU 有各自的物理内存 (通过 PCIe 连接)
//   1. cudaMemcpy H2D: host input → device input
//   2. GPU 处理 (全在 device memory 上)
//   3. cudaMemcpy D2H: device output → host output
// ============================================================================
int cudaVoxelGridFilterRaw(
    const VoxelPoint* input, int N,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float /*min_x*/, float /*max_x*/,
    float /*min_y*/, float /*max_y*/,
    float /*min_z*/, float /*max_z*/)
{
  if (N <= 0) return 0;

  const float inv_voxel = 1.0f / voxel_size;

  g_pool.ensure(N);

  // ---- H2D: 拷贝输入到 device ----
  cudaMemcpy(g_pool.d_input, input, N * sizeof(VoxelPoint),
             cudaMemcpyHostToDevice);

  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  // ---- 计算 voxel key ----
  calcVoxelIndexKernel<<<GRID, BLOCK>>>(
      g_pool.d_input, N, inv_voxel,
      g_pool.voxel_keys, g_pool.point_idx);
  cudaDeviceSynchronize();

  // ---- Thrust sort (纯 device memory) ----
  thrust::device_ptr<uint64_t> keys_begin(g_pool.voxel_keys);
  thrust::device_ptr<uint32_t> vals_begin(g_pool.point_idx);
  thrust::sort_by_key(keys_begin, keys_begin + N, vals_begin);

  // ---- 标记边界 + inclusive scan ----
  markBoundaryKernel<<<GRID, BLOCK>>>(
      g_pool.voxel_keys, N, g_pool.boundary);
  cudaDeviceSynchronize();

  thrust::device_ptr<uint32_t> bnd_begin(g_pool.boundary);
  thrust::device_ptr<uint32_t> pfx_begin(g_pool.prefix);
  thrust::inclusive_scan(bnd_begin, bnd_begin + N, pfx_begin);

  // ---- 读取 num_unique: D2H 4 字节 ----
  uint32_t num_unique = 0;
  cudaMemcpy(&num_unique, g_pool.prefix + N - 1,
             sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (num_unique == 0 || static_cast<int>(num_unique) > max_output) {
    return 0;
  }

  // ---- 提取每个 voxel 第一个点 (写到 device output) ----
  gatherFirstPointKernel<<<GRID, BLOCK>>>(
      g_pool.d_input, g_pool.point_idx, g_pool.boundary, g_pool.prefix,
      N, g_pool.d_output);
  cudaDeviceSynchronize();

  // ---- D2H: 拷贝输出回 host ----
  cudaMemcpy(h_output, g_pool.d_output, num_unique * sizeof(VoxelPoint),
             cudaMemcpyDeviceToHost);

  return static_cast<int>(num_unique);
}

}  // namespace semantic_vslam
