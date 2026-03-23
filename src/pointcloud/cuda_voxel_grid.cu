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
 * Jetson 优化:
 *   1. cudaMallocManaged (UMA 零拷贝: CPU/GPU 共享物理内存, 无 H2D/D2H)
 *   2. 持久显存池 (GPUPool: 一次分配, 跨调用复用, 避免反复 cudaMalloc/Free)
 *   3. thrust::device_ptr 包装 managed 指针 (兼容 Thrust 算法)
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

// 与 header 中的 VoxelPoint 完全一致 (定义在 namespace 内)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

// ---- 空间哈希常量 ----
static constexpr uint64_t VOXEL_BITS = 21;
static constexpr uint64_t VOXEL_OFFSET = (1ULL << (VOXEL_BITS - 1));

// ============================================================================
// 持久显存池: 一次分配, 跨调用复用
//
// 避免每帧 5 次 cudaMalloc + 5 次 cudaFree (~2-5ms 系统调用开销)
// 使用 cudaMallocManaged: Jetson UMA 上 CPU/GPU 共享物理内存, 真正零拷贝
// ============================================================================
struct GPUPool {
  uint64_t*   voxel_keys = nullptr;  // Voxel hash keys (排序用)
  uint32_t*   point_idx  = nullptr;  // 原始点索引 (随 key 一起排序)
  uint32_t*   boundary   = nullptr;  // Voxel 边界标记
  uint32_t*   prefix     = nullptr;  // Inclusive scan 前缀和
  VoxelPoint* output     = nullptr;  // 输出缓冲区
  int capacity = 0;                  // 当前分配的最大点数

  // 确保容量足够, 仅在 N > capacity 时重新分配
  void ensure(int N) {
    if (N <= capacity) return;

    // 增长策略: 至少 1.5x, 避免频繁 realloc
    int new_cap = std::max(N, static_cast<int>(capacity * 1.5));
    free_all();

    cudaMallocManaged(&voxel_keys, new_cap * sizeof(uint64_t));
    cudaMallocManaged(&point_idx,  new_cap * sizeof(uint32_t));
    cudaMallocManaged(&boundary,   new_cap * sizeof(uint32_t));
    cudaMallocManaged(&prefix,     new_cap * sizeof(uint32_t));
    cudaMallocManaged(&output,     new_cap * sizeof(VoxelPoint));
    capacity = new_cap;
  }

  void free_all() {
    if (voxel_keys) { cudaFree(voxel_keys); voxel_keys = nullptr; }
    if (point_idx)  { cudaFree(point_idx);  point_idx  = nullptr; }
    if (boundary)   { cudaFree(boundary);   boundary   = nullptr; }
    if (prefix)     { cudaFree(prefix);     prefix     = nullptr; }
    if (output)     { cudaFree(output);     output     = nullptr; }
    capacity = 0;
  }

  ~GPUPool() { free_all(); }
};

// 全局持久池 (进程生命周期, 只分配一次)
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
// 底层接口: 使用 managed 内存指针 (Jetson UMA 零拷贝)
//
// 调用者通过 cudaVoxelGridGetManagedBuffers() 获取 managed 内存,
// 直接写入数据, 无 H2D 拷贝. 输出同理, 无 D2H 拷贝.
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

  // 确保持久池容量足够
  g_pool.ensure(N);

  // ---- 计算 voxel key ----
  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  calcVoxelIndexKernel<<<GRID, BLOCK>>>(
      input, N, inv_voxel,
      g_pool.voxel_keys, g_pool.point_idx);
  cudaDeviceSynchronize();

  // ---- Thrust sort_by_key (通过 device_ptr 包装 managed 指针) ----
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

  // 同步后读取 managed memory 中的 num_unique (Jetson UMA 直接读)
  cudaDeviceSynchronize();
  uint32_t num_unique = g_pool.prefix[N - 1];

  if (num_unique == 0 || static_cast<int>(num_unique) > max_output) {
    return 0;
  }

  // ---- 提取每个 voxel 第一个点 ----
  gatherFirstPointKernel<<<GRID, BLOCK>>>(
      input, g_pool.point_idx, g_pool.boundary, g_pool.prefix,
      N, g_pool.output);
  cudaDeviceSynchronize();

  // ---- 输出: managed memory, Jetson UMA 无需 D2H 拷贝 ----
  // 如果 h_output 是 managed memory (由 wrapper 通过 getManagedBuffer 分配),
  // 则直接 memcpy 在 UMA 上等同于内存内移动, 无 PCIe 传输
  // 如果是普通 host memory (单元测试等), memcpy 正常工作
  memcpy(h_output, g_pool.output, num_unique * sizeof(VoxelPoint));

  return static_cast<int>(num_unique);
}

// ============================================================================
// Managed 内存分配/释放接口 (供 wrapper 使用)
//
// Jetson UMA: cudaMallocManaged 分配的内存 CPU/GPU 共享物理页面
// 调用者可直接用 CPU 写入 VoxelPoint 数据, GPU 内核直接读取, 零拷贝
// ============================================================================
VoxelPoint* cudaVoxelGridAllocManaged(int max_points) {
  VoxelPoint* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, max_points * sizeof(VoxelPoint));
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
    return nullptr;
  }
  return ptr;
}

void cudaVoxelGridFreeManaged(VoxelPoint* ptr) {
  if (ptr) cudaFree(ptr);
}

}  // namespace semantic_vslam
