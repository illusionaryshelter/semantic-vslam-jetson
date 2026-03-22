/**
 * semantic_map_node.cpp
 *
 * 语义地图累积节点
 *
 * 输出:
 *   1. /semantic_vslam/semantic_map_cloud — 3D 语义着色点云地图
 *   2. /semantic_vslam/grid_map           — 2D 占据栅格地图 (从 3D 点云投影)
 */

#include "semantic_vslam/semantic_map_node.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace semantic_vslam {

SemanticMapNode::SemanticMapNode()
    : Node("semantic_map_node") {

  // ---- 参数 ----
  this->declare_parameter<std::string>("target_frame", "map");
  this->declare_parameter<double>("voxel_size", 0.02);
  this->declare_parameter<int>("max_clouds", 50);
  this->declare_parameter<int>("cloud_decimation", 2);
  this->declare_parameter<double>("publish_rate", 1.0);
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_min_height", 0.1);
  this->declare_parameter<double>("grid_max_height", 2.0);
  this->declare_parameter<bool>("enable_profiling", false);

  target_frame_ = this->get_parameter("target_frame").as_string();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
  max_clouds_ = this->get_parameter("max_clouds").as_int();
  cloud_decimation_ = this->get_parameter("cloud_decimation").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();
  grid_cell_size_ = this->get_parameter("grid_cell_size").as_double();
  grid_min_height_ = this->get_parameter("grid_min_height").as_double();
  grid_max_height_ = this->get_parameter("grid_max_height").as_double();
  enable_profiling_ = this->get_parameter("enable_profiling").as_bool();

  // ---- TF2 ----
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---- 订阅语义点云 ----
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_cloud", 5,
      std::bind(&SemanticMapNode::cloudCallback, this, std::placeholders::_1));

  // ---- 发布 3D 语义地图 + 2D 栅格地图 ----
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_map_cloud", 1);
  grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/semantic_vslam/grid_map", 1);

  // ---- 定时发布 ----
  auto period_ms = static_cast<int>(1000.0 / publish_rate);
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&SemanticMapNode::publishTimer, this));

  RCLCPP_INFO(this->get_logger(),
      "SemanticMapNode ready. voxel=%.3f, grid_cell=%.3f, max_clouds=%d",
      voxel_size_, grid_cell_size_, max_clouds_);
}

// ---------------------------------------------------------------------------
void SemanticMapNode::cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  auto t0 = enable_profiling_ ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  // 查找 TF: semantic_cloud frame → map
  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, msg->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(0.1));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "TF lookup failed: %s", ex.what());
    return;
  }

  Eigen::Isometry3d tf_eigen = tf2::transformToEigen(tf_stamped.transform);
  Eigen::Matrix4f tf_mat = tf_eigen.matrix().cast<float>();

  // 解析输入点云
  pcl::PointCloud<pcl::PointXYZRGB> input_cloud;
  pcl::fromROSMsg(*msg, input_cloud);

  // 抽稀 + 变换 + 过滤 NaN
  auto transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  const int dec = std::max(1, cloud_decimation_);
  const int total = static_cast<int>(input_cloud.size());
  transformed->reserve(total / dec + 1);

  for (int i = 0; i < total; i += dec) {
    const auto &pt = input_cloud.points[i];
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
      continue;
    if (pt.z <= 0.0f) continue;

    Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
    Eigen::Vector4f p_map = tf_mat * p;

    pcl::PointXYZRGB mp;
    mp.x = p_map[0]; mp.y = p_map[1]; mp.z = p_map[2];
    mp.r = pt.r; mp.g = pt.g; mp.b = pt.b;
    transformed->push_back(mp);
  }

  if (transformed->empty()) return;

  // ---- 单帧质心预滤波 (减少存入窗口的点数) ----
  // 每帧 ~77k 点 → ~5k 点 (~1ms), 150 帧全量从 10M → ~750k
  // 使用与 publishTimer 相同的体素尺寸和质心平均, 不损失质量
  if (voxel_size_ > 0.0) {
    struct FrameVoxelAccum {
      double sx, sy, sz, sr, sg, sb;
      int count;
    };
    struct FVKey {
      int x, y, z;
      bool operator==(const FVKey &o) const {
        return x == o.x && y == o.y && z == o.z;
      }
    };
    struct FVKeyHash {
      size_t operator()(const FVKey &k) const {
        size_t h = 2166136261u;
        h ^= std::hash<int>()(k.x); h *= 16777619u;
        h ^= std::hash<int>()(k.y); h *= 16777619u;
        h ^= std::hash<int>()(k.z); h *= 16777619u;
        return h;
      }
    };
    const float inv_vs = static_cast<float>(1.0 / voxel_size_);

    std::unordered_map<FVKey, FrameVoxelAccum, FVKeyHash> frame_voxels;
    frame_voxels.reserve(transformed->size() / 3);

    for (const auto &pt : transformed->points) {
      FVKey key;
      key.x = static_cast<int>(std::floor(pt.x * inv_vs));
      key.y = static_cast<int>(std::floor(pt.y * inv_vs));
      key.z = static_cast<int>(std::floor(pt.z * inv_vs));

      auto &v = frame_voxels[key];
      v.sx += pt.x;  v.sy += pt.y;  v.sz += pt.z;
      v.sr += pt.r;  v.sg += pt.g;  v.sb += pt.b;
      v.count++;
    }

    auto pre_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    pre_filtered->reserve(frame_voxels.size());
    for (const auto &[key, v] : frame_voxels) {
      float inv_n = 1.0f / v.count;
      pcl::PointXYZRGB pt;
      pt.x = static_cast<float>(v.sx * inv_n);
      pt.y = static_cast<float>(v.sy * inv_n);
      pt.z = static_cast<float>(v.sz * inv_n);
      pt.r = static_cast<uint8_t>(std::clamp(v.sr * inv_n, 0.0, 255.0));
      pt.g = static_cast<uint8_t>(std::clamp(v.sg * inv_n, 0.0, 255.0));
      pt.b = static_cast<uint8_t>(std::clamp(v.sb * inv_n, 0.0, 255.0));
      pre_filtered->push_back(pt);
    }
    pre_filtered->width = pre_filtered->size();
    pre_filtered->height = 1;
    pre_filtered->is_dense = true;
    transformed = pre_filtered;
  } else {
    transformed->width = transformed->size();
    transformed->height = 1;
    transformed->is_dense = true;
  }

  // 添加到 sliding window
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cloud_window_.push_back({msg->header.stamp, transformed});
    while (static_cast<int>(cloud_window_.size()) > max_clouds_) {
      cloud_window_.pop_front();
    }
  }
}

// ---------------------------------------------------------------------------
void SemanticMapNode::publishTimer() {
  auto tp0 = std::chrono::steady_clock::now();
  // ---- 哈希质心去重 (替代 pcl::VoxelGrid) ----
  //
  // pcl::VoxelGrid 内部: 排序 O(n log n) + 质心计算
  // 哈希方案:        遍历 O(n) + 质心计算
  //
  // 关键: 必须用质心平均 (centroid), 不能用 last-write-wins,
  //       否则 TF 漂移导致的重影无法被平滑
  //
  struct VoxelAccum {
    double sx, sy, sz;  // 坐标累计 (double 避免大量相加时精度丢失)
    double sr, sg, sb;  // 颜色累计
    int count;
  };

  // 体素 key: 离散化 (x, y, z)
  struct VKey {
    int x, y, z;
    bool operator==(const VKey &o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct VKeyHash {
    size_t operator()(const VKey &k) const {
      size_t h = 2166136261u;
      h ^= std::hash<int>()(k.x); h *= 16777619u;
      h ^= std::hash<int>()(k.y); h *= 16777619u;
      h ^= std::hash<int>()(k.z); h *= 16777619u;
      return h;
    }
  };

  const float inv_vs = (voxel_size_ > 0.0) ?
      static_cast<float>(1.0 / voxel_size_) : 0.0f;

  std::unordered_map<VKey, VoxelAccum, VKeyHash> voxel_map;
  size_t total_pts = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cloud_window_.empty()) return;

    // 预估体素数量用于 reserve (去重后约 1/3~1/2)
    for (const auto &sc : cloud_window_) total_pts += sc.cloud->size();
    voxel_map.reserve(total_pts / 3);

    // 直接遍历窗口帧 → 哈希累积 (跳过 merge 拷贝)
    for (const auto &sc : cloud_window_) {
      for (const auto &pt : sc.cloud->points) {
        VKey key;
        key.x = static_cast<int>(std::floor(pt.x * inv_vs));
        key.y = static_cast<int>(std::floor(pt.y * inv_vs));
        key.z = static_cast<int>(std::floor(pt.z * inv_vs));

        auto &v = voxel_map[key];
        v.sx += pt.x;  v.sy += pt.y;  v.sz += pt.z;
        v.sr += pt.r;  v.sg += pt.g;  v.sb += pt.b;
        v.count++;
      }
    }
  }

  // ---- 输出质心点云 ----
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  merged->reserve(voxel_map.size());
  for (const auto &[key, v] : voxel_map) {
    float inv_n = 1.0f / v.count;
    pcl::PointXYZRGB pt;
    pt.x = static_cast<float>(v.sx * inv_n);
    pt.y = static_cast<float>(v.sy * inv_n);
    pt.z = static_cast<float>(v.sz * inv_n);
    pt.r = static_cast<uint8_t>(std::clamp(v.sr * inv_n, 0.0, 255.0));
    pt.g = static_cast<uint8_t>(std::clamp(v.sg * inv_n, 0.0, 255.0));
    pt.b = static_cast<uint8_t>(std::clamp(v.sb * inv_n, 0.0, 255.0));
    merged->push_back(pt);
  }

  if (merged->empty()) return;

  auto tp2 = std::chrono::steady_clock::now();

  // ---- 发布 3D 语义地图 ----
  {
    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(*merged, pc2);
    pc2.header.stamp = this->now();
    pc2.header.frame_id = target_frame_;
    map_pub_->publish(pc2);
  }

  // ---- 发布 2D 占据栅格地图 (从 3D 点云投影) ----
  {
    // 计算边界
    float xMin = 1e9f, xMax = -1e9f, yMin = 1e9f, yMax = -1e9f;
    for (const auto &pt : merged->points) {
      if (pt.x < xMin) xMin = pt.x;
      if (pt.x > xMax) xMax = pt.x;
      if (pt.y < yMin) yMin = pt.y;
      if (pt.y > yMax) yMax = pt.y;
    }
    xMin -= grid_cell_size_;  yMin -= grid_cell_size_;
    xMax += grid_cell_size_;  yMax += grid_cell_size_;

    int width  = static_cast<int>((xMax - xMin) / grid_cell_size_) + 1;
    int height = static_cast<int>((yMax - yMin) / grid_cell_size_) + 1;

    if (width <= 0 || height <= 0 || width > 10000 || height > 10000)
      return;

    // 初始化: -1 = unknown
    std::vector<int8_t> grid(width * height, -1);

    // 投影: 按高度分类
    for (const auto &pt : merged->points) {
      int gx = static_cast<int>((pt.x - xMin) / grid_cell_size_);
      int gy = static_cast<int>((pt.y - yMin) / grid_cell_size_);
      if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;

      int idx = gy * width + gx;
      if (pt.z >= grid_min_height_ && pt.z <= grid_max_height_) {
        grid[idx] = 100;  // 障碍
      } else if (grid[idx] != 100) {
        grid[idx] = 0;    // 自由 (不覆盖已标记的障碍)
      }
    }

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp = this->now();
    grid_msg.header.frame_id = target_frame_;
    grid_msg.info.resolution = grid_cell_size_;
    grid_msg.info.width = width;
    grid_msg.info.height = height;
    grid_msg.info.origin.position.x = xMin;
    grid_msg.info.origin.position.y = yMin;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;
    grid_msg.data.assign(grid.begin(), grid.end());
    grid_pub_->publish(grid_msg);
  }

  if (enable_profiling_) {
    auto tp3 = std::chrono::steady_clock::now();
    auto ms_dedup = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp0).count();
    auto ms_pub   = std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp0).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[perf] map: dedup=%ldms pub=%ldms total=%ldms | %zu→%zu voxels, %zu frames",
        ms_dedup, ms_pub, ms_total,
        total_pts, merged->size(), cloud_window_.size());
  }
}

} // namespace semantic_vslam

// ---- main ----
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_vslam::SemanticMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
