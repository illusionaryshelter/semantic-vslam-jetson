/*
 * rtabmap_slam_node.cpp
 *
 * ROS2 节点: 使用 RTAB-Map C++ API 进行视觉 SLAM
 *
 * 核心流程:
 *   1. 接收同步的 RGB + Depth 帧
 *   2. 视觉里程计 (OdometryF2M) 计算帧间位姿
 *   3. Rtabmap::process() 执行闭环检测 + 图优化
 *   4. 提取累积 3D 点云地图并发布
 *   5. 生成 2D 占据栅格地图 (OccupancyGrid) 并发布
 */

#include "semantic_vslam/rtabmap_slam_node.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>

#include <chrono>
#include <map>

using namespace std::chrono_literals;

namespace semantic_vslam {

RtabmapSlamNode::RtabmapSlamNode(const rclcpp::NodeOptions &options)
    : Node("rtabmap_slam_node", options) {

  // ---- 参数 ----
  this->declare_parameter<std::string>("rgb_topic", "/semantic_vslam/rgb");
  this->declare_parameter<std::string>("depth_topic", "/semantic_vslam/depth");
  this->declare_parameter<std::string>("cam_info_topic",
                                       "/camera/color/camera_info");
  this->declare_parameter<std::string>("frame_id", "camera_link");
  this->declare_parameter<std::string>("database_path", "rtabmap.db");
  this->declare_parameter<double>("map_publish_rate", 1.0); // Hz
  this->declare_parameter<double>("voxel_size", 0.02); // m, 地图体素滤波

  std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
  std::string depth_topic = this->get_parameter("depth_topic").as_string();
  std::string cam_info = this->get_parameter("cam_info_topic").as_string();
  frame_id_ = this->get_parameter("frame_id").as_string();
  std::string db_path = this->get_parameter("database_path").as_string();
  double map_rate = this->get_parameter("map_publish_rate").as_double();

  // ---- 初始化 RTAB-Map ----
  ULogger::setType(ULogger::kTypeConsole);
  ULogger::setLevel(ULogger::kWarning);

  rtabmap::ParametersMap params;
  params.insert(rtabmap::ParametersPair(
      rtabmap::Parameters::kDbSqlite3InMemory(), "true"));
  params.insert(rtabmap::ParametersPair(
      rtabmap::Parameters::kRtabmapDetectionRate(), "2"));
  // 内存优化 (适合 Jetson Orin Nano)
  params.insert(rtabmap::ParametersPair(
      rtabmap::Parameters::kMemRehearsalSimilarity(), "0.6"));
  params.insert(rtabmap::ParametersPair(
      rtabmap::Parameters::kRGBDOptimizeMaxError(), "3.0"));

  rtabmap_ = std::make_unique<rtabmap::Rtabmap>();
  rtabmap_->init(params, db_path);

  // 2D 栅格参数
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_max_range", 5.0);
  this->declare_parameter<double>("grid_max_height", 2.0);
  this->declare_parameter<double>("grid_min_height", 0.1);
  grid_cell_size_ = static_cast<float>(this->get_parameter("grid_cell_size").as_double());
  grid_max_range_ = static_cast<float>(this->get_parameter("grid_max_range").as_double());
  grid_max_height_ = static_cast<float>(this->get_parameter("grid_max_height").as_double());
  grid_min_height_ = static_cast<float>(this->get_parameter("grid_min_height").as_double());

  // 视觉里程计 (Feature-to-Map)
  odom_ = std::make_unique<rtabmap::OdometryF2M>();

  RCLCPP_INFO(this->get_logger(), "RTAB-Map initialized. DB: %s",
              db_path.c_str());

  // ---- 订阅相机内参 ----
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      cam_info, 1,
      std::bind(&RtabmapSlamNode::cameraInfoCallback, this,
                std::placeholders::_1));

  // ---- 同步 RGB + Depth ----
  rgb_sub_.subscribe(this, rgb_topic);
  depth_sub_.subscribe(this, depth_topic);
  sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), rgb_sub_, depth_sub_);
  sync_->registerCallback(std::bind(&RtabmapSlamNode::syncCallback, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  // ---- 发布者 ----
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/map_cloud", 1);
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/semantic_vslam/odom", 10);
  grid_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/semantic_vslam/grid_map", 1);

  // ---- 定时发布地图 (3D 点云 + 2D 栅格) ----
  if (map_rate > 0) {
    auto period_ms = static_cast<int>(1000.0 / map_rate);
    map_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(period_ms),
        [this]() {
          publishMapCloud();
          publishGridMap();
        });
  }

  RCLCPP_INFO(this->get_logger(),
              "RtabmapSlamNode ready. Subscribed to: [%s] + [%s]",
              rgb_topic.c_str(), depth_topic.c_str());

  // ---- 订阅语义标签图 (semantic_cloud_node 发布) ----
  label_map_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/semantic_vslam/label_map", 5,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
          auto cv_label = cv_bridge::toCvCopy(msg, "mono8");
          latest_label_map_ = cv_label->image.clone();
        } catch (const cv_bridge::Exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Label map cv_bridge error: %s", e.what());
        }
      });
}

RtabmapSlamNode::~RtabmapSlamNode() {
  if (rtabmap_) {
    rtabmap_->close(true);
    RCLCPP_INFO(this->get_logger(), "RTAB-Map database saved.");
  }
}

// ---------------------------------------------------------------------------
// 相机内参
// ---------------------------------------------------------------------------
void RtabmapSlamNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (has_cam_info_)
    return;

  double fx = msg->k[0];
  double fy = msg->k[4];
  double cx = msg->k[2];
  double cy = msg->k[5];
  int width = static_cast<int>(msg->width);
  int height = static_cast<int>(msg->height);

  camera_model_ =
      rtabmap::CameraModel(fx, fy, cx, cy, rtabmap::Transform::getIdentity(), 0,
                           cv::Size(width, height));

  has_cam_info_ = true;
  RCLCPP_INFO(this->get_logger(),
              "CameraModel set: fx=%.1f fy=%.1f cx=%.1f cy=%.1f %dx%d", fx, fy,
              cx, cy, width, height);
}

// ---------------------------------------------------------------------------
// 同步回调: 每帧 RGB+Depth → VO → RTAB-Map
// ---------------------------------------------------------------------------
void RtabmapSlamNode::syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {

  if (!has_cam_info_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "Waiting for camera_info...");
    return;
  }

  auto t0 = std::chrono::steady_clock::now();

  // 1. 转换为 OpenCV
  cv_bridge::CvImageConstPtr cv_rgb, cv_depth;
  try {
    cv_rgb = cv_bridge::toCvShare(rgb_msg, "bgr8");
    cv_depth = cv_bridge::toCvShare(depth_msg);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    return;
  }

  // 2. 构建 SensorData
  double stamp = rclcpp::Time(rgb_msg->header.stamp).seconds();
  rtabmap::SensorData data(cv_rgb->image.clone(), cv_depth->image.clone(),
                           camera_model_, ++frame_count_, stamp);

  // 3. 视觉里程计
  rtabmap::OdometryInfo odom_info;
  rtabmap::Transform odom_pose = odom_->process(data, &odom_info);

  if (odom_pose.isNull()) {
    RCLCPP_WARN(this->get_logger(), "Odometry lost at frame %d", frame_count_);
    return;
  }

  // 4. 发布里程计 (nav_msgs/Odometry)
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header = rgb_msg->header;
  odom_msg.header.frame_id = "odom";
  odom_msg.child_frame_id = frame_id_;

  float x, y, z, roll, pitch, yaw;
  odom_pose.getTranslationAndEulerAngles(x, y, z, roll, pitch, yaw);
  odom_msg.pose.pose.position.x = x;
  odom_msg.pose.pose.position.y = y;
  odom_msg.pose.pose.position.z = z;
  // 四元数
  rtabmap::Transform rot = odom_pose.rotation();
  Eigen::Matrix3f R;
  R << rot.r11(), rot.r12(), rot.r13(), rot.r21(), rot.r22(), rot.r23(),
      rot.r31(), rot.r32(), rot.r33();
  Eigen::Quaternionf q(R);
  odom_msg.pose.pose.orientation.x = q.x();
  odom_msg.pose.pose.orientation.y = q.y();
  odom_msg.pose.pose.orientation.z = q.z();
  odom_msg.pose.pose.orientation.w = q.w();
  odom_pub_->publish(odom_msg);

  // 5. RTAB-Map 处理 (闭环检测 + 图优化)
  rtabmap_->process(data, odom_pose);

  auto t1 = std::chrono::steady_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  RCLCPP_DEBUG(this->get_logger(), "Frame %d: odom=(%.2f,%.2f,%.2f) %ld ms",
               frame_count_, x, y, z, ms);

  // 6. 缓存当前帧的语义标签图
  // 使用 RTAB-Map 的实际节点 ID (而非 frame_count_) 作为缓存 key
  if (!latest_label_map_.empty()) {
    int rtabmap_node_id = rtabmap_->getStatistics().refImageId();
    if (rtabmap_node_id > 0) {
      label_map_cache_[rtabmap_node_id] = latest_label_map_.clone();
      // 限制缓存大小 (FIFO)
      while (static_cast<int>(label_map_cache_.size()) > kMaxCachedFrames) {
        label_map_cache_.erase(label_map_cache_.begin());
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 定时发布累积 3D 地图 (全场景 + 语义着色)
//
// 流程:
//   1. 从 RTAB-Map 内存重建全场景点云 (所有深度像素)
//   2. 如果有对应的语义 label_map 缓存，重新着色语义区域
//   3. 体素降采样
//   4. 发布为 PointCloud2
// ---------------------------------------------------------------------------
void RtabmapSlamNode::publishMapCloud() {
  if (!rtabmap_) return;

  const std::map<int, rtabmap::Transform> &poses =
      rtabmap_->getLocalOptimizedPoses();

  if (poses.empty())
    return;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  double voxel_size = this->get_parameter("voxel_size").as_double();
  const rtabmap::Memory *memory = rtabmap_->getMemory();
  if (!memory)
    return;

  for (const auto &kv : poses) {
    int id = kv.first;
    if (id <= 0)
      continue;

    rtabmap::SensorData sd = memory->getNodeData(id, true, true, false, false);
    sd.uncompressData();

    if (sd.imageRaw().empty() || sd.depthRaw().empty() ||
        sd.cameraModels().empty())
      continue;

    const int decimation = 2;  // 640/2=320, 480/2=240 → ~76K pts/frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_cloud =
        rtabmap::util3d::cloudFromDepthRGB(sd.imageRaw(), sd.depthRaw(),
                                           sd.cameraModels()[0],
                                           decimation, 10.0f, 0.01f);

    if (local_cloud->empty())
      continue;

    // ---- 语义重新着色 ----
    // 查找与此 RTAB-Map 节点 ID 对应的 label_map
    auto it = label_map_cache_.find(id);
    if (it != label_map_cache_.end() && !it->second.empty()) {
      const cv::Mat &lbl = it->second;
      const rtabmap::CameraModel &cam = sd.cameraModels()[0];
      const float fx = cam.fx();
      const float fy = cam.fy();
      const float cx_cam = cam.cx();
      const float cy_cam = cam.cy();

      // 安全的线性迭代: 通过反投影 (3D→2D) 获取像素坐标
      for (auto &pt : local_cloud->points) {
        if (std::isnan(pt.z) || pt.z <= 0.0f) continue;

        // 反投影: 3D 点 → 原始图像像素坐标
        int img_u = static_cast<int>(std::round(pt.x * fx / pt.z + cx_cam));
        int img_v = static_cast<int>(std::round(pt.y * fy / pt.z + cy_cam));

        if (img_u < 0 || img_u >= lbl.cols || img_v < 0 || img_v >= lbl.rows)
          continue;

        uint8_t label_val = lbl.at<uint8_t>(img_v, img_u);
        if (label_val > 0) {
          int cls = label_val - 1;
          if (cls >= 0 && cls < 80) {
            pt.r = kSemanticColors[cls][0];
            pt.g = kSemanticColors[cls][1];
            pt.b = kSemanticColors[cls][2];
          }
        }
        // label_val == 0: 保留原始 RGB
      }
    }

    // 变换到全局坐标系
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed =
        rtabmap::util3d::transformPointCloud(local_cloud, kv.second);

    *map_cloud += *transformed;
  }

  // 移除 NaN 点 (cloudFromDepthRGB 对无效深度生成 NaN 点)
  // voxelize 无法处理含 NaN 的 organized cloud, 会返回空点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clean_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  clean_cloud->reserve(map_cloud->size());
  for (const auto &pt : map_cloud->points) {
    if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
      clean_cloud->push_back(pt);
    }
  }
  clean_cloud->width = clean_cloud->size();
  clean_cloud->height = 1;
  clean_cloud->is_dense = true;
  map_cloud = clean_cloud;

  // 体素降采样
  if (voxel_size > 0 && !map_cloud->empty()) {
    map_cloud = rtabmap::util3d::voxelize(map_cloud, voxel_size);
  }

  if (map_cloud->empty())
    return;

  // 转为 ROS2 消息
  sensor_msgs::msg::PointCloud2 pc2;
  pcl::toROSMsg(*map_cloud, pc2);
  pc2.header.stamp = this->now();
  pc2.header.frame_id = "map";
  map_pub_->publish(pc2);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "Published semantic map cloud: %zu points, %zu nodes",
                       map_cloud->size(), poses.size());
}

// ---------------------------------------------------------------------------
// 定时发布 2D 占据栅格地图 (直接从 3D 地图投影)
//
// 原理: 遍历 RTAB-Map 所有优化位姿节点的深度数据，
// 生成全局3D点后投影到 XY 平面:
//   - 高度 < grid_min_height_ → 地面 (free)
//   - 高度 ∈ [grid_min_height_, grid_max_height_] → 障碍 (occupied)
//   - 其他: 保持 unknown
// ---------------------------------------------------------------------------
void RtabmapSlamNode::publishGridMap() {
  if (!rtabmap_ || grid_map_pub_->get_subscription_count() == 0)
    return;

  const std::map<int, rtabmap::Transform> &poses =
      rtabmap_->getLocalOptimizedPoses();
  if (poses.empty())
    return;

  const rtabmap::Memory *memory = rtabmap_->getMemory();
  if (!memory)
    return;

  // 收集所有 3D 点并投影
  float xMin = std::numeric_limits<float>::max();
  float xMax = std::numeric_limits<float>::lowest();
  float yMin = std::numeric_limits<float>::max();
  float yMax = std::numeric_limits<float>::lowest();

  // 先做一次扫描确定边界
  struct GridPoint { float x; float y; bool obstacle; };
  std::vector<GridPoint> grid_points;
  grid_points.reserve(100000);

  for (const auto &kv : poses) {
    int id = kv.first;
    if (id <= 0) continue;

    rtabmap::SensorData sd = memory->getNodeData(id, true, true, false, false);
    sd.uncompressData();
    if (sd.imageRaw().empty() || sd.depthRaw().empty() || sd.cameraModels().empty())
      continue;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
        rtabmap::util3d::cloudFromDepthRGB(
            sd.imageRaw(), sd.depthRaw(), sd.cameraModels()[0],
            8, grid_max_range_, 0.01f);  // decimation=8 for speed (grid only)
    if (cloud->empty()) continue;

    // 变换到全局坐标系
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed =
        rtabmap::util3d::transformPointCloud(cloud, kv.second);

    for (const auto &pt : transformed->points) {
      if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) continue;
      float dist = std::sqrt(pt.x*pt.x + pt.y*pt.y);
      if (dist > grid_max_range_) continue;

      bool is_obstacle = (pt.z >= grid_min_height_ && pt.z <= grid_max_height_);
      grid_points.push_back({pt.x, pt.y, is_obstacle});

      xMin = std::min(xMin, pt.x);
      xMax = std::max(xMax, pt.x);
      yMin = std::min(yMin, pt.y);
      yMax = std::max(yMax, pt.y);
    }
  }

  if (grid_points.empty()) return;

  // 添加边距
  xMin -= grid_cell_size_;
  yMin -= grid_cell_size_;
  xMax += grid_cell_size_;
  yMax += grid_cell_size_;

  int width = static_cast<int>((xMax - xMin) / grid_cell_size_) + 1;
  int height = static_cast<int>((yMax - yMin) / grid_cell_size_) + 1;

  if (width <= 0 || height <= 0 || width > 10000 || height > 10000)
    return;

  // 初始化: -1 = unknown
  std::vector<int8_t> data(width * height, -1);

  // 填充栅格
  for (const auto &gp : grid_points) {
    int gx = static_cast<int>((gp.x - xMin) / grid_cell_size_);
    int gy = static_cast<int>((gp.y - yMin) / grid_cell_size_);
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;

    int idx = gy * width + gx;
    if (gp.obstacle) {
      data[idx] = 100;  // occupied
    } else if (data[idx] != 100) {
      data[idx] = 0;    // free (不覆盖已标记的 occupied)
    }
  }

  // 发布
  nav_msgs::msg::OccupancyGrid grid_msg;
  grid_msg.header.stamp = this->now();
  grid_msg.header.frame_id = "map";
  grid_msg.info.resolution = grid_cell_size_;
  grid_msg.info.width = width;
  grid_msg.info.height = height;
  grid_msg.info.origin.position.x = xMin;
  grid_msg.info.origin.position.y = yMin;
  grid_msg.info.origin.position.z = 0.0;
  grid_msg.info.origin.orientation.w = 1.0;
  grid_msg.data = std::move(data);

  grid_map_pub_->publish(grid_msg);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Published grid map: %dx%d cells, %zu points projected",
      width, height, grid_points.size());
}

} // namespace semantic_vslam

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_vslam::RtabmapSlamNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
