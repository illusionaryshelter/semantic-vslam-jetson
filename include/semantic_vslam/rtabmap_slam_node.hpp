#pragma once

/*
 * rtabmap_slam_node.hpp
 *
 * ROS2 节点: 使用 RTAB-Map C++ API 进行视觉 SLAM
 *
 * 发布:
 *   /semantic_vslam/map_cloud    — 累积的 3D 语义地图 (PointCloud2)
 *   /semantic_vslam/odom         — 视觉里程计 (nav_msgs/Odometry)
 *   /semantic_vslam/grid_map     — 2D 占据栅格地图 (nav_msgs/OccupancyGrid)
 */

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/odometry/OdometryF2M.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UStl.h>

#include "semantic_vslam/semantic_colors.hpp"

#include <memory>
#include <string>
#include <map>

namespace semantic_vslam {

class RtabmapSlamNode : public rclcpp::Node {
public:
  explicit RtabmapSlamNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~RtabmapSlamNode() override;

private:
  void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
                    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void publishMapCloud();
  void publishGridMap();

  // RTAB-Map 核心
  std::unique_ptr<rtabmap::Rtabmap> rtabmap_;
  std::unique_ptr<rtabmap::OdometryF2M> odom_;
  rtabmap::CameraModel camera_model_;

  // ROS2 通信
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                      sensor_msgs::msg::Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<Synchronizer> sync_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub_;

  rclcpp::TimerBase::SharedPtr map_timer_;

  bool has_cam_info_ = false;
  std::string frame_id_ = "camera_link";
  int frame_count_ = 0;

  // 语义标签图缓存: frame_count → label_map (CV_8UC1)
  // 用于在 publishMapCloud/publishGridMap 中对重建的点云进行语义重新着色
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr label_map_sub_;
  std::map<int, cv::Mat> label_map_cache_;
  cv::Mat latest_label_map_;  // 最新一帧的 label_map
  static constexpr int kMaxCachedFrames = 200;  // 限制缓存大小

  // 栅格参数
  float grid_cell_size_ = 0.05f;      // 5cm
  float grid_max_range_ = 5.0f;       // 最大投影距离
  float grid_max_height_ = 2.0f;      // 障碍物最大高度
  float grid_min_height_ = 0.1f;      // 地面最大高度
};

} // namespace semantic_vslam
