"""
test_rtabmap_standalone.launch.py

独立测试 RTAB-Map 功能: 官方 rtabmap_ros + Astra Pro，不依赖 semantic_vslam。

用途: 验证 RTAB-Map SLAM 在 Astra Pro 硬件上是否正常工作 (里程计、3D 建图、2D 栅格)

用法:
  ros2 launch semantic_vslam test_rtabmap_standalone.launch.py

  # 或直接用 launch 文件路径:
  ros2 launch /home/shelter/Desktop/ANTI/launch/test_rtabmap_standalone.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import AnyLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([

        # ---- 参数 ----
        DeclareLaunchArgument('launch_camera', default_value='true',
            description='Whether to launch Astra Pro camera driver'),
        DeclareLaunchArgument('rviz', default_value='true',
            description='Launch RViz2 for visualization'),
        DeclareLaunchArgument('rtabmap_viz', default_value='false',
            description='Launch RTAB-Map built-in visualizer'),
        DeclareLaunchArgument('database_path', default_value='/tmp/rtabmap_test.db',
            description='Path to RTAB-Map database file'),

        # ============================================================
        # 1. Astra Pro 摄像头驱动
        # ============================================================
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('astra_camera'),
                    'launch',
                    'astra_pro.launch.xml'
                ])
            ),
            launch_arguments={
                'camera_name': 'camera',
                'depth_registration': 'true',
                'color_depth_synchronization': 'true',
                'color_width': '640',
                'color_height': '480',
                'color_fps': '30',
                'depth_width': '640',
                'depth_height': '480',
                'depth_fps': '30',
                'enable_point_cloud': 'false',
                'enable_colored_point_cloud': 'false',
            }.items(),
            condition=IfCondition(LaunchConfiguration('launch_camera')),
        ),

        # ============================================================
        # 2. 官方 RTAB-Map (rtabmap_ros) — SLAM + 里程计 + 可视化
        #
        # 这里直接使用官方的 rtabmap.launch.py，
        # 将 topic 重映射到 Astra Pro 的发布名称
        # ============================================================
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('rtabmap_launch'),
                    'launch',
                    'rtabmap.launch.py'
                ])
            ),
            launch_arguments={
                # ---- Topic 映射 (Astra Pro) ----
                'rgb_topic':            '/camera/color/image_raw',
                'depth_topic':          '/camera/depth/image_raw',
                'camera_info_topic':    '/camera/color/camera_info',

                # ---- 坐标系 ----
                'frame_id':             'camera_link',

                # ---- 数据库 ----
                'database_path':        LaunchConfiguration('database_path'),

                # ---- 可视化 ----
                'rviz':                 LaunchConfiguration('rviz'),
                'rtabmap_viz':          LaunchConfiguration('rtabmap_viz'),

                # ---- RTAB-Map 参数 ----
                'approx_sync':          'true',       # 近似时间同步 (必须)
                'subscribe_depth':      'true',       # RGB-D 模式
                'queue_size':           '20',
                'qos':                  '1',          # best-effort to match camera

                # ---- 里程计 ----
                'visual_odometry':      'true',       # 使用视觉里程计
                'odom_frame_id':        'odom',

                # ---- 参数 (加快适应 Jetson) ----
                'args': '--delete_db_on_start '       # 每次启动清除旧数据库
                        '--Rtabmap/DetectionRate 2 '   # 2Hz 检测率
                        '--Mem/RehearsalSimilarity 0.6 '
                        '--RGBD/OptimizeMaxError 3.0',
            }.items(),
        ),
    ])
