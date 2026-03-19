# Semantic VSLAM Project

## 项目概述
本项目是一个纯视觉 SLAM 系统，使用ROS2作为统一接口，全程使用 C++。
**核心流程**：RGB 图像输入 -> YOLOv8-seg 分割提取掩码 -> 结合深度/双目信息生成带有语义标签的自定义点云 -> 输入给 RTAB-Map 进行配准和建图。
**目标硬件**：Jetson Orin Nano 8GB（请在编写 YOLOv8-seg 推理代码时，优先考虑使用 TensorRT 以获得实时的端侧性能）。

## 技术栈与依赖
- C++17, CMake
- 核心算法：RTAB-Map, PCL (Point Cloud Library)
- 图像与推理：OpenCV, TensorRT / ONNX Runtime

## 调试与开发规则（严格遵守）
1. **接口友好**： 无需使用复杂的设计模式，因为用户比较熟悉c++，而是充分考虑性能以及良好的注释
2. **防御性编程**：在 OpenCV 矩阵操作、模型加载和点云坐标转换时，必须加入完善的 `try-catch` 块和空指针检查，拒绝静默失败（Silent Failures）。
3. **构建可视化**：在生成 `CMakeLists.txt` 时，默认开启 `set(CMAKE_VERBOSE_MAKEFILE ON)`，确保在编译报错时能看到完整的链接信息，方便调试。
4. **渐进式开发**：禁止一次性生成整个项目的代码。必须按模块推进（例如：先只写 YOLOv8-seg 的推理与掩码可视化，测试通过后再写点云转换，最后接入 RTAB-Map）。


一些bug修订，对于如果yolov8n-seg不能识别类型的部分（例如墙面），仍然应该保留在语义3d点云和2d grid map里，并且所有不能识别类型的部分作为同一个unknown类型即可，否则可能出现3d 点云极其稀少，且2d grid map空旷。目前似乎有些问题，对于物体类型稀疏的情况会出现：“[rtabmap_slam_node-3] [ WARN] (2026-03-10 16:03:53.666) util3d_filtering.cpp:756::voxelizeImpl() Cannot voxelize a not dense (organized) cloud with empty indices! (input=76800 pts). Returning empty cloud!”

ros2 launch semantic_vslam semantic_slam.launch.py \
    engine_path:=$HOME/Desktop/ANTI/models/yolov8n-seg.engine

rviz2 -d $(ros2 pkg prefix semantic_vslam)/share/semantic_vslam/config/semantic_slam.rviz



[ WARN] (2026-03-11 13:56:27.392) util3d_filtering.cpp:756::voxelizeImpl() Cannot voxelize a not dense (organized) cloud with empty indices! (input=710400 pts). Returning empty cloud!
 [ WARN] (2026-03-11 13:56:17.946) OdometryF2M.cpp:319::computeTransform() Failed to find a transformation with the provided guess (xyz=0.331288,0.124574,-0.136400 rpy=0.225166,-0.182120,0.028591), trying again without a guess.
[ WARN] (2026-03-11 13:56:18.162) OdometryF2M.cpp:636::computeTransform() Trial with no guess succeeded!
[rviz]: Message Filter dropping message: frame 'odom' at time 1773208574.764 for reason 'discarding message because the queue is full'
[rviz]: Message Filter dropping message: frame 'camera_color_optical_frame' at time 1773208583.683 for reason 'discarding message because the queue is full'
