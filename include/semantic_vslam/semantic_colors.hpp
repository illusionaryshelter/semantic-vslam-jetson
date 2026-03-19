#pragma once

/*
 * semantic_colors.hpp
 *
 * COCO 80 类语义颜色表
 * 独立 header，不依赖 ROS2，可被 ROS 节点和独立测试共同使用
 */

#include <array>
#include <cstdint>

namespace semantic_vslam {

static const std::array<std::array<uint8_t, 3>, 80> kSemanticColors = {{
    {{255, 0, 0}},     //  0 person      — 红
    {{0, 255, 0}},     //  1 bicycle     — 绿
    {{0, 0, 255}},     //  2 car         — 蓝
    {{255, 255, 0}},   //  3 motorcycle   — 黄
    {{255, 0, 255}},   //  4 airplane     — 品红
    {{0, 255, 255}},   //  5 bus          — 青
    {{128, 0, 0}},     //  6 train
    {{0, 128, 0}},     //  7 truck
    {{0, 0, 128}},     //  8 boat
    {{128, 128, 0}},   //  9 traffic light
    {{128, 0, 128}},   // 10 fire hydrant
    {{0, 128, 128}},   // 11 stop sign
    {{64, 0, 0}},      // 12 parking meter
    {{0, 64, 0}},      // 13 bench
    {{0, 0, 64}},      // 14 bird
    {{64, 64, 0}},     // 15 cat
    {{64, 0, 64}},     // 16 dog
    {{0, 64, 64}},     // 17 horse
    {{192, 0, 0}},     // 18 sheep
    {{0, 192, 0}},     // 19 cow
    {{0, 0, 192}},     // 20 elephant
    {{192, 192, 0}},   // 21 bear
    {{192, 0, 192}},   // 22 zebra
    {{0, 192, 192}},   // 23 giraffe
    {{128, 64, 0}},    // 24 backpack
    {{0, 128, 64}},    // 25 umbrella
    {{64, 0, 128}},    // 26 handbag
    {{64, 128, 0}},    // 27 tie
    {{128, 0, 64}},    // 28 suitcase
    {{0, 64, 128}},    // 29 frisbee
    {{200, 100, 50}},  // 30 skis
    {{50, 200, 100}},  // 31 snowboard
    {{100, 50, 200}},  // 32 sports ball
    {{150, 100, 50}},  // 33 kite
    {{50, 150, 100}},  // 34 baseball bat
    {{100, 50, 150}},  // 35 baseball glove
    {{200, 150, 50}},  // 36 skateboard
    {{50, 200, 150}},  // 37 surfboard
    {{150, 50, 200}},  // 38 tennis racket
    {{100, 200, 50}},  // 39 bottle
    {{50, 100, 200}},  // 40 wine glass
    {{200, 50, 100}},  // 41 cup
    {{80, 160, 240}},  // 42 fork
    {{240, 160, 80}},  // 43 knife
    {{160, 80, 240}},  // 44 spoon
    {{80, 240, 160}},  // 45 bowl
    {{240, 80, 160}},  // 46 banana
    {{160, 240, 80}},  // 47 apple
    {{120, 60, 180}},  // 48 sandwich
    {{60, 180, 120}},  // 49 orange
    {{180, 120, 60}},  // 50 broccoli
    {{60, 120, 180}},  // 51 carrot
    {{180, 60, 120}},  // 52 hot dog
    {{120, 180, 60}},  // 53 pizza
    {{140, 70, 210}},  // 54 donut
    {{70, 210, 140}},  // 55 cake
    {{210, 140, 70}},  // 56 chair
    {{70, 140, 210}},  // 57 couch
    {{210, 70, 140}},  // 58 potted plant
    {{140, 210, 70}},  // 59 bed
    {{90, 45, 180}},   // 60 dining table
    {{45, 180, 90}},   // 61 toilet
    {{180, 90, 45}},   // 62 tv
    {{45, 90, 180}},   // 63 laptop
    {{180, 45, 90}},   // 64 mouse
    {{90, 180, 45}},   // 65 remote
    {{110, 55, 165}},  // 66 keyboard
    {{55, 165, 110}},  // 67 cell phone
    {{165, 110, 55}},  // 68 microwave
    {{55, 110, 165}},  // 69 oven
    {{165, 55, 110}},  // 70 toaster
    {{110, 165, 55}},  // 71 sink
    {{130, 65, 195}},  // 72 refrigerator
    {{65, 195, 130}},  // 73 book
    {{195, 130, 65}},  // 74 clock
    {{65, 130, 195}},  // 75 vase
    {{195, 65, 130}},  // 76 scissors
    {{130, 195, 65}},  // 77 teddy bear
    {{100, 150, 200}}, // 78 hair dryer
    {{200, 100, 150}}, // 79 toothbrush
}};

} // namespace semantic_vslam
