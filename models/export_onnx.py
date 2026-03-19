from ultralytics import YOLO

# 加载你的 PyTorch 权重文件
model = YOLO("yolov8n-seg.pt")

# 导出为 ONNX 格式
# 注意：作为 C++ 新手，强烈建议不要开启 dynamic (动态输入尺寸)
# 固定为 640x640 能让你后续在 C++ 里写 cudaMalloc (显存分配) 时简单十倍！
model.export(
    format="onnx",
    imgsz=640,
    opset=12,        # TensorRT 对 opset 12 的支持最稳定
    simplify=True    # 极其重要！简化计算图，防止 TensorRT 编译报错
)
