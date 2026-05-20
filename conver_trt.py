from ultralytics import YOLO

# 加载模型
# model = YOLO('models/face-parts-yolov8n.pt')
model = YOLO('models/nudenet_640m.pt')
# model = YOLO('models/yolov9_e_wholebody34_0100_1x3x640x640.onnx')

# 直接导出 TensorRT
model.export(
    format='engine',      # 导出为 TensorRT
    device=0,             # GPU 设备
    half=True,            # FP16 精度
    # imgsz=320,            # 输入尺寸
    imgsz=(640, 384),
    batch=2,              # 批次
    workspace=4,          # 工作空间 (GB)
    dynamic=True,
    verbose=True
)

print("转换完成！模型保存为: models/nudenet_640m.engine")