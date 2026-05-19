from ultralytics import YOLO

# 加载模型
model = YOLO('models/face-parts-yolov8n.pt')
# model = YOLO('models/yolov9_e_wholebody34_0100_1x3x640x640.onnx')

# 直接导出 TensorRT
model.export(
    format='engine',      # 导出为 TensorRT
    device=0,             # GPU 设备
    half=True,            # FP16 精度
    # imgsz=640,            # 输入尺寸
    imgsz=(320, 320),
    batch=1,              # 批次
    workspace=8,          # 工作空间 (GB)
    verbose=True
)

print("转换完成！模型保存为: models/nudenet_640m.engine")