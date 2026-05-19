from ultralytics import YOLO
import cv2
import numpy as np
import os

# NudeNet 标签映射
NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# 敏感类别
SENSITIVE_LABELS = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

def get_label_name(class_id):
    """获取标签名称"""
    if 0 <= class_id < len(NUDENET_LABELS):
        return NUDENET_LABELS[class_id]
    return f"UNKNOWN_{class_id}"

def detect_and_annotate(image_path, model_path='models/nudenet_640m.engine', output_path=None):
    """
    使用 NudeNet 模型检测并标注图片
    
    Args:
        image_path: 输入图片路径
        model_path: 模型路径
        output_path: 输出图片路径（默认在原图同目录下添加 _annotated 后缀）
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件不存在 {image_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return None
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path, task='detect')
    
    # 读取图片
    print(f"读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    
    original_height, original_width = image.shape[:2]
    print(f"图片尺寸: {original_width}x{original_height}")
    
    # 检测
    print("正在检测...")
    results = model(image, conf=0.3)
    
    # 获取检测结果
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label_name = get_label_name(cls)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls,
                'class_name': label_name,
                'is_sensitive': label_name in SENSITIVE_LABELS
            })
    
    print(f"检测到 {len(detections)} 个目标")
    
    # 标注图片
    annotated_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        label_name = det['class_name']
        is_sensitive = det['is_sensitive']
        
        # 选择颜色
        if is_sensitive:
            color = (0, 0, 255)  # 红色 - 敏感
            thickness = 3
            label = f"⚠️ {label_name}: {conf:.2f}"
        else:
            color = (0, 255, 255)  # 黄色 - 普通
            thickness = 2
            label = f"{label_name}: {conf:.2f}"
        
        # 绘制边界框
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 5 if y1 - 5 > h else y1 + h + 5
        cv2.rectangle(annotated_image, (x1, label_y - h - 3), (x1 + w, label_y + 3), color, -1)
        
        # 绘制标签文字
        cv2.putText(annotated_image, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 打印检测信息
        print(f"  {label_name}: conf={conf:.3f}, 位置=({x1},{y1})-({x2},{y2}), 敏感={is_sensitive}")
    
    # 保存结果
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"
    
    cv2.imwrite(output_path, annotated_image)
    print(f"标注图片已保存: {output_path}")
    
    # 显示图片（可选）
    cv2.imshow('NudeNet Detection', annotated_image)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detections, annotated_image

def batch_detect(image_dir='.', pattern='debug_roi/debug_roi_*.jpg', model_path='models/nudenet_640m.engine'):
    """
    批量检测目录下的所有匹配图片
    """
    import glob
    
    # 查找所有匹配的图片
    image_files = glob.glob(os.path.join(image_dir, pattern))
    
    if not image_files:
        print(f"未找到匹配的图片: {pattern}")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    for image_path in image_files:
        print(f"\n{'='*50}")
        print(f"处理: {image_path}")
        detect_and_annotate(image_path, model_path)

if __name__ == "__main__":
    # 检测单张图片
    detect_and_annotate(
        image_path='debug_roi_0_0.jpg',  # 你的 ROI 图片路径
        model_path='models/nudenet_640m.engine',
        output_path='debug_roi_0_0_annotated.jpg'  # 输出路径
    )
    
    # 如果有多张 ROI 图片，可以批量处理
    batch_detect(
        image_dir='.',
        pattern='./debug_roi/debug_roi_*.jpg',
        model_path='models/nudenet_640m.engine'
    )