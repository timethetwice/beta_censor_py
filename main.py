from ultralytics import YOLO
import cv2
import numpy as np
from threading import Thread
from queue import Queue, Empty

class RealtimeCascadeDetector:
    def __init__(self, fast_model='models/yolo26n.engine', precise_model='models/nudenet_640m.engine'):
        """
        级联检测器
        fast_model: 快速检测模型（检测人）
        precise_model: NudeNet 敏感内容检测模型
        """
        self.fast_model = YOLO(fast_model, task='detect')
        # NudeNet 模型也需要指定 task='detect'
        self.precise_model = YOLO(precise_model, task='detect')
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.running = False
        
        # NudeNet 标签映射
        self.nudenet_labels = [
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
        
        # 敏感类别列表（可根据需要调整）
        self.sensitive_labels = [
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
        ]
    
    def get_label_name(self, class_id):
        """获取标签名称"""
        if 0 <= class_id < len(self.nudenet_labels):
            return self.nudenet_labels[class_id]
        return f"UNKNOWN_{class_id}"
    
    def frame_processor(self):
        """帧处理线程"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                frame, frame_id = frame_data
                
                # 第一阶段：检测人
                results = self.fast_model(frame, conf=0.3, classes=[0])
                
                candidates = []
                height, width = frame.shape[:2]
                
                for r in results[0].boxes:
                    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                    # 扩大边界，确保人体完整
                    padding = 50
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)
                    candidates.append((x1, y1, x2, y2))
                
                # 第二阶段：NudeNet 敏感内容检测
                final_results = []
                for (x1, y1, x2, y2) in candidates:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # 使用 NudeNet 模型检测
                    precise_results = self.precise_model(roi, conf=0.3)
                    
                    for pr in precise_results[0].boxes:
                        rx1, ry1, rx2, ry2 = map(int, pr.xyxy[0].tolist())
                        conf = float(pr.conf[0])
                        cls = int(pr.cls[0])
                        label_name = self.get_label_name(cls)
                        
                        final_results.append({
                            'bbox': (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2),
                            'class': cls,
                            'class_name': label_name,
                            'confidence': conf,
                            'is_sensitive': label_name in self.sensitive_labels
                        })
                
                self.result_queue.put((frame_id, final_results))
                
            except Empty:
                continue
            except Exception as e:
                print(f"处理错误: {e}")
    
    def draw_results(self, frame, detections):
        """绘制检测框（不同颜色表示不同敏感度）"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # 根据敏感度选择颜色
            if det['is_sensitive']:
                color = (0, 0, 255)  # 红色 - 敏感内容
                thickness = 3
            else:
                color = (0, 255, 255)  # 黄色 - 普通内容
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 添加标签
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            if det['is_sensitive']:
                label = "⚠️ " + label
            
            # 计算文字位置
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def start(self, video_source=0, output_path="output.mp4"):
        """启动级联检测"""
        self.running = True
        
        # 启动处理线程
        processor_thread = Thread(target=self.frame_processor)
        processor_thread.start()
        
        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"错误：无法打开视频源 {video_source}")
            self.running = False
            return
        
        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕")
                break
            
            # 存入队列
            try:
                self.frame_queue.put((frame, frame_id), timeout=1)
            except:
                pass
            
            # 获取检测结果
            try:
                _, results = self.result_queue.get(timeout=0.05)
                frame = self.draw_results(frame, results)
            except Empty:
                pass
            
            # 写入和显示
            out.write(frame)
            cv2.imshow('NudeNet Detection', frame)
            frame_id += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.running = False
        processor_thread.join()


if __name__ == "__main__":
    detector = RealtimeCascadeDetector(
        fast_model='models/yolo26n.engine',           # 快速检测人
        precise_model='models/nudenet_640m.engine' # NudeNet 敏感内容检测
    )
    
    detector.start(
        video_source="./test/【兰幼金】夏日辣妹Bubble Pop❤超元气可爱小马达！ P1 横屏   - 0.00.13-0.00.23.mp4",  # 你的视频路径
        output_path="nudenet_result.mp4"
    )