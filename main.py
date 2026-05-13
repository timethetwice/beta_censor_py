from ultralytics import YOLO
import cv2
import os
import tkinter as tk
from tkinter import ttk

class DetectionControlPanel:
    def __init__(self, nudenet_labels, sensitive_labels):
        """
        GUI 控制面板：先选择要绘制的类别，点击开始后再检测

        Args:
            nudenet_labels: 所有标签列表
            sensitive_labels: 敏感标签列表
        """
        self.nudenet_labels = nudenet_labels
        self.sensitive_labels = sensitive_labels

        # 创建根窗口
        self.root = tk.Tk()
        self.root.title("检测框显示控制")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._running = True
        self._started = False  # 是否已点击开始

        # 创建变量
        self._label_vars = {}

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)


        ttk.Separator(main_frame).pack(fill=tk.X, pady=5)

        # 逐类别控制
        detail_frame = ttk.LabelFrame(main_frame, text="逐类别控制", padding=5)
        detail_frame.pack(fill=tk.BOTH, expand=True)

        # 全选/取消全选按钮
        btn_frame = ttk.Frame(detail_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="取消全选", command=self._deselect_all).pack(side=tk.LEFT)

        # 画布+滚动条
        canvas = tk.Canvas(detail_frame, height=400)
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 为每个标签创建复选框
        self._label_vars = {}
        for label in self.nudenet_labels:
            var = tk.BooleanVar(value=True)
            self._label_vars[label] = var
            is_sensitive = label in self.sensitive_labels
            display_text = f"{'⚠️' if is_sensitive else '✓'} {label}"
            cb = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var)
            cb.pack(anchor='w', pady=1)

        ttk.Separator(main_frame).pack(fill=tk.X, pady=10)

        # 开始按钮
        self.start_btn = ttk.Button(main_frame, text="▶  开始检测", command=self._on_start)
        self.start_btn.pack(pady=5)

        # 状态标签
        self.status_label = ttk.Label(main_frame, text="请在开始前选择要显示的检测框类别", foreground="gray")
        self.status_label.pack()

    def _select_all(self):
        for var in self._label_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._label_vars.values():
            var.set(False)

    def _on_start(self):
        """点击开始按钮"""
        self._started = True
        self.start_btn.configure(text="●  检测中...", state="disabled")
        selected_count = sum(1 for v in self._label_vars.values() if v.get())
        self.status_label.configure(
            text=f"已开始检测 | 选中 {selected_count}/{len(self._label_vars)} 个类别",
            foreground="green"
        )

    def is_started(self):
        """检查是否已点击开始"""
        return self._started

    def should_draw(self, class_name):
        """
        判断某个检测框是否应该绘制
        Args:
            class_name: 类别名称（如 'FEMALE_BREAST_EXPOSED'）
        Returns:
            bool: 是否绘制
        """
        if not self._running:
            return False

        is_sensitive = class_name in self.sensitive_labels


        # 逐类别检查
        if class_name in self._label_vars:
            return self._label_vars[class_name].get()
        return True  # 未知类别默认显示

    def update(self):
        """更新 tkinter 事件循环（非阻塞）"""
        if self._running:
            self.root.update()
        return self._running

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def close(self):
        self._running = False
        self.root.destroy()







class RealtimeCascadeDetector:
    def __init__(self, fast_model='models/yolo26n.engine', precise_model='models/nudenet_640m.engine'):
        self.fast_model = YOLO(fast_model, task='detect')
        self.precise_model = YOLO(precise_model, task='detect')

        # NudeNet 标签映射
        self.nudenet_labels = [
            "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
            "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED",
            "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
            "MALE_GENITALIA_EXPOSED", "ANUS_COVERED",
            "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
        ]
        self.sensitive_labels = [
            "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED",
        ]
        # 新增：控制面板（延迟初始化，避免影响模型加载）
        self.control_panel = None

    def get_label_name(self, class_id):
        if 0 <= class_id < len(self.nudenet_labels):
            return self.nudenet_labels[class_id]
        return f"UNKNOWN_{class_id}"

    def draw_results(self, frame, detections):
        for det in detections:
            # 检查是否应该绘制
            if self.control_panel is not None and not self.control_panel.should_draw(det['class_name']):
                continue
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['is_sensitive'] else (0, 255, 255)
            thickness = 3 if det['is_sensitive'] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            if det['is_sensitive']:
                label = "⚠️ " + label

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def run(self, video_source=0, output_path="output.mp4"):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        os.makedirs('./debug_roi', exist_ok=True)
        frame_id = 0
        # 新增：初始化控制面板
        self.control_panel = DetectionControlPanel(
            nudenet_labels=self.nudenet_labels,
            sensitive_labels=self.sensitive_labels
        )

        # ⚠️ 关键：等待用户点击"开始"按钮
        print("等待用户选择检测类别并点击开始...")
        while self.control_panel.is_started() == False:
            if not self.control_panel.update():
                print("用户关闭了控制面板，退出程序")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return
            # 直接休眠，让出 CPU
            import time
            time.sleep(0.05)

        print("开始检测！")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ----------------- 第一阶段：检测人 -----------------
            results_fast = self.fast_model(frame, conf=0.3, classes=[0])
            candidates = []
            h, w = frame.shape[:2]
            for box in results_fast[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pad = 50
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                candidates.append((x1, y1, x2, y2))

            # ----------------- 第二阶段：NudeNet 检测 -----------------
            final_detections = []
            for idx, (x1, y1, x2, y2) in enumerate(candidates):
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 关键：不手动 resize，直接送入 ROI，且指定 imgsz 与引擎输出一致
                precise_results = self.precise_model(
                    roi,
                    conf=0.3,
                    imgsz=(640, 384)   # 高640，宽384
                )

                for pr in precise_results[0].boxes:
                    rx1, ry1, rx2, ry2 = map(int, pr.xyxy[0].tolist())
                    conf = float(pr.conf[0])
                    cls = int(pr.cls[0])
                    label_name = self.get_label_name(cls)

                    final_detections.append({
                        'bbox': (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2),
                        'class': cls,
                        'class_name': label_name,
                        'confidence': conf,
                        'is_sensitive': label_name in self.sensitive_labels
                    })

                # 调试保存（带检测框的 ROI）
                if frame_id % 30 == 0:
                    roi_annotated = roi.copy()
                    # 只画属于当前 roi 的检测框
                    for det in final_detections:
                        # 检测框坐标是全局的，需要转换回 roi 内坐标
                        gx1, gy1, gx2, gy2 = det['bbox']
                        # 检查这个框是否落在当前 roi 区域内（简单包含判断）
                        if gx1 >= x1 and gy1 >= y1 and gx2 <= x2 and gy2 <= y2:
                            rx1 = gx1 - x1
                            ry1 = gy1 - y1
                            rx2 = gx2 - x1
                            ry2 = gy2 - y1
                            color = (0, 0, 255) if det['is_sensitive'] else (0, 255, 255)
                            cv2.rectangle(roi_annotated, (rx1, ry1), (rx2, ry2), color, 2)
                            label = f"{det['class_name']}: {det['confidence']:.2f}"
                            cv2.putText(roi_annotated, label, (rx1, ry1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.imwrite(f'./debug_roi/debug_roi_{frame_id}_{idx}.jpg', roi_annotated)

        # ----------------- 绘制并保存 -----------------
            annotated = self.draw_results(frame, final_detections)
            out.write(annotated)
            cv2.imshow('Cascade Detection', annotated)
            frame_id += 1

            # 新增：更新控制面板（替代原有简单的 waitKey）
            if not self.control_panel.update():
                break  # GUI 关闭时退出循环

            # 仍然保留键盘退出方式
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 清理
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if self.control_panel is not None:
            self.control_panel.close()
        print("处理完成")


if __name__ == "__main__":
    detector = RealtimeCascadeDetector(
        fast_model='models/yolo26n.engine',           # 快速检测人
        precise_model='models/nudenet_640m.engine' # NudeNet 敏感内容检测
    )
    
    detector.run(
        video_source="./test/【兰幼金】夏日辣妹Bubble Pop❤超元气可爱小马达！ P1 横屏   - 0.00.13-0.00.23.mp4",  # 你的视频路径
        output_path="nudenet_result.mp4"
    )