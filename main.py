from ultralytics import YOLO
import cv2
import os
import tkinter as tk
from tkinter import ttk

class DetectionControlPanel:
    def __init__(self, nudenet_labels, sensitive_labels):
        """
        GUI 控制面板：选择遮蔽方式和要处理的类别

        Args:
            nudenet_labels: 所有标签列表
            sensitive_labels: 敏感标签列表
        """
        self.nudenet_labels = nudenet_labels
        self.sensitive_labels = sensitive_labels

        # 创建根窗口
        self.root = tk.Tk()
        self.root.title("检测控制面板")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._running = True
        self._started = False

        # 遮蔽方式
        self._censor_mode = tk.StringVar(value="black")  # black, blur, pixelate, distortion
        
        # 遮蔽参数
        self._blur_kernel = tk.IntVar(value=31)          # 高斯模糊核大小
        self._pixel_size = tk.IntVar(value=10)           # 像素化块大小
        self._distortion_strength = tk.IntVar(value=15)  # 失真强度
        
        # 标签选择
        self._label_vars = {}

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== 遮蔽方式选择 ==========
        mode_frame = ttk.LabelFrame(main_frame, text="遮蔽方式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        modes = [
            ("黑块遮蔽", "black"),
            ("高斯模糊", "blur"),
            ("像素化", "pixelate"),
            ("失真效果", "distortion"),
        ]
        
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self._censor_mode, 
                          value=value, command=self._on_mode_change).pack(anchor='w', pady=2)

        # 参数调节区
        self.params_frame = ttk.LabelFrame(mode_frame, text="参数调节", padding=5)
        self.params_frame.pack(fill=tk.X, pady=(10, 0))

        # 高斯模糊参数（默认显示）
        self.blur_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.blur_frame, text="模糊核大小 (奇数):").pack(side=tk.LEFT)
        ttk.Scale(self.blur_frame, from_=5, to=99, variable=self._blur_kernel, 
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.blur_frame, textvariable=self._blur_kernel).pack(side=tk.LEFT)

        # 像素化参数
        self.pixel_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.pixel_frame, text="像素块大小:").pack(side=tk.LEFT)
        ttk.Scale(self.pixel_frame, from_=2, to=50, variable=self._pixel_size, 
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.pixel_frame, textvariable=self._pixel_size).pack(side=tk.LEFT)

        # 失真参数
        self.distortion_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.distortion_frame, text="扭曲强度:").pack(side=tk.LEFT)
        ttk.Scale(self.distortion_frame, from_=5, to=50, variable=self._distortion_strength, 
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.distortion_frame, textvariable=self._distortion_strength).pack(side=tk.LEFT)

        self._on_mode_change()  # 初始化显示

        ttk.Separator(main_frame).pack(fill=tk.X, pady=10)

        # ========== 类别选择 ==========
        class_frame = ttk.LabelFrame(main_frame, text="选择要遮蔽的类别", padding=5)
        class_frame.pack(fill=tk.BOTH, expand=True)

        # 全选/取消全选按钮
        btn_frame = ttk.Frame(class_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="取消全选", command=self._deselect_all).pack(side=tk.LEFT)

        # 画布+滚动条
        canvas = tk.Canvas(class_frame, height=300)
        scrollbar = ttk.Scrollbar(class_frame, orient=tk.VERTICAL, command=canvas.yview)
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
        self.status_label = ttk.Label(main_frame, text="请选择遮蔽方式和要处理的类别后点击开始", foreground="gray")
        self.status_label.pack()

    def _on_mode_change(self):
        """遮蔽方式改变时显示对应的参数"""
        mode = self._censor_mode.get()
        
        # 隐藏所有参数
        for frame in [self.blur_frame, self.pixel_frame, self.distortion_frame]:
            frame.pack_forget()
        
        # 显示对应参数
        if mode == "blur":
            self.blur_frame.pack(fill=tk.X, pady=5)
        elif mode == "pixelate":
            self.pixel_frame.pack(fill=tk.X, pady=5)
        elif mode == "distortion":
            self.distortion_frame.pack(fill=tk.X, pady=5)

    def _select_all(self):
        for var in self._label_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._label_vars.values():
            var.set(False)

    def _on_start(self):
        self._started = True
        self.start_btn.configure(text="●  检测中...", state="disabled")
        selected_count = sum(1 for v in self._label_vars.values() if v.get())
        self.status_label.configure(
            text=f"已开始 | 遮蔽方式: {self.get_censor_mode_name()} | 选中 {selected_count} 个类别",
            foreground="green"
        )

    def is_started(self):
        return self._started

    def should_draw(self, class_name):
        """判断是否需要对某个类别进行遮蔽"""
        if not self._running:
            return False
        if class_name in self._label_vars:
            return self._label_vars[class_name].get()
        return True

    def get_censor_mode(self):
        """获取当前遮蔽方式"""
        return self._censor_mode.get()

    def get_censor_mode_name(self):
        """获取遮蔽方式中文名"""
        names = {
            "black": "黑块",
            "blur": "高斯模糊",
            "pixelate": "像素化",
            "distortion": "失真"
        }
        return names.get(self._censor_mode.get(), "未知")

    def get_censor_params(self):
        """获取当前遮蔽参数"""
        mode = self._censor_mode.get()
        if mode == "blur":
            return {"kernel_size": self._blur_kernel.get()}
        elif mode == "pixelate":
            return {"pixel_size": self._pixel_size.get()}
        elif mode == "distortion":
            return {"strength": self._distortion_strength.get()}
        return {}

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

import cv2
import numpy as np

class CensorEffects:
    """遮蔽效果工具类"""
    
    @staticmethod
    def black_block(frame, x1, y1, x2, y2):
        """黑块遮蔽"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return frame
    
    @staticmethod
    def gaussian_blur(frame, x1, y1, x2, y2, kernel_size=31):
        """高斯模糊遮蔽"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        # 确保 kernel_size 为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        frame[y1:y2, x1:x2] = blurred
        return frame
    
    @staticmethod
    def pixelate(frame, x1, y1, x2, y2, pixel_size=10):
        """像素化遮蔽"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        h, w = roi.shape[:2]
        # 缩小
        temp = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), 
                         interpolation=cv2.INTER_LINEAR)
        # 放大
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = pixelated
        return frame
    
    @staticmethod
    def distortion(frame, x1, y1, x2, y2, strength=15):
        """失真效果（网格扭曲）"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        h, w = roi.shape[:2]
        
        # 创建网格
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # 添加随机偏移
        noise_x = np.random.randint(-strength, strength, (h, w))
        noise_y = np.random.randint(-strength, strength, (h, w))
        
        map_x = (grid_x + noise_x).astype(np.float32)
        map_y = (grid_y + noise_y).astype(np.float32)
        
        # 边界裁剪
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = distorted
        return frame





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
        """根据 GUI 选择对检测区域进行遮蔽"""
        mode = self.control_panel.get_censor_mode()
        params = self.control_panel.get_censor_params()

        for det in detections:
            # 检查是否需要处理
            if self.control_panel is not None and not self.control_panel.should_draw(det['class_name']):
                continue

            x1, y1, x2, y2 = det['bbox']

            # 根据选择的模式进行遮蔽
            if mode == "black":
                CensorEffects.black_block(frame, x1, y1, x2, y2)
            elif mode == "blur":
                kernel = params.get("kernel_size", 31)
                CensorEffects.gaussian_blur(frame, x1, y1, x2, y2, kernel_size=kernel)
            elif mode == "pixelate":
                pixel = params.get("pixel_size", 10)
                CensorEffects.pixelate(frame, x1, y1, x2, y2, pixel_size=pixel)
            elif mode == "distortion":
                strength = params.get("strength", 15)
                CensorEffects.distortion(frame, x1, y1, x2, y2, strength=strength)

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