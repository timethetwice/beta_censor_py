from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import os
import numpy as np

class DetectionControlPanel:
    def __init__(self, nudenet_labels, sensitive_labels, female_sensitive_labels, default_video=""):
        """
        GUI 控制面板：选择遮蔽方式和要处理的类别

        Args:
            nudenet_labels: 所有标签列表
            sensitive_labels: 敏感标签列表
            female_sensitive_labels: 女性
            default_video: 默认视频路径（可选）
        """
        self.nudenet_labels = nudenet_labels
        self.sensitive_labels = sensitive_labels
        self.female_sensitive_labels = female_sensitive_labels

        # 创建根窗口
        self.root = tk.Tk()
        self.root.title("检测控制面板")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._running = True
        self._started = False

        # 视频路径
        self._video_path = tk.StringVar(value=default_video)
        self._output_path = tk.StringVar(value="output_censored.mp4")

        # 遮蔽方式
        self._censor_mode = tk.StringVar(value="black")
        
        # 遮蔽参数
        self._blur_kernel = tk.IntVar(value=31)
        self._pixel_size = tk.IntVar(value=10)
        self._distortion_strength = tk.IntVar(value=15)
        self._buffer_frames = tk.IntVar(value=5)  
        
        # 标签选择
        self._label_vars = {}

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== 视频文件选择 ==========
        video_frame = ttk.LabelFrame(main_frame, text="视频设置", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))

        # 输入视频
        input_row = ttk.Frame(video_frame)
        input_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_row, text="输入视频:", width=10).pack(side=tk.LEFT)
        
        self.video_entry = ttk.Entry(input_row, textvariable=self._video_path)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(input_row, text="浏览...", command=self._browse_video).pack(side=tk.RIGHT)
        ttk.Button(input_row, text="摄像头", command=self._use_camera).pack(side=tk.RIGHT, padx=(0, 5))

        # 输出视频
        output_row = ttk.Frame(video_frame)
        output_row.pack(fill=tk.X)
        ttk.Label(output_row, text="输出视频:", width=10).pack(side=tk.LEFT)
        
        self.output_entry = ttk.Entry(output_row, textvariable=self._output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(output_row, text="浏览...", command=self._browse_output).pack(side=tk.RIGHT)

        ttk.Separator(main_frame).pack(fill=tk.X, pady=10)

        # ========== 遮蔽方式选择 ==========
        mode_frame = ttk.LabelFrame(main_frame, text="遮蔽方式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        modes = [
            ("黑块遮蔽", "black"),
            ("高斯模糊", "blur"),
            ("像素化", "pixelate"),
            ("失真效果", "distortion"),
        ]
        
        # 使用两行两列布局
        mode_grid = ttk.Frame(mode_frame)
        mode_grid.pack()
        for i, (text, value) in enumerate(modes):
            row, col = i // 2, i % 2
            ttk.Radiobutton(mode_grid, text=text, variable=self._censor_mode, 
                          value=value, command=self._on_mode_change).grid(row=row, column=col, 
                                                                          sticky='w', padx=20, pady=3)

        # 参数调节区
        self.params_frame = ttk.LabelFrame(mode_frame, text="参数调节", padding=5)
        self.params_frame.pack(fill=tk.X, pady=(10, 0))

        # 高斯模糊参数
        self.blur_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.blur_frame, text="模糊核大小 (奇数):").pack(side=tk.LEFT)
        ttk.Scale(
            self.blur_frame, 
            from_=5, 
            to=99, 
            variable=self._blur_kernel, 
            orient=tk.HORIZONTAL, 
            length=200,
            command=lambda v: self._blur_kernel.set(int(float(v)) | 1)  # 强制奇数
        ).pack(side=tk.LEFT, padx=5)
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

        # 缓冲帧数（始终显示，所有模式通用）
        self.buffer_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.buffer_frame, text="缓冲帧数:").pack(side=tk.LEFT)
        ttk.Scale(
            self.buffer_frame, 
            from_=0, 
            to=30, 
            variable=self._buffer_frames, 
            orient=tk.HORIZONTAL, 
            length=200,
            command=lambda v: self._buffer_frames.set(int(float(v)))  # 强制取整
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.buffer_frame, textvariable=self._buffer_frames).pack(side=tk.LEFT)

        # 初始显示缓冲帧数
        self.buffer_frame.pack(fill=tk.X, pady=5)

        self._on_mode_change()

        ttk.Separator(main_frame).pack(fill=tk.X, pady=10)

        # ========== 类别选择 ==========
        class_frame = ttk.LabelFrame(main_frame, text="选择要遮蔽的类别", padding=5)
        class_frame.pack(fill=tk.BOTH, expand=True)

        # 全选/取消全选按钮
        btn_frame = ttk.Frame(class_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="取消全选", command=self._deselect_all).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="仅选敏感", command=self._select_sensitive_only).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(btn_frame, text="仅选女性", command=self._select_female_only).pack(side=tk.LEFT, padx=(10, 0))

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
        self.status_label = ttk.Label(main_frame, text="请选择视频文件、遮蔽方式和要处理的类别后点击开始", foreground="gray")
        self.status_label.pack()

    # ========== 新增方法 ==========
    
    def _browse_video(self):
        """浏览选择视频文件"""
        filetypes = (
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("所有文件", "*.*")
        )
        filename = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=filetypes,
            initialdir="."
        )
        if filename:
            self._video_path.set(filename)
            # 自动生成输出文件名
            dir_name = os.path.dirname(filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self._output_path.set(os.path.join(dir_name, f"{base_name}_censored.mp4"))

    def _browse_output(self):
        """浏览选择输出路径"""
        filetypes = (
            ("MP4 视频", "*.mp4"),
            ("所有文件", "*.*")
        )
        filename = filedialog.asksaveasfilename(
            title="保存输出视频",
            filetypes=filetypes,
            defaultextension=".mp4",
            initialdir="."
        )
        if filename:
            self._output_path.set(filename)

    def _use_camera(self):
        """使用摄像头"""
        self._video_path.set("0")
        self._output_path.set("camera_output.mp4")

    def _select_sensitive_only(self):
        """仅选择敏感类别"""
        for label, var in self._label_vars.items():
            var.set(label in self.sensitive_labels)
    def _select_female_only(self):
        """仅选择敏感类别"""
        for label, var in self._label_vars.items():
            var.set(label in self.female_sensitive_labels)

    def _on_mode_change(self):
        """遮蔽方式改变时显示对应的参数，缓冲帧数始终显示"""
        mode = self._censor_mode.get()
        
        # 隐藏所有特定模式参数
        for frame in [self.blur_frame, self.pixel_frame, self.distortion_frame]:
            frame.pack_forget()
        self.buffer_frame.pack_forget()
        
        # 显示对应参数
        if mode == "blur":
            self.blur_frame.pack(fill=tk.X, pady=5)
        elif mode == "pixelate":
            self.pixel_frame.pack(fill=tk.X, pady=5)
        elif mode == "distortion":
            self.distortion_frame.pack(fill=tk.X, pady=5)
        
        # 缓冲帧数始终显示
        self.buffer_frame.pack(fill=tk.X, pady=5)

    def get_buffer_frames(self):
        """获取缓冲帧数"""
        return self._buffer_frames.get()
    
    def _select_all(self):
        for var in self._label_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._label_vars.values():
            var.set(False)

    def _on_start(self):
        # 检查视频路径
        video_path = self._video_path.get().strip()
        if not video_path:
            self.status_label.configure(text="请先选择输入视频文件！", foreground="red")
            return
        
        if video_path != "0" and not os.path.exists(video_path):
            self.status_label.configure(text=f"视频文件不存在: {video_path}", foreground="red")
            return

        self._started = True
        self.start_btn.configure(text="●  检测中...", state="disabled")
        selected_count = sum(1 for v in self._label_vars.values() if v.get())
        
        video_display = "摄像头" if video_path == "0" else os.path.basename(video_path)
        self.status_label.configure(
            text=f"已开始 | 视频: {video_display} | 遮蔽: {self.get_censor_mode_name()} | 选中 {selected_count} 个类别",
            foreground="green"
        )

    def is_started(self):
        return self._started

    def should_draw(self, class_name):
        if not self._running:
            return False
        if class_name in self._label_vars:
            return self._label_vars[class_name].get()
        return True

    def get_censor_mode(self):
        return self._censor_mode.get()

    def get_censor_mode_name(self):
        names = {
            "black": "黑块",
            "blur": "高斯模糊",
            "pixelate": "像素化",
            "distortion": "失真"
        }
        return names.get(self._censor_mode.get(), "未知")

    def get_censor_params(self):
        mode = self._censor_mode.get()
        if mode == "blur":
            return {"kernel_size": self._blur_kernel.get()}
        elif mode == "pixelate":
            return {"pixel_size": self._pixel_size.get()}
        elif mode == "distortion":
            return {"strength": self._distortion_strength.get()}
        return {}

    def get_video_path(self):
        """获取视频路径（字符串 "0" 表示摄像头）"""
        path = self._video_path.get().strip()
        if path == "0":
            return 0  # 摄像头
        return path

    def get_output_path(self):
        """获取输出路径"""
        return self._output_path.get().strip()

    def update(self):
        if self._running:
            self.root.update()
        return self._running

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def close(self):
        self._running = False
        self.root.destroy()

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
        self.female_sensitive_labels = [
            "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
            "ANUS_COVERED",
            "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
        ]
        # 新增：控制面板（延迟初始化，避免影响模型加载）
        self.control_panel = None

        # 遮蔽缓冲区：保存最近 N 帧的检测结果
        self.censor_buffer = []          # 列表，每个元素是 (剩余帧数, detections)
        self.buffer_frames = 5           # 缓冲帧数

    def get_label_name(self, class_id):
        if 0 <= class_id < len(self.nudenet_labels):
            return self.nudenet_labels[class_id]
        return f"UNKNOWN_{class_id}"
    
    def _merge_breast_boxes(self, breast_detections):
        """
        合并同一帧内的 breast 检测框
        如果只有 1 个，直接返回；如果有多个，合并为最小外接矩形
        """
        if len(breast_detections) == 0:
            return []
        
        if len(breast_detections) == 1:
            return breast_detections
        
        # 多个 breast 框：计算最小外接矩形
        min_x = min(det['bbox'][0] for det in breast_detections)
        min_y = min(det['bbox'][1] for det in breast_detections)
        max_x = max(det['bbox'][2] for det in breast_detections)
        max_y = max(det['bbox'][3] for det in breast_detections)
        
        # 取最高置信度作为合并后的置信度
        max_conf = max(det['confidence'] for det in breast_detections)
        
        # 判断是否为敏感类别（有任何一个 exposed 就算敏感）
        is_sensitive = any(det['is_sensitive'] for det in breast_detections)
        class_name = "FEMALE_BREAST_EXPOSED" if is_sensitive else "FEMALE_BREAST_COVERED"
        
        merged = {
            'bbox': (min_x, min_y, max_x, max_y),
            'class': breast_detections[0]['class'],  # 保持原类别 ID
            'class_name': class_name,
            'confidence': max_conf,
            'is_sensitive': is_sensitive
        }
        
        return [merged]

    def draw_results(self, frame, detections):
        """根据 GUI 选择对检测区域进行遮蔽，合并同 ROI 内的 breast 框"""
        mode = self.control_panel.get_censor_mode()
        params = self.control_panel.get_censor_params()
        buffer_frames = self.control_panel.get_buffer_frames()

        # 第一步：找出所有 breast 相关的框，进行合并
        breast_labels = ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED","MALE_BREAST_EXPOSED"]
        
        # 分离 breast 和其他检测
        breast_detections = []
        other_detections = []
        
        for det in detections:
            if det['class_name'] in breast_labels:
                breast_detections.append(det)
            else:
                other_detections.append(det)
        
        merged_breast = self._merge_breast_boxes(breast_detections)
        current_detections = merged_breast + other_detections

        # ========== 更新缓冲区 ==========
        if buffer_frames > 0:
            self.censor_buffer.append((buffer_frames, current_detections))

        # ========== 收集所有需要遮蔽的框 ==========
        all_boxes = []
        
        # 如果缓冲帧数 > 0，使用缓冲
        if buffer_frames > 0:
            for remaining, dets in self.censor_buffer:
                for det in dets:
                    all_boxes.append(det)
        else:
            # 缓冲帧数为 0，只使用当前帧
            all_boxes = current_detections

        # 第二步：对所有检测进行遮蔽
        for det in all_boxes:
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

        # ========== 更新缓冲区：减少剩余帧数，移除过期条目 ==========
        new_buffer = []
        for remaining, dets in self.censor_buffer:
            remaining -= 1
            if remaining > 0:
                new_buffer.append((remaining, dets))
        self.censor_buffer = new_buffer

        return frame

    def run(self):
        """启动检测（视频路径从 GUI 获取）"""
        import time

        # 初始化控制面板
        self.control_panel = DetectionControlPanel(
            nudenet_labels=self.nudenet_labels,
            sensitive_labels=self.sensitive_labels,
            female_sensitive_labels = self.female_sensitive_labels,
            default_video="./test/【兰幼金】夏日辣妹Bubble Pop❤超元气可爱小马达！ P1 横屏   - 0.00.13-0.00.23.mp4"  # 可选默认值
        )

        # 等待用户点击开始
        print("等待用户选择视频并点击开始...")
        while not self.control_panel.is_started():
            if not self.control_panel.update():
                print("用户关闭了控制面板，退出程序")
                return
            time.sleep(0.05)

        # 获取用户选择的路径
        video_source = self.control_panel.get_video_path()
        output_path = self.control_panel.get_output_path()

        print(f"输入视频: {video_source}")
        print(f"输出视频: {output_path}")
        print("开始检测！")

        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            self.control_panel.status_label.configure(
                text=f"无法打开视频源: {video_source}", foreground="red"
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        os.makedirs('./debug_roi', exist_ok=True)
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕" if video_source != 0 else "摄像头断开")
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
                        'is_sensitive': label_name in self.sensitive_labels,
                        'is_female': label_name in self.female_sensitive_labels
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

            # 绘制并保存
            annotated = self.draw_results(frame, final_detections)
            out.write(annotated)
            cv2.imshow('Cascade Detection', annotated)
            frame_id += 1

            # 更新控制面板
            if not self.control_panel.update():
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if self.control_panel is not None:
            self.control_panel.close()
        print("处理完成")


if __name__ == "__main__":
    detector = RealtimeCascadeDetector(
        fast_model='models/yolo26n.engine',
        precise_model='models/nudenet_640m.engine'
    )
    detector.run()  # 不需要传参了，全部从 GUI 获取