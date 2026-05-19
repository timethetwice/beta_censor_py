from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import os
import numpy as np
import subprocess
import platform


import tkinter as tk
from tkinter import ttk, filedialog
import os
import cv2


class DetectionControlPanel:
    def __init__(
        self,
        nudenet_labels,
        sensitive_labels,
        female_sensitive_labels,
        default_video="",
    ):
        self.nudenet_labels = nudenet_labels
        self.sensitive_labels = sensitive_labels
        self.female_sensitive_labels = female_sensitive_labels

        self.root = tk.Tk()
        self.root.title("检测控制面板")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._running = True
        self._started = False
        self._stopped = False
        self._current_fps = 0.0

        self.root.geometry("500x600")
        self.root.minsize(450, 500)

        self._video_paths = []
        self._video_path = tk.StringVar(value=default_video)
        self._output_path = tk.StringVar(value="output_censored.mp4")
        self._target_size_mb = tk.DoubleVar(value=0)

        self._censor_mode = tk.StringVar(value="black")
        self._blur_kernel = tk.IntVar(value=81)
        self._pixel_size = tk.IntVar(value=13)
        self._distortion_strength = tk.IntVar(value=15)
        self._buffer_frames = tk.IntVar(value=5)

        self._label_vars = {}

        self._build_ui()

    def _build_ui(self):
        outer_frame = ttk.Frame(self.root, padding=10)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(outer_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_video = ttk.Frame(notebook, padding=10)
        notebook.add(tab_video, text="视频设置")
        self._build_video_tab(tab_video)

        tab_censor = ttk.Frame(notebook, padding=10)
        notebook.add(tab_censor, text="遮蔽方式")
        self._build_censor_tab(tab_censor)

        tab_class = ttk.Frame(notebook, padding=10)
        notebook.add(tab_class, text="类别选择")
        self._build_class_tab(tab_class)

        bottom_frame = ttk.Frame(outer_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(
            btn_frame, text="▶  开始检测", command=self._on_start
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(
            btn_frame, text="■  停止", command=self._on_stop, state="disabled"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.fps_label = ttk.Label(bottom_frame, text="FPS: 0.0", foreground="gray")
        self.fps_label.pack()
        self.status_label = ttk.Label(
            bottom_frame,
            text="请选择视频文件、遮蔽方式和要处理的类别后点击开始",
            foreground="gray",
        )
        self.status_label.pack()

    def _build_video_tab(self, parent):
        # 输入视频
        input_row = ttk.Frame(parent)
        input_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_row, text="输入视频:", width=10).pack(side=tk.LEFT)
        self.video_entry = ttk.Entry(input_row, textvariable=self._video_path)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(input_row, text="文件...", command=self._browse_video).pack(
            side=tk.RIGHT
        )
        ttk.Button(input_row, text="文件夹...", command=self._browse_folder).pack(
            side=tk.RIGHT, padx=(0, 5)
        )
        ttk.Button(input_row, text="摄像头", command=self._use_camera).pack(
            side=tk.RIGHT, padx=(0, 5)
        )

        # 输出视频
        output_row = ttk.Frame(parent)
        output_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(output_row, text="输出视频:", width=10).pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(output_row, textvariable=self._output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_row, text="浏览...", command=self._browse_output).pack(
            side=tk.RIGHT
        )

        # 目标文件大小
        size_row = ttk.Frame(parent)
        size_row.pack(fill=tk.X)
        ttk.Label(size_row, text="目标文件大小:", width=10).pack(side=tk.LEFT)
        self.size_entry = ttk.Entry(
            size_row, textvariable=self._target_size_mb, width=10
        )
        self.size_entry.pack(side=tk.LEFT, padx=(0, 3))
        ttk.Label(size_row, text="MB (0=不限制)").pack(side=tk.LEFT)

        # 预览开关
        self._show_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="显示预览窗口（关闭可加速）", variable=self._show_preview
        ).pack(anchor="w", pady=(10, 0))

    def _build_censor_tab(self, parent):
        # 遮蔽方式选择
        modes = [
            ("黑块遮蔽", "black"),
            ("高斯模糊", "blur"),
            ("像素化", "pixelate"),
            ("失真效果", "distortion"),
        ]
        mode_grid = ttk.Frame(parent)
        mode_grid.pack()
        for i, (text, value) in enumerate(modes):
            row, col = i // 2, i % 2
            ttk.Radiobutton(
                mode_grid,
                text=text,
                variable=self._censor_mode,
                value=value,
                command=self._on_mode_change,
            ).grid(row=row, column=col, sticky="w", padx=20, pady=3)

        # 参数调节
        self.params_frame = ttk.LabelFrame(parent, text="参数调节", padding=5)
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
            command=lambda v: self._blur_kernel.set(int(float(v)) | 1),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.blur_frame, textvariable=self._blur_kernel).pack(side=tk.LEFT)

        # 像素化参数
        self.pixel_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.pixel_frame, text="像素块数量:").pack(side=tk.LEFT)
        ttk.Scale(
            self.pixel_frame,
            from_=5,
            to=50,
            variable=self._pixel_size,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._pixel_size.set(int(float(v))),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.pixel_frame, textvariable=self._pixel_size).pack(side=tk.LEFT)

        # 失真参数
        self.distortion_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.distortion_frame, text="扭曲强度:").pack(side=tk.LEFT)
        ttk.Scale(
            self.distortion_frame,
            from_=5,
            to=50,
            variable=self._distortion_strength,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._distortion_strength.set(int(float(v))),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.distortion_frame, textvariable=self._distortion_strength).pack(
            side=tk.LEFT
        )

        # 缓冲帧数（始终显示）
        self.buffer_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.buffer_frame, text="缓冲帧数:").pack(side=tk.LEFT)
        ttk.Scale(
            self.buffer_frame,
            from_=0,
            to=30,
            variable=self._buffer_frames,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._buffer_frames.set(int(float(v))),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.buffer_frame, textvariable=self._buffer_frames).pack(
            side=tk.LEFT
        )

        self.buffer_frame.pack(fill=tk.X, pady=5)
        self._on_mode_change()

    def _build_class_tab(self, parent):
        # 按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="取消全选", command=self._deselect_all).pack(
            side=tk.LEFT
        )
        ttk.Button(
            btn_frame, text="仅选敏感", command=self._select_sensitive_only
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(btn_frame, text="仅选女性", command=self._select_female_only).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # 滚动类别列表
        canvas = tk.Canvas(parent, height=250)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._label_vars = {}
        for label in self.nudenet_labels:
            var = tk.BooleanVar(value=True)
            self._label_vars[label] = var
            is_sensitive = label in self.sensitive_labels
            display_text = f"{'⚠️' if is_sensitive else '✓'} {label}"
            cb = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var)
            cb.pack(anchor="w", pady=1)

    # ========== 文件浏览方法 ==========

    def _browse_video(self):
        """选择单个视频文件"""
        filetypes = (
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("所有文件", "*.*"),
        )
        filename = filedialog.askopenfilename(
            title="选择视频文件", filetypes=filetypes, initialdir="."
        )
        if filename:
            self._video_path.set(filename)
            self._video_paths = [filename]
            dir_name = os.path.dirname(filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self._output_path.set(os.path.join(dir_name, f"{base_name}_censored.mp4"))

    def _browse_folder(self):
        """选择文件夹（批量处理）"""
        folder = filedialog.askdirectory(title="选择包含视频的文件夹", initialdir=".")
        if folder:
            video_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
            self._video_paths = []
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(video_exts):
                    self._video_paths.append(os.path.join(folder, f))
            count = len(self._video_paths)
            if count:
                self._video_path.set(f"📁 {folder} ({count} 个视频)")
            else:
                self._video_path.set(f"📁 {folder} (无视频文件)")
            self._output_path.set(folder)

    def _browse_output(self):
        """选择输出路径"""
        filetypes = (("MP4 视频", "*.mp4"), ("所有文件", "*.*"))
        filename = filedialog.asksaveasfilename(
            title="保存输出视频",
            filetypes=filetypes,
            defaultextension=".mp4",
            initialdir=".",
        )
        if filename:
            self._output_path.set(filename)

    def _use_camera(self):
        """使用摄像头"""
        self._video_path.set("0")
        self._video_paths = [0]
        self._output_path.set("camera_output.mp4")

    # ========== 模式切换 ==========

    def _on_mode_change(self):
        """遮蔽方式改变时显示对应参数"""
        for frame in [
            self.blur_frame,
            self.pixel_frame,
            self.distortion_frame,
            self.buffer_frame,
        ]:
            frame.pack_forget()

        mode = self._censor_mode.get()
        if mode == "blur":
            self.blur_frame.pack(fill=tk.X, pady=5)
        elif mode == "pixelate":
            self.pixel_frame.pack(fill=tk.X, pady=5)
        elif mode == "distortion":
            self.distortion_frame.pack(fill=tk.X, pady=5)

        # 缓冲帧数始终显示
        self.buffer_frame.pack(fill=tk.X, pady=5)

    # ========== 选择方法 ==========

    def _select_all(self):
        for var in self._label_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._label_vars.values():
            var.set(False)

    def _select_sensitive_only(self):
        for label, var in self._label_vars.items():
            var.set(label in self.sensitive_labels)

    def _select_female_only(self):
        for label, var in self._label_vars.items():
            var.set(label in self.female_sensitive_labels)

    # ========== 开始/状态 ==========

    def _on_start(self):
        video_path = self._video_path.get().strip()
        if not video_path or video_path.startswith("📁") and not self._video_paths:
            self.status_label.configure(
                text="请先选择有效的视频文件！", foreground="red"
            )
            return
        self._started = True
        self._stopped = False
        self.start_btn.configure(text="●  检测中...", state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="已开始检测...", foreground="green")

    def _on_stop(self):
        self._stopped = True
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="正在停止...", foreground="orange")

    def is_stopped(self):
        return self._stopped

    def reset_stopped(self):
        self._stopped = False

    def update_fps(self, fps):
        self._current_fps = fps
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

    def is_started(self):
        return self._started

    def should_draw(self, class_name):
        if not self._running:
            return False
        if class_name in self._label_vars:
            return self._label_vars[class_name].get()
        return True

    # ========== Getter 方法 ==========

    def get_censor_mode(self):
        return self._censor_mode.get()

    def get_censor_params(self):
        mode = self._censor_mode.get()
        if mode == "blur":
            return {"kernel_size": self._blur_kernel.get()}
        elif mode == "pixelate":
            return {"pixel_size": self._pixel_size.get()}
        elif mode == "distortion":
            return {"strength": self._distortion_strength.get()}
        return {}

    def get_buffer_frames(self):
        return self._buffer_frames.get()

    def get_video_paths(self):
        if self._video_paths:
            return self._video_paths
        path = self._video_path.get().strip()
        if path == "0":
            return [0]
        if path and not path.startswith("📁"):
            return [path]
        return []

    def get_output_path(self):
        return self._output_path.get().strip()

    def get_preview(self):
        return self._show_preview.get()

    def get_target_size_mb(self):
        return self._target_size_mb.get()

    # ========== 生命周期 ==========

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
    def gaussian_blur(frame, x1, y1, x2, y2, kernel_size=81):
        """高斯模糊遮蔽（缩小加速版）"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        if kernel_size % 2 == 0:
            kernel_size += 1

        h, w = roi.shape[:2]

        # 缩小因子（根据区域大小动态调整）
        if max(w, h) > 100:
            scale = 0.1  # 缩小到 1/10
        else:
            scale = 0.5  # 小区域不缩太小

        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # 核也等比缩小
        small_kernel = max(3, int(kernel_size * scale))
        if small_kernel % 2 == 0:
            small_kernel += 1

        blurred_small = cv2.GaussianBlur(small_roi, (small_kernel, small_kernel), 0)
        blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

        frame[y1:y2, x1:x2] = blurred
        return frame

    @staticmethod
    def pixelate(frame, x1, y1, x2, y2, pixel_size=10):
        """像素化遮蔽（GUI 参数控制块数）"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        h, w = roi.shape[:2]

        # pixel_size 表示目标块数（沿短边），范围建议 5-50
        blocks = max(3, pixel_size)

        # 计算实际像素块大小
        size = max(1, min(h, w) // blocks)

        # 缩小再放大
        small_w = max(1, w // size)
        small_h = max(1, h // size)

        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
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
    def __init__(self, fast_model=None, precise_model=None):
        if platform.system() == "Darwin":  # macOS
            self.device = "mps"
            self.fast_model = YOLO(
                fast_model or "models/yolo26n.mlpackage", task="detect"
            )
            self.precise_model = YOLO(
                precise_model or "models/nudenet_640m.mlpackage", task="detect"
            )
        else:  # Windows
            self.device = "cuda"
            self.fast_model = YOLO(fast_model or "models/yolo26n.engine", task="detect")
            self.precise_model = YOLO(
                precise_model or "models/nudenet_640m.engine", task="detect"
            )
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
        self.sensitive_labels = [
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
        ]
        self.female_sensitive_labels = [
            "FEMALE_GENITALIA_COVERED",
            "FACE_FEMALE",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
            "ANUS_COVERED",
            "FEMALE_BREAST_COVERED",
            "BUTTOCKS_COVERED",
        ]
        # 新增：控制面板（延迟初始化，避免影响模型加载）
        self.control_panel = None

        # 遮蔽缓冲区：保存最近 N 帧的检测结果
        self.censor_buffer = []  # 列表，每个元素是 (剩余帧数, detections)
        self.buffer_frames = 5  # 缓冲帧数
        # 检查 ffmpeg 是否可用
        self.has_ffmpeg = self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg():
        """检查 ffmpeg 是否可用"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("警告: 未找到 ffmpeg，输出视频将没有音频")
            return False

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
        min_x = min(det["bbox"][0] for det in breast_detections)
        min_y = min(det["bbox"][1] for det in breast_detections)
        max_x = max(det["bbox"][2] for det in breast_detections)
        max_y = max(det["bbox"][3] for det in breast_detections)

        # 取最高置信度作为合并后的置信度
        max_conf = max(det["confidence"] for det in breast_detections)

        # 判断是否为敏感类别（有任何一个 exposed 就算敏感）
        is_sensitive = any(det["is_sensitive"] for det in breast_detections)
        class_name = (
            "FEMALE_BREAST_EXPOSED" if is_sensitive else "FEMALE_BREAST_COVERED"
        )

        merged = {
            "bbox": (min_x, min_y, max_x, max_y),
            "class": breast_detections[0]["class"],  # 保持原类别 ID
            "class_name": class_name,
            "confidence": max_conf,
            "is_sensitive": is_sensitive,
        }

        return [merged]

    def draw_results(self, frame, detections):
        """根据 GUI 选择对检测区域进行遮蔽，合并同 ROI 内的 breast 框"""
        mode = self.control_panel.get_censor_mode()
        params = self.control_panel.get_censor_params()
        buffer_frames = self.control_panel.get_buffer_frames()

        # 第一步：找出所有 breast 相关的框，进行合并
        breast_labels = [
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_BREAST_COVERED",
            "MALE_BREAST_EXPOSED",
        ]

        # 分离 breast 和其他检测
        breast_detections = []
        other_detections = []

        for det in detections:
            if det["class_name"] in breast_labels:
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
            if self.control_panel is not None and not self.control_panel.should_draw(
                det["class_name"]
            ):
                continue

            x1, y1, x2, y2 = det["bbox"]

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

    def _merge_audio(self, video_source, temp_video, output_path):
        """用 ffmpeg 将原视频的音频合并到输出视频（视频流直接 copy）"""
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(video_source),
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)

            if "audio" not in result.stdout:
                print("原视频没有音频轨道，跳过音频合并")
                os.rename(temp_video, output_path)
                return

            merge_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video,
                "-i",
                str(video_source),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                output_path,
            ]

            print("正在合并音频...")
            subprocess.run(merge_cmd, capture_output=True, check=True)
            print(f"音频合并完成: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"音频合并失败: {e.stderr.decode() if e.stderr else str(e)}")
            os.rename(temp_video, output_path)
            print(f"无声视频已保存为: {output_path}")

    def _process_video(self, video_source, output_path, preview):
        """处理单个视频，返回是否正常完成"""
        import time

        if video_source == 0:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 检查输入视频时长（用于码率计算）
        duration = 0
        if video_source != 0:
            try:
                dur_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(video_source),
                ]
                dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
                duration = float(dur_result.stdout.strip())
            except:
                duration = total_frames / fps if fps > 0 else 0

        target_size = self.control_panel.get_target_size_mb()

        # 构建 ffmpeg 命令
        temp_output = output_path.replace(".mp4", "_noaudio.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]

        if target_size > 0 and duration > 0:
            # 有目标大小：计算码率
            target_bitrate = (target_size * 8 * 1024) / duration  # kbps
            video_bitrate = int(target_bitrate * 0.9)  # 预留音频空间
            ffmpeg_cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-b:v",
                    f"{video_bitrate}k",
                    "-maxrate",
                    f"{int(target_bitrate)}k",
                    "-bufsize",
                    f"{int(target_bitrate * 2)}k",
                ]
            )
            print(f"目标码率: {video_bitrate}kbps")
        else:
            # 无目标大小：CRF 模式
            ffmpeg_cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])

        ffmpeg_cmd.append(temp_output)

        # 启动 ffmpeg 进程
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.makedirs("./debug_roi", exist_ok=True)
        frame_id = 0
        self.censor_buffer = []

        # FPS 计时
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        while True:
            if self.control_panel.is_stopped():
                print("用户停止处理")
                break
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

                precise_results = self.precise_model(roi, conf=0.3, imgsz=(640, 384))

                for pr in precise_results[0].boxes:
                    rx1, ry1, rx2, ry2 = map(int, pr.xyxy[0].tolist())
                    conf = float(pr.conf[0])
                    cls = int(pr.cls[0])
                    label_name = self.get_label_name(cls)

                    final_detections.append(
                        {
                            "bbox": (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2),
                            "class": cls,
                            "class_name": label_name,
                            "confidence": conf,
                            "is_sensitive": label_name in self.sensitive_labels,
                            "is_female": label_name in self.female_sensitive_labels,
                        }
                    )

                # 调试保存（带检测框的 ROI）
                if frame_id % 30 == 0:
                    roi_annotated = roi.copy()
                    for det in final_detections:
                        gx1, gy1, gx2, gy2 = det["bbox"]
                        if gx1 >= x1 and gy1 >= y1 and gx2 <= x2 and gy2 <= y2:
                            rx1 = gx1 - x1
                            ry1 = gy1 - y1
                            rx2 = gx2 - x1
                            ry2 = gy2 - y1
                            color = (
                                (0, 0, 255) if det["is_sensitive"] else (0, 255, 255)
                            )
                            cv2.rectangle(
                                roi_annotated, (rx1, ry1), (rx2, ry2), color, 2
                            )
                            label = f"{det['class_name']}: {det['confidence']:.2f}"
                            cv2.putText(
                                roi_annotated,
                                label,
                                (rx1, ry1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                            )
                    cv2.imwrite(
                        f"./debug_roi/debug_roi_{frame_id}_{idx}.jpg", roi_annotated
                    )

            # 绘制并写入 ffmpeg 管道
            annotated = self.draw_results(frame, final_detections)
            ffmpeg_proc.stdin.write(annotated.tobytes())

            # FPS 计算（每 10 帧更新一次）
            fps_frame_count += 1
            if fps_frame_count >= 10:
                elapsed = time.time() - fps_start_time
                current_fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0
                self.control_panel.update_fps(current_fps)

            # 预览（可选）
            if preview:
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    annotated,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Cascade Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 控制台进度（每 100 帧打印一次）
            frame_id += 1
            if frame_id % 100 == 0 and total_frames > 0:
                progress = frame_id / total_frames * 100
                print(
                    f"进度: {progress:.1f}% | 帧: {frame_id}/{total_frames} | FPS: {current_fps:.1f}"
                )

            # 更新控制面板
            if not self.control_panel.update():
                break

        # 最终统计
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if preview:
            cv2.destroyAllWindows()

        stopped = self.control_panel.is_stopped()

        if stopped:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            output_censored = output_path
            if os.path.exists(output_censored):
                os.remove(output_censored)
            self.control_panel.status_label.configure(
                text="处理已停止，文件已删除", foreground="orange"
            )
            return False

        # 合并音频（视频已是 H.264，直接 copy，不重编码）
        if self.has_ffmpeg and video_source != 0:
            self._merge_audio(video_source, temp_output, output_path)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        else:
            os.rename(temp_output, output_path)

        # 不关闭控制面板，只更新状态
        self.control_panel.status_label.configure(
            text=f"处理完成！输出: {os.path.basename(output_path)}", foreground="blue"
        )
        return True

    def run(self):
        import time

        self.control_panel = DetectionControlPanel(
            nudenet_labels=self.nudenet_labels,
            sensitive_labels=self.sensitive_labels,
            female_sensitive_labels=self.female_sensitive_labels,
            default_video="...",
        )

        print("等待用户选择视频并点击开始...")
        while not self.control_panel.is_started():
            if not self.control_panel.update():
                print("用户关闭了控制面板，退出程序")
                return
            time.sleep(0.05)

        while True:
            video_sources = self.control_panel.get_video_paths()
            output_base = self.control_panel.get_output_path()
            preview = self.control_panel.get_preview()

            # 判断处理模式
            if video_sources == [0]:
                # 摄像头模式
                self._process_video(0, "camera_output.mp4", preview)
                success = True
            elif len(video_sources) == 1:
                # 单文件
                video_path = video_sources[0]
                if video_path.endswith("_censored.mp4"):
                    output_path = video_path
                else:
                    dir_name = os.path.dirname(video_path)
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    output_path = os.path.join(dir_name, f"{base_name}_censored.mp4")
                self._process_video(video_path, output_path, preview)
                success = True
            else:
                # 多文件或文件夹
                for i, video_source in enumerate(video_sources):
                    dir_name = (
                        os.path.dirname(video_source)
                        if os.path.dirname(video_source)
                        else "."
                    )
                    base_name = os.path.splitext(os.path.basename(video_source))[0]
                    output_path = os.path.join(dir_name, f"{base_name}_censored.mp4")

                    if i > 0 and preview:
                        cv2.destroyAllWindows()

                    self.control_panel.status_label.configure(
                        text=f"处理中 [{i + 1}/{len(video_sources)}]: {os.path.basename(video_source)}",
                        foreground="orange",
                    )
                    self.control_panel.update()
                    self._process_video(
                        video_source,
                        output_path,
                        preview and i == len(video_sources) - 1,
                    )
                success = True

            # 恢复 GUI 状态
            self.control_panel._started = False
            self.control_panel.reset_stopped()
            self.control_panel.start_btn.configure(text="▶  开始检测", state="normal")
            self.control_panel.stop_btn.configure(state="disabled")

            if success:
                self.control_panel.status_label.configure(
                    text=f"处理完成！共处理 {len(video_sources) if video_sources and video_sources[0] != 0 else 0} 个视频 | 可重新选择视频并开始",
                    foreground="blue",
                )
            else:
                self.control_panel.status_label.configure(
                    text=f"无法打开视频源", foreground="red"
                )

            print("GUI 窗口保持打开，可重新选择视频开始检测，或关闭窗口退出")

            # 等待用户再次点击开始或关闭窗口
            while not self.control_panel.is_started():
                if not self.control_panel.update():
                    print("程序退出")
                    return
                time.sleep(0.05)

        self.control_panel.close()
        print("程序退出")


if __name__ == "__main__":
    detector = RealtimeCascadeDetector()  # 自动按平台选模型
    #     detector = RealtimeCascadeDetector(
    #     fast_model='models/yolo26n.mlpackage',
    #     precise_model='models/nudenet_640m.mlpackage'
    # )
    detector.run()
