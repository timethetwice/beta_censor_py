from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import os
import numpy as np
import subprocess
import platform
import queue
import threading

from Screen_Censor import TransparentProtectionWindow


class VideoReader:
    def __init__(self, source, queue_size=4, use_gpu=False):
        self.source = source
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.running = False
        self.cap = None
        self.proc = None
        self.ret = False
        self.frame = None
        self.use_gpu = use_gpu and source != 0
        self.gpu_fallback_reason = None

    @staticmethod
    def _parse_fps(fps_text):
        if not fps_text or fps_text == "0/0":
            return 0.0
        if "/" in fps_text:
            num, den = fps_text.split("/", 1)
            den_value = float(den)
            if den_value == 0:
                return 0.0
            return float(num) / den_value
        return float(fps_text)

    def _probe_video_info(self):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate,nb_frames",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=0",
                    str(self.source),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False

        info = {}
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()

        self.codec_name = info.get("codec_name", "")
        self.width = int(info.get("width", "0") or 0)
        self.height = int(info.get("height", "0") or 0)
        self.fps = self._parse_fps(info.get("avg_frame_rate", "0/0"))
        duration = float(info.get("duration", "0") or 0)
        nb_frames = info.get("nb_frames", "0")
        if nb_frames.isdigit():
            self.frame_count = int(nb_frames)
        elif duration > 0 and self.fps > 0:
            self.frame_count = int(duration * self.fps)
        else:
            self.frame_count = 0
        return self.width > 0 and self.height > 0

    def _start_cpu_capture(self):
        if self.source == 0:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            return True

        self.use_gpu = False
        return self._probe_video_info()

    def _fallback_to_cpu(self, reason, existing_cap=None):
        self.gpu_fallback_reason = reason
        print(f"GPU decode unavailable for this file, falling back to ffmpeg CPU decode: {reason}")
        self.use_gpu = False
        if existing_cap is not None:
            self.cap = existing_cap
            return self.cap.isOpened()
        return self._start_cpu_capture()

    def _reader_loop(self):
        if self.use_gpu:
            self._gpu_reader_loop()
        else:
            self._cpu_reader_loop()

    def _cpu_reader_loop(self):
        if self.source != 0:
            self._ffmpeg_reader_loop()
            return

        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put((None, None))
                break
            self.queue.put((ret, frame.copy()))

    def _ffmpeg_reader_loop(self, decoder=None):
        cmd = ["ffmpeg"]
        if decoder:
            cmd.extend(["-c:v", decoder])
        cmd.extend([
            "-i",
            str(self.source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ])

        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            frame_size = self.width * self.height * 3
            if frame_size <= 0:
                raise ValueError(f"invalid frame size {self.width}x{self.height}")
            while self.running:
                raw = self.proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                self.queue.put((True, frame))
        except Exception as e:
            mode = "GPU" if decoder else "CPU"
            print(f"{mode} decode error: {e}")
        finally:
            if self.proc:
                self.proc.terminate()
                self.proc = None
            self.queue.put((False, None))

    def _get_cuvid_decoder(self, source):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(source)],
                capture_output=True, text=True, timeout=5
            )
            codec = result.stdout.strip().lower()
            if codec in ("hevc", "h265"):
                return "hevc_cuvid"
            return "h264_cuvid"
        except:
            return "h264_cuvid"

    def _gpu_reader_loop(self):
        cuvid = self._get_cuvid_decoder(self.source)
        self._ffmpeg_reader_loop(decoder=cuvid)

    def start(self):
        if self.source == 0:
            self.use_gpu = False
            if not self._start_cpu_capture():
                return False
        elif self.use_gpu:
            if not self._probe_video_info():
                return self._fallback_to_cpu("ffprobe failed to read video dimensions")
            if self.fps <= 0:
                self.fps = 30.0
        else:
            if not self._start_cpu_capture():
                return False

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def read(self):
        try:
            self.ret, self.frame = self.queue.get(timeout=1.0)
            return self.ret, self.frame
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.proc:
            self.proc.terminate()
            self.proc = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_fps(self):
        if self.use_gpu or self.cap is None:
            return getattr(self, 'fps', 30.0)
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0

    def get_frame_count(self):
        if self.use_gpu or self.cap is None:
            return getattr(self, 'frame_count', 0)
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def get_width(self):
        if self.use_gpu or self.cap is None:
            return getattr(self, 'width', 0)
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    def get_height(self):
        if self.use_gpu or self.cap is None:
            return getattr(self, 'height', 0)
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    def is_opened(self):
        return (
            (self.cap is not None and self.cap.isOpened())
            or self.proc is not None
            or (self.source != 0 and getattr(self, 'width', 0) > 0 and getattr(self, 'height', 0) > 0)
        )


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
        self._current_frame = 0
        self._total_frames = 0
        self._debug_enabled = tk.BooleanVar(value=False)

        self.root.geometry("500x600")
        self.root.minsize(450, 500)

        self._video_paths = []
        self._video_path = tk.StringVar(value=default_video)
        self._output_path = tk.StringVar(value="output_censored.mp4")
        self._target_size_mb = tk.DoubleVar(value=0)
        self._nude_model = tk.StringVar(value="640m")

        self._censor_mode = tk.StringVar(value="black")
        self._blur_kernel = tk.IntVar(value=81)
        self._pixel_size = tk.IntVar(value=10)
        self._distortion_strength = tk.IntVar(value=15)
        self._buffer_frames = tk.IntVar(value=5)
        self._genitalia_buffer_frames = tk.IntVar(value=10)

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
        self.fps_label = ttk.Label(
            bottom_frame, text="FPS: 0.0 | 帧: 0 / 0", foreground="gray"
        )
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
        ttk.Button(input_row, text="屏幕", command=self._use_screen, width=6).pack(
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

        # NudeNet 模型选择
        model_row = ttk.Frame(parent)
        model_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(model_row, text="检测模型:", width=10).pack(side=tk.LEFT)
        ttk.Radiobutton(model_row, text="NudeNet 320 (快速)", variable=self._nude_model, value="320").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(model_row, text="NudeNet 640 (精确)", variable=self._nude_model, value="640m").pack(side=tk.LEFT)

        # 预览开关
        self._show_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="显示预览窗口（关闭可加速）", variable=self._show_preview
        ).pack(anchor="w", pady=(10, 0))

        # 720p 加速开关
        self._downscale_720p = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            parent, text="启用 720p 加速（降低分辨率加快处理）", variable=self._downscale_720p
        ).pack(anchor="w", pady=(5, 0))

        # 调试开关
        ttk.Checkbutton(
            parent, text="保存调试图片", variable=self._debug_enabled
        ).pack(anchor="w", pady=(5, 0))

    def _build_censor_tab(self, parent):
        # 遮蔽方式选择
        modes = [
            ("黑块遮蔽", "black"),
            ("马赛克遮蔽", "mosaic"),
            ("高斯模糊", "blur"),
            ("像素化", "pixelate"),
            ("失真效果", "distortion"),
        ]
        mode_grid = ttk.Frame(parent)
        mode_grid.pack()
        for i, (text, value) in enumerate(modes):
            row, col = i // 3, i % 3
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
        ttk.Label(self.pixel_frame, text="像素块尺寸:").pack(side=tk.LEFT)
        ttk.Scale(
            self.pixel_frame,
            from_=2,
            to=30,
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

        # 私密部位缓冲帧数（始终显示）
        self.genitalia_buffer_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.genitalia_buffer_frame, text="私密部位缓冲:").pack(side=tk.LEFT)
        ttk.Scale(
            self.genitalia_buffer_frame,
            from_=0,
            to=30,
            variable=self._genitalia_buffer_frames,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._genitalia_buffer_frames.set(int(float(v))),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.genitalia_buffer_frame, textvariable=self._genitalia_buffer_frames).pack(
            side=tk.LEFT
        )

        self.buffer_frame.pack(fill=tk.X, pady=5)
        self.genitalia_buffer_frame.pack(fill=tk.X, pady=5)
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
        current_output = self._output_path.get().strip()
        if os.path.isdir(current_output):
            folder = filedialog.askdirectory(
                title="选择输出文件夹",
                initialdir=current_output,
            )
            if folder:
                self._output_path.set(folder)
        else:
            filetypes = (("MP4 视频", "*.mp4"), ("所有文件", "*.*"))
            filename = filedialog.asksaveasfilename(
                title="保存输出视频",
                filetypes=filetypes,
                defaultextension=".mp4",
                initialdir=current_output if os.path.isdir(current_output) else ".",
            )
            if filename:
                self._output_path.set(filename)

    def _use_camera(self):
        """使用摄像头"""
        self._video_path.set("0")
        self._video_paths = [0]
        self._output_path.set("camera_output.mp4")

    def _use_screen(self):
        """使用屏幕捕获"""
        self._video_path.set("screen")
        self._video_paths = ["screen"]
        self._output_path.set("screen_output.mp4")

    # ========== 模式切换 ==========

    def _on_mode_change(self):
        """遮蔽方式改变时显示对应参数"""
        for frame in [
            self.blur_frame,
            self.pixel_frame,
            self.distortion_frame,
            self.buffer_frame,
            self.genitalia_buffer_frame,
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
        self.genitalia_buffer_frame.pack(fill=tk.X, pady=5)

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
        self.fps_label.configure(
            text=f"FPS: {fps:.1f} | 帧: {self._current_frame} / {self._total_frames}"
        )

    def update_frame_count(self, current, total):
        self._current_frame = current
        self._total_frames = total
        self.fps_label.configure(
            text=f"FPS: {self._current_fps:.1f} | 帧: {current} / {total}"
        )

    def is_started(self):
        return self._started

    def should_draw(self, class_name):
        if not self._running:
            return False
        if class_name in self._label_vars:
            enabled = self._label_vars[class_name].get()
            if class_name in ("FACE_FEMALE", "FACE_MALE") and enabled:
                eye_enabled = self._label_vars.get("eye", None)
                if eye_enabled and eye_enabled.get():
                    return False
            return enabled
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

    def get_genitalia_buffer_frames(self):
        return self._genitalia_buffer_frames.get()

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

    def get_debug_enabled(self):
        return self._debug_enabled.get()

    def get_downscale_720p(self):
        return self._downscale_720p.get()

    def get_nude_model(self):
        return self._nude_model.get()

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
    def mosaic_block(frame, x1, y1, x2, y2, block_size=8):
        """灰黑马赛克遮蔽（高速）"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        h, w = roi.shape[:2]

        # 向量化创建棋盘格掩码
        yy = np.arange(h, dtype=np.int32)
        xx = np.arange(w, dtype=np.int32)
        y_idx, x_idx = np.meshgrid(yy, xx, indexing='ij')
        mask = ((y_idx // block_size) + (x_idx // block_size)) % 2 == 0

        # 赋值两种灰度
        roi[mask] = 20   # 深灰
        roi[~mask] = 60  # 浅灰
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
        """像素化遮蔽（pixel_size 为像素块尺寸）"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        h, w = roi.shape[:2]

        # pixel_size 表示像素块边长（像素），统一所有 ROI
        size = max(2, pixel_size)

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
            self.precise_models = {
                "320": YOLO("models/nudenet_320n.mlpackage", task="detect"),
                "640m": YOLO("models/nudenet_640m.mlpackage", task="detect"),
            }
            self.eye_model = YOLO("models/face-parts-yolov8n.mlpackage", task="detect")
        else:  # Windows
            self.device = "cuda"
            self.fast_model = YOLO(fast_model or "models/yolo26n.engine", task="detect")
            self.precise_models = {
                "320": YOLO("models/nudenet_320n.engine", task="detect"),
                "640m": YOLO("models/nudenet_640m.engine", task="detect"),
            }
            self.eye_model = YOLO("models/face-parts-yolov8n.engine", task="detect")

        self.nude_imgsz = {"320": (320, 192), "640m": (640, 384)}

        # 私密部位 padding（横向、纵向放大系数）
        self.GENITALIA_PAD_X = 4.0
        self.GENITALIA_PAD_Y = 1.6

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
            "eye",
        ]
        self.sensitive_labels = [
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "eye",
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
            "eye",
        ]
        # 新增：控制面板（延迟初始化，避免影响模型加载）
        self.control_panel = None

        # 遮蔽缓冲区：保存最近 N 帧的检测结果（延迟初始化为列表）
        self.censor_buffer = None
        self.genitalia_censor_buffer = None
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

    def get_nude_model_and_imgsz(self):
        model_key = self.control_panel.get_nude_model() if self.control_panel else "640m"
        return self.precise_models[model_key], self.nude_imgsz[model_key]

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

    def _merge_eye_boxes(self, eye_detections):
        """
        合并同一张脸上的 eye 检测框
        eye_detections 是同一 face_idx 的眼睛列表
        """
        eye_padding_x_ratio = 0.3
        eye_padding_y_ratio = 0.3
        if len(eye_detections) == 0:
            return []
        if len(eye_detections) == 1:
            det = eye_detections[0]
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            # 使用比例：x方向扩展宽度的0.2倍，y方向扩展高度的0.1倍
            padding_x = int(width * eye_padding_x_ratio)
            padding_y = int(height * eye_padding_y_ratio)
            merged = {
                "bbox": (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y),
                "class": 999,
                "class_name": "eye",
                "confidence": det["confidence"],
                "is_sensitive": True,
                "is_female": False,
            }
            return [merged]

        min_x = min(det["bbox"][0] for det in eye_detections)
        min_y = min(det["bbox"][1] for det in eye_detections)
        max_x = max(det["bbox"][2] for det in eye_detections)
        max_y = max(det["bbox"][3] for det in eye_detections)
        max_conf = max(det["confidence"] for det in eye_detections)
        
        # 计算合并框的宽高
        width = max_x - min_x
        height = max_y - min_y
        # 使用比例计算padding
        padding_x = int(width * eye_padding_x_ratio)
        padding_y = int(height * eye_padding_y_ratio)

        merged = {
            "bbox": (min_x - padding_x, min_y - padding_y, max_x + padding_x, max_y + padding_y),
            "class": 999,
            "class_name": "eye",
            "confidence": max_conf,
            "is_sensitive": True,
            "is_female": False,
        }
        return [merged]

    def draw_results(self, frame, detections):
        """根据 GUI 选择对检测区域进行遮蔽，合并同 ROI 内的 breast 框"""
        mode = self.control_panel.get_censor_mode()
        params = self.control_panel.get_censor_params()
        buffer_frames = self.control_panel.get_buffer_frames()
        genitalia_buffer_frames = self.control_panel.get_genitalia_buffer_frames()

        breast_labels = [
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_BREAST_COVERED",
            "MALE_BREAST_EXPOSED",
        ]
        genitalia_labels = [
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_GENITALIA_COVERED",
        ]

        breast_detections = []
        genitalia_detections = []
        eye_detections = []
        other_detections = []

        for det in detections:
            if det["class_name"] in breast_labels:
                breast_detections.append(det)
            elif det["class_name"] in genitalia_labels:
                genitalia_detections.append(det)
            elif det["class_name"] == "eye":
                eye_detections.append(det)
            else:
                other_detections.append(det)

        person_breast_map = {}
        for det in breast_detections:
            pid = det.pop("person_idx", None)
            if pid not in person_breast_map:
                person_breast_map[pid] = []
            person_breast_map[pid].append(det)

        person_eye_map = {}
        for det in eye_detections:
            pid = det.pop("person_idx", None)
            if pid not in person_eye_map:
                person_eye_map[pid] = []
            person_eye_map[pid].append(det)

        merged_breast = []
        for pid, breasts in person_breast_map.items():
            merged_breast.extend(self._merge_breast_boxes(breasts))

        merged_genitalia = list(genitalia_detections)

        merged_eyes = []
        for pid, eyes in person_eye_map.items():
            merged_eyes.extend(self._merge_eye_boxes(eyes))

        current_detections = merged_breast + merged_eyes + other_detections

        # ========== 缓冲区递减 & 清理 ==========
        if self.censor_buffer is None:
            self.censor_buffer = []
        if self.genitalia_censor_buffer is None:
            self.genitalia_censor_buffer = []

        new_buffer = []
        for remaining, dets in self.censor_buffer:
            remaining -= 1
            if remaining > 0:
                new_buffer.append((remaining, dets))
        self.censor_buffer = new_buffer

        new_genitalia_buffer = []
        for remaining, dets in self.genitalia_censor_buffer:
            remaining -= 1
            if remaining > 0:
                new_genitalia_buffer.append((remaining, dets))
        self.genitalia_censor_buffer = new_genitalia_buffer

        # ========== 更新缓冲区（追加当前帧） ==========
        if buffer_frames > 0:
            self.censor_buffer.append((buffer_frames, current_detections))
        if genitalia_buffer_frames > 0:
            self.genitalia_censor_buffer.append((genitalia_buffer_frames, merged_genitalia))

        # ========== 收集所有需要遮蔽的框 ==========
        all_boxes = list(current_detections)
        for remaining, dets in self.censor_buffer:
            all_boxes.extend(dets)
        for remaining, dets in self.genitalia_censor_buffer:
            all_boxes.extend(dets)

        # 第二步：对所有检测进行遮蔽
        for det in all_boxes:
            if self.control_panel is not None and not self.control_panel.should_draw(
                det["class_name"]
            ):
                continue

            x1, y1, x2, y2 = det["bbox"]

            if mode == "black":
                CensorEffects.black_block(frame, x1, y1, x2, y2)
            elif mode == "mosaic":
                CensorEffects.mosaic_block(frame, x1, y1, x2, y2)
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

    def _annotate_frame(self, frame):
        """对单帧进行检测和遮蔽（用于屏幕捕获模式）"""
        height, width = frame.shape[:2]

        results_fast = self.fast_model(frame, conf=0.3, classes=[0])
        candidates = []
        for box in results_fast[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad = 50
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            candidates.append((x1, y1, x2, y2))

        final_detections = []
        face_rois = {}

        person_rois = []
        person_indices = []
        person_roi_data = []
        for idx, (x1, y1, x2, y2) in enumerate(candidates):
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            person_rois.append(roi)
            person_indices.append(idx)
            person_roi_data.append((idx, x1, y1))

        precise_model, nude_imgsz = self.get_nude_model_and_imgsz()

        BATCH_SIZE = 2
        if person_rois:
            for batch_start in range(0, len(person_rois), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(person_rois))
                batch_rois = person_rois[batch_start:batch_end]
                batch_indices = person_indices[batch_start:batch_end]
                batch_roi_data = person_roi_data[batch_start:batch_end]

                precise_results = precise_model(batch_rois, conf=0.2, imgsz=nude_imgsz, rect=False)

                for i, res in enumerate(precise_results):
                    person_idx = batch_indices[i]
                    idx, x1, y1 = batch_roi_data[i]

                    for pr in res.boxes:
                        rx1, ry1, rx2, ry2 = map(int, pr.xyxy[0].tolist())
                        conf = float(pr.conf[0])
                        cls = int(pr.cls[0])
                        label_name = self.get_label_name(cls)

                        bx1, by1, bx2, by2 = x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2

                        if label_name in ("FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED"):
                            w, h = bx2 - bx1, by2 - by1
                            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                            new_w, new_h = int(w * self.GENITALIA_PAD_X), int(h * self.GENITALIA_PAD_Y)
                            bx1, by1 = cx - new_w // 2, cy - new_h // 2
                            bx2, by2 = cx + new_w // 2, cy + new_h // 2

                        final_detections.append(
                            {
                                "bbox": (bx1, by1, bx2, by2),
                                "class": cls,
                                "class_name": label_name,
                                "confidence": conf,
                                "is_sensitive": label_name in self.sensitive_labels,
                                "is_female": label_name in self.female_sensitive_labels,
                                "person_idx": person_idx,
                            }
                        )

                        if label_name in ("FACE_FEMALE", "FACE_MALE"):
                            face_rois[person_idx] = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)

        if face_rois:
            face_roi_list = []
            face_idx_list = []
            face_roi_coords = {}
            for person_idx, (fx1, fy1, fx2, fy2) in face_rois.items():
                face_roi = frame[fy1:fy2, fx1:fx2]
                if face_roi.size == 0:
                    continue
                face_roi_list.append(face_roi)
                face_idx_list.append(person_idx)
                face_roi_coords[person_idx] = (fx1, fy1)

            if face_roi_list:
                for batch_start in range(0, len(face_roi_list), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(face_roi_list))
                    batch_rois = face_roi_list[batch_start:batch_end]
                    batch_indices = face_idx_list[batch_start:batch_end]

                    eye_results = self.eye_model(batch_rois, conf=0.2, imgsz=320, rect=False)

                    for i, res in enumerate(eye_results):
                        person_idx = batch_indices[i]
                        fx1, fy1 = face_roi_coords[person_idx]

                        for er in res.boxes:
                            cls_id = int(er.cls[0])
                            label_name = self.eye_model.names[cls_id]

                            ex1, ey1, ex2, ey2 = map(int, er.xyxy[0].tolist())
                            final_detections.append(
                                {
                                    "bbox": (fx1 + ex1, fy1 + ey1, fx1 + ex2, fy1 + ey2),
                                    "class": cls_id,
                                    "class_name": label_name,
                                    "confidence": float(er.conf[0]),
                                    "is_sensitive": label_name in self.sensitive_labels,
                                    "is_female": label_name in self.female_sensitive_labels,
                                    "person_idx": person_idx,
                                }
                            )

        return self.draw_results(frame, final_detections)

    def _merge_audio(self, video_source, temp_video, output_path):
        """用 ffmpeg 将原视频的音频合并到输出视频（视频流直接 copy）"""
        if not os.path.exists(temp_video):
            print(f"音频合并跳过：未找到临时视频 {temp_video}")
            return False

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
                return True

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
            return True

        except subprocess.CalledProcessError as e:
            print(f"音频合并失败: {e.stderr.decode() if e.stderr else str(e)}")
            if os.path.exists(temp_video):
                os.rename(temp_video, output_path)
                print(f"无声视频已保存为: {output_path}")
                return True
            print("音频合并失败，且无声临时视频不存在")
            return False

    def _process_video(self, video_source, output_path, preview):
        """处理单个视频，返回是否正常完成"""
        import time

        reader = VideoReader(video_source, use_gpu=True)
        if not reader.start():
            print(f"无法打开视频源: {video_source}")
            return

        fps = reader.get_fps()
        total_frames = reader.get_frame_count()
        orig_width = reader.get_width()
        orig_height = reader.get_height()

        # 720p 加速模式
        downscale_720p = self.control_panel.get_downscale_720p()
        if downscale_720p:
            target_height = 720
            target_width = int(orig_width * (target_height / orig_height))
            target_width = (target_width // 2) * 2
            target_height = (target_height // 2) * 2
        else:
            target_width = orig_width
            target_height = orig_height

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
            f"{target_width}x{target_height}",
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
        self.censor_buffer = None  # 延迟初始化为列表

        # FPS 计时
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        while True:
            if self.control_panel.is_stopped():
                print("用户停止处理")
                break
            ret, frame = reader.read()
            if not ret:
                break

            if (
                frame is None
                or frame.size == 0
                or len(frame.shape) < 2
                or frame.shape[0] == 0
                or frame.shape[1] == 0
            ):
                print("跳过空帧，避免触发模型预处理错误")
                continue

            # 720p 加速模式：缩小到 720p
            if downscale_720p:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # ----------------- 第一阶段：检测人 -----------------
            results_fast = self.fast_model(frame, conf=0.3, classes=[0])
            candidates = []
            for box in results_fast[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pad = 50
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(target_width, x2 + pad)
                y2 = min(target_height, y2 + pad)
                candidates.append((x1, y1, x2, y2))

            # ----------------- 第二阶段：NudeNet 检测（批量分片）-----------------
            precise_model, nude_imgsz = self.get_nude_model_and_imgsz()
            final_detections = []
            face_rois = {}

            person_rois = []
            person_indices = []
            person_roi_data = []
            for idx, (x1, y1, x2, y2) in enumerate(candidates):
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                person_rois.append(roi)
                person_indices.append(idx)
                person_roi_data.append((idx, x1, y1))

            BATCH_SIZE = 2
            if person_rois:
                for batch_start in range(0, len(person_rois), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(person_rois))
                    batch_rois = person_rois[batch_start:batch_end]
                    batch_indices = person_indices[batch_start:batch_end]
                    batch_roi_data = person_roi_data[batch_start:batch_end]

                    precise_results = precise_model(batch_rois, conf=0.2, imgsz=nude_imgsz, rect=False)

                    for i, res in enumerate(precise_results):
                        person_idx = batch_indices[i]
                        idx, x1, y1 = batch_roi_data[i]

                        for pr in res.boxes:
                            rx1, ry1, rx2, ry2 = map(int, pr.xyxy[0].tolist())
                            conf = float(pr.conf[0])
                            cls = int(pr.cls[0])
                            label_name = self.get_label_name(cls)

                            bx1, by1, bx2, by2 = x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2

                            if label_name in ("FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED"):
                                w, h = bx2 - bx1, by2 - by1
                                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                                new_w, new_h = int(w * self.GENITALIA_PAD_X), int(h * self.GENITALIA_PAD_Y)
                                bx1, by1 = cx - new_w // 2, cy - new_h // 2
                                bx2, by2 = cx + new_w // 2, cy + new_h // 2

                            final_detections.append(
                                {
                                    "bbox": (bx1, by1, bx2, by2),
                                    "class": cls,
                                    "class_name": label_name,
                                    "confidence": conf,
                                    "is_sensitive": label_name in self.sensitive_labels,
                                    "is_female": label_name in self.female_sensitive_labels,
                                    "person_idx": person_idx,
                                }
                            )

                            if label_name in ("FACE_FEMALE", "FACE_MALE"):
                                face_rois[person_idx] = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)

                        if self.control_panel.get_debug_enabled() and frame_id % 30 == 0:
                            roi_annotated = batch_rois[i].copy()
                            for det in final_detections:
                                if det["person_idx"] == person_idx:
                                    gx1, gy1, gx2, gy2 = det["bbox"]
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
                                f"./debug_roi/debug_roi_{frame_id}_{person_idx}.jpg", roi_annotated
                            )

            # ----------------- 第三阶段：Eye 模型检测（批量分片）-----------------
            if face_rois:
                face_roi_list = []
                face_idx_list = []
                face_roi_coords = {}
                for person_idx, (fx1, fy1, fx2, fy2) in face_rois.items():
                    face_roi = frame[fy1:fy2, fx1:fx2]
                    if face_roi.size == 0:
                        continue
                    face_roi_list.append(face_roi)
                    face_idx_list.append(person_idx)
                    face_roi_coords[person_idx] = (fx1, fy1)

                if face_roi_list:
                    for batch_start in range(0, len(face_roi_list), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(face_roi_list))
                        batch_rois = face_roi_list[batch_start:batch_end]
                        batch_indices = face_idx_list[batch_start:batch_end]

                        eye_results = self.eye_model(batch_rois, conf=0.2, imgsz=320, rect=False)

                        for i, res in enumerate(eye_results):
                            person_idx = batch_indices[i]
                            fx1, fy1 = face_roi_coords[person_idx]

                            for er in res.boxes:
                                cls_id = int(er.cls[0])
                                label_name = self.eye_model.names[cls_id]
                                if label_name != "eye":
                                    continue
                                er_x1, er_y1, er_x2, er_y2 = map(int, er.xyxy[0].tolist())
                                final_detections.append(
                                    {
                                        "bbox": (
                                            fx1 + er_x1,
                                            fy1 + er_y1,
                                            fx1 + er_x2,
                                            fy1 + er_y2,
                                        ),
                                        "class": 999,
                                        "class_name": "eye",
                                        "confidence": float(er.conf[0]),
                                        "is_sensitive": True,
                                        "is_female": True,
                                        "person_idx": person_idx,
                                    }
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
                self.control_panel.update_frame_count(frame_id, total_frames)

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
        reader.stop()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if preview:
            cv2.destroyAllWindows()

        if not os.path.exists(temp_output):
            self.control_panel.status_label.configure(
                text="处理失败：输出视频未生成", foreground="red"
            )
            print(f"输出失败：未生成临时视频 {temp_output}")
            return False

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
            merged = self._merge_audio(video_source, temp_output, output_path)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if not merged:
                self.control_panel.status_label.configure(
                    text="处理失败：音频合并或输出生成失败", foreground="red"
                )
                return False
        else:
            os.rename(temp_output, output_path)

        # 不关闭控制面板，只更新状态
        self.control_panel.status_label.configure(
            text=f"处理完成！输出: {os.path.basename(output_path)}", foreground="blue"
        )
        return True

    def _process_screen(self, output_path, preview):
        """处理屏幕捕获，实时遮蔽显示器输出"""
        from Screen_Censor import TransparentProtectionWindow
        import time

        screen_window = TransparentProtectionWindow()
        if not screen_window.create_window():
            print("无法创建保护窗口")
            return False

        screen_window.show_window()

        width = screen_window.screen_width
        height = screen_window.screen_height
        fps = 30.0

        self.control_panel.update_frame_count(0, 0)
        frame_id = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        try:
            while not self.control_panel.is_stopped():
                frame = screen_window.capture_screen()
                if frame is None:
                    break

                frame_id += 1
                fps_frame_count += 1

                if frame_id % 10 == 0:
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0
                    self.control_panel.update_fps(current_fps)

                annotated = self._annotate_frame(frame)

                screen_window.update_display_content(annotated)

                if not self.control_panel.update():
                    break

        finally:
            screen_window.close()

        self.control_panel.status_label.configure(
            text="屏幕遮蔽已停止", foreground="blue"
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
            elif video_sources == ["screen"]:
                # 屏幕捕获模式
                self._process_screen(output_base, preview)
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
                output_dir = output_base if os.path.isdir(output_base) else "."
                for i, video_source in enumerate(video_sources):
                    base_name = os.path.splitext(os.path.basename(video_source))[0]
                    output_path = os.path.join(output_dir, f"{base_name}_censored.mp4")

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
