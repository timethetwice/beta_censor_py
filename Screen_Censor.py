"""
Windows 屏幕内容安全保护系统
功能：创建一个全屏透明窗口，鼠标点击穿透，截图工具自动排除本窗口
要求：Windows 10 2004 或更高版本
"""
import platform

if platform.system() == "Windows":

    import ctypes
    import ctypes.wintypes
    import sys
    import win32gui
    import win32api
    import win32con
    import win32ui
    import cv2
    from typing import Tuple
    import numpy as np
    from PIL import Image

    # 防止 Windows 缩放导致截屏不全
    ctypes.windll.user32.SetProcessDPIAware()

    # ============== Windows API 常量定义 ==============

    # 窗口扩展样式
    WS_EX_LAYERED = 0x80000
    WS_EX_TRANSPARENT = 0x20      # 鼠标点击穿透
    WS_EX_TOPMOST = 0x8
    WS_EX_TOOLWINDOW = 0x80       # 不在任务栏显示

    # 分层窗口属性
    LWA_COLORKEY = 0x1            # 透明色键模式
    LWA_ALPHA = 0x2               # Alpha 混合模式

    # 标准窗口样式（无边框）
    WS_POPUP = 0x80000000
    WS_VISIBLE = 0x10000000

    # SetWindowDisplayAffinity 参数
    WDA_NONE = 0x00000000
    WDA_MONITOR = 0x00000001      # 窗口在截图中显示为黑色
    WDA_EXCLUDEFROMCAPTURE = 0x00000011  # 截图时排除本窗口（Win10 2004+）

    # 窗口消息
    WM_PAINT = 0x000F
    WM_ERASEBKGND = 0x0014
    WM_DESTROY = 0x0002

    # ============== 加载 user32.dll ==============

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    SetWindowDisplayAffinity.restype = ctypes.c_bool

    # 加载 gdi32.dll
    gdi32 = ctypes.windll.gdi32


    class TransparentProtectionWindow:
        """全屏透明保护窗口类"""
        
        def __init__(self):
            self.hwnd = None
            self.screen_width = 0
            self.screen_height = 0
            self.transparent_color = win32api.RGB(0, 255, 0)  # 纯绿色作为透明色
            self.running = False
            
            # 注册窗口类
            self._register_window_class()
        
        def _register_window_class(self):
            """注册窗口类（仅在需要时调用）"""
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._window_proc
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.lpszClassName = "TransparentProtectionWindow"
            wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
            wc.hbrBackground = win32con.COLOR_WINDOW + 1
            
            try:
                win32gui.RegisterClass(wc)
            except Exception as e:
                # 类可能已经注册过，忽略错误
                pass
        
        def _window_proc(self, hwnd, msg, wparam, lparam):
            """窗口消息处理函数"""
            if msg == WM_PAINT:
                # 处理绘制消息：绘制透明背景
                hdc, paint_struct = win32gui.BeginPaint(hwnd)
                # 用透明色填充整个窗口
                brush = win32gui.CreateSolidBrush(self.transparent_color)
                rect = win32gui.GetClientRect(hwnd)
                win32gui.FillRect(hdc, rect, brush)
                win32gui.DeleteObject(brush)
                win32gui.EndPaint(hwnd, paint_struct)
                return 0
                
            elif msg == WM_ERASEBKGND:
                # 返回 1 表示我们已经处理了背景擦除
                return 1
                
            elif msg == WM_DESTROY:
                self.running = False
                win32gui.PostQuitMessage(0)
                return 0
                
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
        
        def get_screen_size(self) -> Tuple[int, int]:
            """获取屏幕尺寸"""
            self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            return self.screen_width, self.screen_height
        
        def create_window(self) -> bool:
            """创建全屏透明窗口"""
            # 获取屏幕尺寸
            self.get_screen_size()
            
            # 创建窗口
            self.hwnd = win32gui.CreateWindowEx(
                WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                "TransparentProtectionWindow",
                "Screen Protection Layer",
                WS_POPUP | WS_VISIBLE,
                0, 0,
                self.screen_width, self.screen_height,
                None, None,
                win32api.GetModuleHandle(None),
                None
            )
            
            if not self.hwnd:
                print("错误：无法创建窗口")
                return False
            
            # 设置分层窗口的透明色（已改用 UpdateLayeredWindow，无需此调用）
            # win32gui.SetLayeredWindowAttributes(
            #     self.hwnd,
            #     self.transparent_color,  # 透明色键
            #     0,                       # Alpha 值（LWA_COLORKEY 时忽略）
            #     LWA_COLORKEY             # 使用颜色键模式
            # )

            # 设置截图排除
            self.set_capture_exclusion()
            
            print(f"窗口创建成功！")
            print(f"  屏幕尺寸: {self.screen_width}x{self.screen_height}")
            print(f"  透明色: RGB(0, 255, 0)")
            print(f"  点击穿透: 已启用")
            print(f"  截图排除: 已启用 (Windows 10 2004+)")
            
            return True
        
        def set_capture_exclusion(self) -> bool:
            """设置窗口在屏幕截图中被排除（穿透捕获）"""
            if not self.hwnd:
                return False
            
            # 检查 Windows 版本（简单检查，要求 Win10 2004+）
            # 实际上 WDA_EXCLUDEFROMCAPTURE 会失败于旧版本
            try:
                result = SetWindowDisplayAffinity(self.hwnd, WDA_EXCLUDEFROMCAPTURE)
                if result:
                    return True
                else:
                    error_code = ctypes.GetLastError()
                    print(f"警告：SetWindowDisplayAffinity 失败，错误码 {error_code}")
                    print("  可能原因：Windows 版本低于 10 2004 或权限不足")
                    return False
            except Exception as e:
                print(f"警告：SetWindowDisplayAffinity 调用异常: {e}")
                return False
        
        def show_window(self):
            """显示窗口"""
            if self.hwnd:
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
                win32gui.UpdateWindow(self.hwnd)
                # 强制重绘以应用透明色
                win32gui.InvalidateRect(self.hwnd, None, True)
        
        def hide_window(self):
            """隐藏窗口"""
            if self.hwnd:
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)

        def update_display_content(self, image_data=None):
            """
            更新窗口显示内容（供外部调用）
            """
            if not self.hwnd or image_data is None:
                return

            try:
                import numpy as np
                import ctypes
                from ctypes import wintypes

    # 确保显示全屏
                img_h, img_w = image_data.shape[:2]
                if img_w != self.screen_width or img_h != self.screen_height:
                    image_data = cv2.resize(image_data, (self.screen_width, self.screen_height))

                # 转换: PIL 返回 RGB, 转为 BGRA (需要交换 R 和 B)
                r, g, b = cv2.split(image_data)
                alpha = np.full((self.screen_height, self.screen_width), 255, dtype=np.uint8)
                rgba = cv2.merge([b, g, r, alpha])

                hdc_screen = win32gui.GetDC(0)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)

                # BITMAPINFO
                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ("biSize", wintypes.DWORD),
                        ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD),
                    ]

                class BITMAPINFO(ctypes.Structure):
                    _fields_ = [("bmiHeader", BITMAPINFOHEADER)]

                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = self.screen_width
                bmi.bmiHeader.biHeight = -self.screen_height  # 负数表示top-down
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 32
                bmi.bmiHeader.biCompression = 0  # BI_RGB

                # 创建 HBITMAP
                hBitmap = gdi32.CreateDIBSection(
                    hdc_mem, ctypes.byref(bmi), 0, None, None, 0
                )

                if hBitmap:
                    old_bmp = gdi32.SelectObject(hdc_mem, hBitmap)

                    # 复制像素数据
                    gdi32.SetDIBits(
                        hdc_mem, hBitmap, 0, self.screen_height,
                        rgba.tobytes(), ctypes.byref(bmi), 0
                    )

                    # BLENDFUNCTION as tuple: (BlendOp, BlendFlags, SourceConstantAlpha, AlphaFormat)
                    blend = (win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)

                    win32gui.UpdateLayeredWindow(
                        self.hwnd, hdc_screen, (0, 0),
                        (self.screen_width, self.screen_height),
                        hdc_mem, (0, 0), 0, blend, win32con.ULW_ALPHA
                    )

                    # 保持窗口顶层
                    win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                        win32con.SWP_NOACTIVATE | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

                    gdi32.SelectObject(hdc_mem, old_bmp)
                    gdi32.DeleteObject(hBitmap)

                gdi32.DeleteDC(hdc_mem)
                win32gui.ReleaseDC(0, hdc_screen)

            except Exception as e:
                import traceback
                print(f"Update display error: {e}")
                traceback.print_exc()

        def capture_screen(self):
            """捕获整个屏幕，返回 numpy array (BGR)"""
            import numpy as np
            from PIL import Image

            width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

            hwnd_desktop = win32gui.GetDesktopWindow()
            desktop_dc = win32gui.GetWindowDC(hwnd_desktop)
            img_dc = win32ui.CreateDCFromHandle(desktop_dc)

            mem_dc = img_dc.CreateCompatibleDC()
            screenshot = win32ui.CreateBitmap()
            screenshot.CreateCompatibleBitmap(img_dc, width, height)
            old_bmp = mem_dc.SelectObject(screenshot)
            mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)

            bmpinfo = screenshot.GetInfo()
            bmpstr = screenshot.GetBitmapBits(True)
            img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

            mem_dc.SelectObject(old_bmp)
            mem_dc.DeleteDC()
            img_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd_desktop, desktop_dc)

            return np.array(img)

        def run_message_loop(self):
            """运行消息循环"""
            self.running = True
            while self.running:
                # 使用 PeekMessage 非阻塞处理消息
                # 这样可以允许主线程做其他工作
                win32gui.PumpWaitingMessages()
                
                # 也可以使用 GetMessage 阻塞方式：
                # msg = win32gui.GetMessage(None, 0, 0)
                # if msg == 0: break
                # win32gui.TranslateMessage(msg)
                # win32gui.DispatchMessage(msg)
                
                # 简单的延时，避免 CPU 占用过高
                win32api.Sleep(10)
        
        def close(self):
            """关闭窗口"""
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None
            self.running = False


    # ============== 辅助函数 ==============

    def check_windows_version() -> bool:
        """检查 Windows 版本是否支持 WDA_EXCLUDEFROMCAPTURE"""
        try:
            version = sys.getwindowsversion()
            major, minor, build = version.major, version.minor, version.build
            
            # Windows 10 2004 对应 build 19041
            # Windows 11 也支持
            if major == 10 and build >= 19041:
                print(f"Windows 版本: {major}.{minor} Build {build} - 支持截图排除")
                return True
            elif major > 10:
                print(f"Windows 版本: {major}.{minor} Build {build} - 支持截图排除")
                return True
            else:
                print(f"警告：Windows 版本 {major}.{minor} Build {build} 可能不支持 WDA_EXCLUDEFROMCAPTURE")
                print("  需要 Windows 10 2004 (build 19041) 或更高版本")
                return False
        except Exception:
            return False


    def capture_screen_region(x: int = 0, y: int = 0, width: int = None, height: int = None):
        """
        捕获屏幕指定区域（示例函数）
        这里使用 win32ui 实现简单的屏幕捕获
        """
        if width is None:
            width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        if height is None:
            height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        # 获取桌面 DC
        hwnd_desktop = win32gui.GetDesktopWindow()
        desktop_dc = win32gui.GetWindowDC(hwnd_desktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        
        # 创建兼容 DC 和位图
        mem_dc = img_dc.CreateCompatibleDC()
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        
        # 复制屏幕内容
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (x, y), win32con.SRCCOPY)
        
        # 转换为 numpy 数组（如果需要）
        # 这里返回 PIL Image 对象供后续处理
        from PIL import Image
        import numpy as np
        
        bmpinfo = screenshot.GetInfo()
        bmpstr = screenshot.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
        
        # 清理资源
        mem_dc.DeleteDC()
        img_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd_desktop, desktop_dc)
        
        return np.array(img)


    # ============== 主程序 ==============

    def main():
        """主程序入口"""
        # 检查 Windows 版本
        check_windows_version()
        
        print("=" * 50)
        print("屏幕内容安全保护系统")
        print("=" * 50)
        
        # 创建保护窗口
        window = TransparentProtectionWindow()
        
        if not window.create_window():
            print("创建窗口失败")
            return
        
        window.show_window()
        
        print("\n保护窗口已启动")
        print("- 窗口完全透明")
        print("- 鼠标点击会穿透到底层窗口")
        print("- 屏幕截图时本窗口不会出现")
        print("- 按 Ctrl+C 退出程序")
        print("=" * 50)
        
        try:
            # 运行消息循环
            window.run_message_loop()
        except KeyboardInterrupt:
            print("\n正在退出...")
        finally:
            window.close()
            print("程序已退出")

else:
    # macOS/Linux 上定义虚拟类
    class TransparentProtectionWindow:
        def __init__(self, *args, **kwargs):
            print("透明保护窗口在 macOS 上不可用")
            self.hwnd = None
            self.running = False
        
        def create_window(self, use_preview=True):
            print("透明保护窗口仅在 Windows 上可用")
            return False
        
        def show_window(self):
            pass
        
        def update_display_content(self, frame):
            pass
        
        def run_message_loop(self):
            pass
        
        def close(self):
            pass

    # 如果你的遮盖程序需要集成，可以这样使用：
    """
    from your_redaction_module import redact_image

    def run_with_redaction():
        window = TransparentProtectionWindow()
        window.create_window()
        window.show_window()
        
        while window.running:
            # 捕获屏幕（会自动排除保护窗口）
            captured = capture_screen_region()
            
            # 调用你的遮盖程序
            redacted = redact_image(captured)
            
            # 更新窗口显示（需要实现 update_display_content）
            window.update_display_content(redacted)
            
            # 处理消息
            win32gui.PumpWaitingMessages()
    """

if __name__ == "__main__":
    main()