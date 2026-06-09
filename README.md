# My Beta Censor

AI-powered video censorship detection tool with cascade detection pipeline, supporting video file processing and real-time screen overlay.

## Features

- **Cascade detection**: Fast person detection (YOLO) → precise NudeNet classification per ROI → optional eye detection
- **Model selection**: NudeNet 320 (fast) or NudeNet 640m (accurate)
- **Multiple censor modes**: black block, mosaic, gaussian blur, pixelate, distortion
- **Buffer system**: separate configurable frame buffers for regular detections and genitalia detections to reduce flicker
- **Screen capture mode**: real-time transparent overlay window with click-through and screenshot exclusion (Windows 10 2004+)
- **Batch processing**: process all videos in a folder
- **720p downscale**: optional resolution reduction for faster processing
- **Target file size**: specify output video size via bitrate control
- **Camera support**: live webcam processing
- **Debug image saving**: save annotated ROI images for inspection

## Requirements

- Python >= 3.12
- NVIDIA GPU with CUDA (for `.engine` models on Windows) or Apple Silicon (for `.mlpackage` on macOS)

## Installation

```bash
uv sync  # or: pip install -e .
```

## Models

Place model files in `models/`:

| Model | Windows | macOS |
|-------|---------|-------|
| Fast person detection | `yolo26n.engine` | `yolo26n.mlpackage` |
| NudeNet 320 | `nudenet_320n.engine` | `nudenet_320n.mlpackage` |
| NudeNet 640m | `nudenet_640m.engine` | `nudenet_640m.mlpackage` |
| Face parts (eyes) | `face-parts-yolov8n.engine` | `face-parts-yolov8n.mlpackage` |

## Usage

```bash
python main.py
```

### GUI controls

1. **Video Settings tab**: select input video (file/folder/camera/screen), output path, model version, 720p toggle, preview toggle, debug toggle
2. **Censor Mode tab**: choose censor effect and adjust parameters
3. **Class Selection tab**: choose which detection classes to censor

Click **Start** to begin processing. The GUI stays open after completion for repeated use.

## Screen Capture Mode (Windows only)

Click the **Screen** button to create a transparent overlay window that captures and censors the screen in real time. The overlay window:
- Is transparent with click-through
- Is excluded from screenshots (Windows 10 2004+)
- Updates via `UpdateLayeredWindow` for smooth display

## Project Structure

```
main.py          — Main application (Windows, detection pipeline, GUI)
main_mac.py      — macOS-compatible variant
Screen_Censor.py — Transparent overlay window for screen capture
models/          — YOLO / NudeNet model files
conver_trt.py    — TensorRT engine conversion utility
debug_roi/       — Debug output (annotated ROIs)
```

## Notes

- Windows: uses TensorRT `.engine` models with CUDA
- macOS: uses CoreML `.mlpackage` models with MPS
- ffmpeg is required for audio merging in output videos
- Batch size limited to 2 ROIs per inference pass to avoid OOM
