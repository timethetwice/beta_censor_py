from ultralytics import YOLO
import cv2

from main import VideoReader


MODEL_PATH = "./models/yolo26n-pose.pt"
SOURCE = r"E:\哔哩哔哩\一条小糖糖\[2026-06-18 22-00-34][一条小糖糖][大方 展示 美丽]_PART000.mp4"
WINDOW_NAME = "YOLO Annotate Preview"


def main():
    model = YOLO(MODEL_PATH)
    reader = VideoReader(SOURCE, use_gpu=True)

    if not reader.start():
        raise RuntimeError(f"Unable to open video source: {SOURCE}")

    print(
        f"Video opened: {reader.get_width()}x{reader.get_height()} @ {reader.get_fps():.2f} FPS"
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
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
                continue

            results = model(frame, verbose=False)
            annotated = results[0].plot()
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
