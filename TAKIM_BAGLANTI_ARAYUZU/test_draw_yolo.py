import cv2
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
IMAGE_PATH = "data/frames_rgb/frame_001110.jpg"
OUTPUT_PATH = "outputs/tests/yolo_draw_test.jpg"


def main():
    model = YOLO(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(f"Görsel okunamadı: {IMAGE_PATH}")

    results = model(IMAGE_PATH, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if conf < 0.25:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"class={cls_id} conf={conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    Path("outputs/tests").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"Kaydedildi: {OUTPUT_PATH}")
    print(f"Tespit sayısı: {len(results.boxes)}")


if __name__ == "__main__":
    main()