from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("models/best.pt")

SESSION_DIR = Path(
    "_images/THYZ_2026_Online_Yarisma_Test_Oturumu"
)

OUTPUT_DIR = Path(
    "outputs/tests/task3_yolo_debug"
)

FRAME_INDICES = [
    175,
    183,
    188,
    190,
    194,
    199,
]

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(str(MODEL_PATH))

    for frame_index in FRAME_INDICES:
        frame_path = (
            SESSION_DIR
            / f"frame_{frame_index:06d}.webp"
        )

        image = cv2.imread(str(frame_path))

        if image is None:
            print("Okunamadı:", frame_path)
            continue

        results = model.predict(
            source=image,
            imgsz=1280,
            conf=0.01,
            iou=0.50,
            max_det=100,
            verbose=False,
        )

        output = image.copy()

        print()
        print("=" * 72)
        print(frame_path.name)
        print("=" * 72)

        detection_count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                detection_count += 1

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}",
                )

                print(
                    f"class={class_id} {class_name:<6} "
                    f"conf={confidence:.4f} "
                    f"bbox={(x1, y1, x2, y2)}"
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    4,
                )

                label = (
                    f"{class_name} "
                    f"{confidence:.3f}"
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(35, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        print("Toplam detection:", detection_count)

        output_path = (
            OUTPUT_DIR
            / f"debug_{frame_index:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            output,
        )

        print("Kaydedildi:", output_path)


if __name__ == "__main__":
    main()