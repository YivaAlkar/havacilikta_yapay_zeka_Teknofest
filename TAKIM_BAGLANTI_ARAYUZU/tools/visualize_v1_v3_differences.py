from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

FRAME_DIR = ROOT / "data" / "frames_rgb"

MODEL_PATHS = {
    "V1": ROOT / "models" / "best.pt",
    "V3": ROOT / "models" / "best_v3.pt",
}

OUT_DIR = ROOT / "outputs" / "v1_v3_visual_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_NAMES = [
    "frame_000540.jpg",
    "frame_000570.jpg",
    "frame_001350.jpg",
    "frame_002010.jpg",
    "frame_008370.jpg",
    "frame_008460.jpg",
    "frame_008550.jpg",
    "frame_000600.jpg",
    "frame_000630.jpg",
    "frame_000660.jpg",
    "frame_001050.jpg",
    "frame_001440.jpg",
]

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

CONFIDENCE = 0.40
IOU = 0.50


def draw_predictions(image, result, model_name):
    output = image.copy()
    object_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = [
            int(value)
            for value in box.xyxy[0].tolist()
        ]

        label = (
            f"{CLASS_NAMES[class_id]} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        object_count += 1

    header = f"{model_name} | objects={object_count}"

    cv2.rectangle(
        output,
        (0, 0),
        (460, 45),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        header,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def resize_same_height(left, right):
    target_height = min(left.shape[0], right.shape[0])

    def resize(image):
        ratio = target_height / image.shape[0]
        width = int(image.shape[1] * ratio)

        return cv2.resize(
            image,
            (width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    return resize(left), resize(right)


def main():
    models = {
        name: YOLO(str(path))
        for name, path in MODEL_PATHS.items()
    }

    saved = 0

    for frame_name in FRAME_NAMES:
        frame_path = FRAME_DIR / frame_name

        if not frame_path.exists():
            print(f"[WARN] Bulunamadi: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"[WARN] Okunamadi: {frame_path}")
            continue

        rendered = {}

        for model_name, model in models.items():
            result = model.predict(
                source=str(frame_path),
                conf=CONFIDENCE,
                iou=IOU,
                device=0,
                verbose=False,
            )[0]

            rendered[model_name] = draw_predictions(
                image,
                result,
                model_name,
            )

        left, right = resize_same_height(
            rendered["V1"],
            rendered["V3"],
        )

        comparison = cv2.hconcat([left, right])

        output_path = (
            OUT_DIR
            / f"{frame_path.stem}_v1_vs_v3.jpg"
        )

        cv2.imwrite(str(output_path), comparison)

        saved += 1
        print(f"[OK] {output_path.name}")

    print("\n" + "=" * 80)
    print(f"[DONE] saved={saved}")
    print(f"[OUT] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()