import cv2
from pathlib import Path

from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

MOVING_NAMES = {
    "1": "Hareketli",
    "0": "Sabit",
    "-1": "Tasit Degil",
}

LANDING_NAMES = {
    "1": "Inilebilir",
    "0": "Inilemez",
    "-1": "Inis Alani Degil",
}


def draw_prediction(frame_path, output_path, model):
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"Görsel okunamadı: {frame_path}")
        return

    frame_name = Path(frame_path).stem

    prediction = FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/{frame_name}.jpg",
        video_name="visual_task1_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    for obj in result.detected_objects:
        cls_id = int(obj.cls[0])
        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        moving_name = MOVING_NAMES.get(str(obj.moving_status), str(obj.moving_status))
        landing_name = LANDING_NAMES.get(str(obj.landing_status), str(obj.landing_status))

        x1 = int(float(obj.top_left_x))
        y1 = int(float(obj.top_left_y))
        x2 = int(float(obj.bottom_right_x))
        y2 = int(float(obj.bottom_right_y))

        # Renk belirtmiyoruz diye düşünme, burada OpenCV çizimi için teknik BGR tuple gerekiyor.
        # Görsel test amaçlı.
        color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} | {moving_name} | {landing_name}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.imwrite(str(output_path), image)
    print(f"{frame_name}: objects={len(result.detected_objects)} -> {output_path}")


def main():
    output_dir = Path("outputs/tests/task1_visual")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ObjectDetectionModel("http://test/")

    frames = [
        "data/frames_rgb/frame_001110.jpg",
        "data/frames_rgb/frame_001140.jpg",
        "data/frames_rgb/frame_001170.jpg",
        "data/frames_rgb/frame_001200.jpg",
        "data/frames_rgb/frame_008820.jpg",
    ]

    for frame in frames:
        frame_path = Path(frame)
        output_path = output_dir / f"{frame_path.stem}_task1.jpg"
        draw_prediction(frame_path, output_path, model)


if __name__ == "__main__":
    main()