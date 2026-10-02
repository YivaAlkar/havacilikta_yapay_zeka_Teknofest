from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


FRAME_DIR = ROOT / "data" / "frames_rgb"


def get_frames():
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(FRAME_DIR.glob(ext))
    return sorted(files)


def make_prediction_for_frame(frame_path: Path):
    frame_name = frame_path.stem

    return FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/frames/{frame_path.name}",
        video_name="stats_rgb_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )


def main():
    model = ObjectDetectionModel("http://test/")

    frames = get_frames()
    print(f"[INFO] frame count: {len(frames)}")

    frames_with_objects = 0
    total_objects = 0

    class_counts = {
        0: 0,  # Tasit
        1: 0,  # Insan
        2: 0,  # UAP
        3: 0,  # UAI
    }

    examples = []

    for frame_path in frames:
        prediction = make_prediction_for_frame(frame_path)

        result = model.detect(
            prediction=prediction,
            health_status="1",
            active_refs=[],
            ref_image_paths={},
            frame_image_path=str(frame_path),
        )

        objects = getattr(result, "detected_objects", [])

        if objects:
            frames_with_objects += 1
            examples.append((frame_path.name, len(objects)))

        total_objects += len(objects)

        for obj in objects:
            cls = obj.cls

            # cls yanlışlıkla tuple gelirse güvenli düzelt
            if isinstance(cls, tuple):
                cls = cls[0]

            if cls in class_counts:
                class_counts[cls] += 1

    print("\n" + "=" * 80)
    print(f"Toplam frame: {len(frames)}")
    print(f"Tespit olan frame: {frames_with_objects}")
    print(f"Toplam obje: {total_objects}")
    print(f"Tasit: {class_counts[0]}")
    print(f"Insan: {class_counts[1]}")
    print(f"UAP: {class_counts[2]}")
    print(f"UAI: {class_counts[3]}")

    print("\nÖrnek tespitli frame'ler:")
    for name, count in examples[:30]:
        print(f"{name}: {count}")

    print("=" * 80)


if __name__ == "__main__":
    main()