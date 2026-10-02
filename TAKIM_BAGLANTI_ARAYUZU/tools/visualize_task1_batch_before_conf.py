from pathlib import Path
import sys
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


OUT_DIR = ROOT / "outputs" / "task1_batch_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_DIR = ROOT / "data" / "frames_rgb"


def get_frames():
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(FRAME_DIR.glob(ext))
    return sorted(files)


def draw_objects(frame_path, objects, out_path):
    img = cv2.imread(str(frame_path))

    if img is None:
        print(f"[ERR] frame okunamadı: {frame_path}")
        return False

    for obj in objects:
        cls = getattr(obj, "cls", "?")
        landing = getattr(obj, "landing_status", "?")
        moving = getattr(obj, "moving_status", "?")

        x1 = int(float(getattr(obj, "top_left_x", 0)))
        y1 = int(float(getattr(obj, "top_left_y", 0)))
        x2 = int(float(getattr(obj, "bottom_right_x", 0)))
        y2 = int(float(getattr(obj, "bottom_right_y", 0)))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label = f"cls={cls} land={landing} mov={moving}"
        cv2.putText(
            img,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img)
    return True


def make_prediction_for_frame(frame_path: Path):
    frame_name = frame_path.stem

    prediction = FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/frames/{frame_path.name}",
        video_name="batch_rgb_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )

    return prediction


def main():
    print("[INFO] Task 1 batch visualization başlıyor.")

    model = ObjectDetectionModel("http://test/")

    frames = get_frames()
    print(f"[INFO] frame count: {len(frames)}")

    selected = frames[::10]
    print(f"[INFO] selected count: {len(selected)}")

    total_objects = 0
    saved = 0

    for idx, frame_path in enumerate(selected):
        prediction = make_prediction_for_frame(frame_path)

        result = model.detect(
            prediction=prediction,
            health_status="1",
            active_refs=[],
            ref_image_paths={},
            frame_image_path=str(frame_path),
        )

        objects = getattr(result, "detected_objects", [])
        total_objects += len(objects)

        out_path = OUT_DIR / f"{frame_path.stem}_task1.jpg"

        ok = draw_objects(frame_path, objects, out_path)

        if ok:
            saved += 1
            print(f"[OK] {frame_path.name}: objects={len(objects)} -> {out_path}")

    print("\n" + "=" * 80)
    print(f"[DONE] saved={saved}")
    print(f"[DONE] total_objects={total_objects}")
    print(f"[OUT] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()