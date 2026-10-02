from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


def make_prediction(frame_name, gt=(10.0, 20.0, -5.0)):
    return FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/frames/{frame_name}.jpg",
        video_name="task2_csv_test",
        gt_translation_x=gt[0],
        gt_translation_y=gt[1],
        gt_translation_z=gt[2],
    )


def print_result(title, result):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Translations:", len(result.translations))

    for tr in result.translations:
        print(
            "TR:",
            tr.translation_x,
            tr.translation_y,
            tr.translation_z,
        )


def main():
    model = ObjectDetectionModel("http://test/")

    frame_path = ROOT / "data" / "frames_rgb" / "frame_001110.jpg"

    # CSV'de olması beklenen frame
    prediction_csv = make_prediction("frame_001110")

    result_csv_healthy = model.detect(
        prediction=prediction_csv,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    print_result("CASE 1 - CSV frame + health_status=1", result_csv_healthy)

    # CSV'de olmayan frame; GT fallback çalışmalı
    prediction_missing = make_prediction("frame_DOES_NOT_EXIST", gt=(111.0, 222.0, 333.0))

    result_missing_healthy = model.detect(
        prediction=prediction_missing,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    print_result("CASE 2 - Missing CSV frame + health_status=1 -> GT fallback", result_missing_healthy)

    # health_status=None ise CSV olsa bile translation eklememeli
    prediction_none = make_prediction("frame_001110")

    result_none = model.detect(
        prediction=prediction_none,
        health_status=None,
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    print_result("CASE 3 - CSV frame + health_status=None -> no translation", result_none)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
    
