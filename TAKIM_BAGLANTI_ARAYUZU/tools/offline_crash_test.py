from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions

try:
    from src_custom.reference_matcher import match_reference
except Exception:
    match_reference = None

try:
    from src_custom.csv_utils import TranslationReader
except Exception:
    TranslationReader = None


def make_prediction(frame_name="frame_001110"):
    return FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/frames/{frame_name}.jpg",
        video_name="offline_crash_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )


def run_case(name, fn, should_raise=False):
    print("\n" + "=" * 80)
    print(f"[CASE] {name}")
    print("=" * 80)

    try:
        result = fn()
        print("[OK] Çökmedi.")
        if result is not None:
            print(result)

        if should_raise:
            print("[WARN] Bu case hata vermeliydi ama vermedi.")
            return False

        return True

    except Exception as e:
        print(f"[EXCEPTION] {type(e).__name__}: {e}")
        traceback.print_exc()

        if should_raise:
            print("[OK] Beklenen hata yakalandı.")
            return True

        print("[FAIL] Beklenmeyen hata.")
        return False


def case_valid_detect():
    model = ObjectDetectionModel("http://test/")
    prediction = make_prediction("frame_001110")

    frame_path = ROOT / "data" / "frames_rgb" / "frame_001110.jpg"

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    return (
        f"objects={len(result.detected_objects)}, "
        f"translations={len(result.translations)}, "
        f"refs={len(result.reference_predictions)}"
    )


def case_missing_frame_path():
    model = ObjectDetectionModel("http://test/")
    prediction = make_prediction("missing_frame")

    missing_path = ROOT / "data" / "frames_rgb" / "does_not_exist.jpg"

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(missing_path),
    )

    return (
        f"objects={len(result.detected_objects)}, "
        f"translations={len(result.translations)}, "
        f"refs={len(result.reference_predictions)}"
    )


def case_health_none():
    model = ObjectDetectionModel("http://test/")
    prediction = make_prediction("frame_001110")

    frame_path = ROOT / "data" / "frames_rgb" / "frame_001110.jpg"

    result = model.detect(
        prediction=prediction,
        health_status=None,
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    return (
        f"objects={len(result.detected_objects)}, "
        f"translations={len(result.translations)}, "
        f"refs={len(result.reference_predictions)}"
    )


def case_empty_refs():
    model = ObjectDetectionModel("http://test/")
    prediction = make_prediction("frame_001110")

    frame_path = ROOT / "data" / "frames_rgb" / "frame_001110.jpg"

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=str(frame_path),
    )

    return f"refs={len(result.reference_predictions)}"


def case_reference_matcher_bad_paths():
    if match_reference is None:
        return "match_reference import edilemedi, skip"

    bad_frame = ROOT / "data" / "frames_rgb" / "missing.jpg"
    bad_ref = ROOT / "data" / "references_rgb" / "missing.jpg"

    bbox = match_reference(str(bad_frame), str(bad_ref), debug=True)

    return f"bbox={bbox}"


def case_csv_reader_missing_frame():
    if TranslationReader is None:
        return "TranslationReader import edilemedi, skip"

    csv_dir = ROOT / "data" / "translations"
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        return "CSV bulunamadı, skip"

    reader = TranslationReader(csv_files[0])

    by_name = reader.get_by_frame_name("definitely_missing_frame.jpg")
    by_index = reader.get_by_frame_index(99999999)

    return f"missing_name={by_name}, missing_index={by_index}"


def main():
    cases = [
        ("valid detect", case_valid_detect),
        ("missing frame path", case_missing_frame_path),
        ("health_status None", case_health_none),
        ("empty refs", case_empty_refs),
        ("reference matcher bad paths", case_reference_matcher_bad_paths),
        ("csv reader missing frame", case_csv_reader_missing_frame),
    ]

    passed = 0
    failed = 0

    for name, fn in cases:
        ok = run_case(name, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"passed={passed}")
    print(f"failed={failed}")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()