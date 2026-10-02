from __future__ import annotations

from dataclasses import dataclass

from src.object_detection_model import ObjectDetectionModel


@dataclass
class DummyTranslation:
    translation_x: float
    translation_y: float
    translation_z: float


class DummyPrediction:
    def __init__(
        self,
        gt_x=None,
        gt_y=None,
        gt_z=None,
        frame_name="frame_000000",
    ):
        self.frame_url = f"/api/frames/{frame_name}/"
        self.image_url = f"/media/frames/{frame_name}.jpg"
        self.video_name = "ONLINE_YARISMA_2026"

        self.gt_translation_x = gt_x
        self.gt_translation_y = gt_y
        self.gt_translation_z = gt_z

        self.translations = []

    def add_translation_object(self, translation):
        self.translations.append(translation)


class DummyReader:
    def __init__(self, value):
        self.value = value

    def get_by_frame_name(self, frame_name):
        normalized = str(frame_name)

        if normalized in {
            "frame_000000",
            "frame_000000.jpg",
            "frame_000000.png",
        }:
            return self.value

        return None


def create_model_without_yolo():
    model = ObjectDetectionModel.__new__(
        ObjectDetectionModel
    )

    model.last_translation = None
    model.translation_readers = []
    model._translation_csv_ambiguity_logged = False

    return model


def assert_translation(
    prediction,
    expected,
    test_name,
):
    if len(prediction.translations) != 1:
        raise AssertionError(
            f"{test_name}: tam olarak bir translation bekleniyordu, "
            f"bulunan={len(prediction.translations)}"
        )

    result = prediction.translations[0]

    actual = (
        float(result.translation_x),
        float(result.translation_y),
        float(result.translation_z),
    )

    if actual != expected:
        raise AssertionError(
            f"{test_name}: expected={expected}, actual={actual}"
        )


def test_healthy_gt_has_priority_over_csv():
    model = create_model_without_yolo()

    model.translation_readers = [
        DummyReader((100.0, 200.0, 300.0))
    ]

    prediction = DummyPrediction(
        gt_x=1.0,
        gt_y=2.0,
        gt_z=3.0,
    )

    model._add_task_2_translation(
        prediction,
        health_status="1",
    )

    assert_translation(
        prediction,
        (1.0, 2.0, 3.0),
        "healthy_gt_has_priority_over_csv",
    )


def test_multiple_csv_readers_are_rejected():
    model = create_model_without_yolo()

    model.translation_readers = [
        DummyReader((10.0, 20.0, 30.0)),
        DummyReader((100.0, 200.0, 300.0)),
    ]

    prediction = DummyPrediction()

    result = model._get_csv_translation_for_prediction(
        prediction
    )

    if result is not None:
        raise AssertionError(
            "multiple_csv_readers_are_rejected: "
            f"None bekleniyordu, actual={result}"
        )


def test_unhealthy_uses_last_translation_when_csv_ambiguous():
    model = create_model_without_yolo()

    model.translation_readers = [
        DummyReader((10.0, 20.0, 30.0)),
        DummyReader((100.0, 200.0, 300.0)),
    ]

    model.last_translation = (4.0, 5.0, 6.0)

    prediction = DummyPrediction()

    model._add_task_2_translation(
        prediction,
        health_status="0",
    )

    assert_translation(
        prediction,
        (4.0, 5.0, 6.0),
        "unhealthy_uses_last_translation_when_csv_ambiguous",
    )


def test_single_csv_still_works_when_gt_missing():
    model = create_model_without_yolo()

    model.translation_readers = [
        DummyReader((7.0, 8.0, 9.0))
    ]

    prediction = DummyPrediction()

    model._add_task_2_translation(
        prediction,
        health_status="0",
    )

    assert_translation(
        prediction,
        (7.0, 8.0, 9.0),
        "single_csv_still_works_when_gt_missing",
    )


def test_health_none_adds_nothing():
    model = create_model_without_yolo()

    prediction = DummyPrediction(
        gt_x=1.0,
        gt_y=2.0,
        gt_z=3.0,
    )

    model._add_task_2_translation(
        prediction,
        health_status=None,
    )

    if prediction.translations:
        raise AssertionError(
            "health_none_adds_nothing: "
            "translation eklenmemeliydi."
        )


def main():
    tests = [
        test_healthy_gt_has_priority_over_csv,
        test_multiple_csv_readers_are_rejected,
        test_unhealthy_uses_last_translation_when_csv_ambiguous,
        test_single_csv_still_works_when_gt_missing,
        test_health_none_adds_nothing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"[PASS] {test.__name__}")

        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] {test.__name__}: "
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 70)
    print("TASK 2 SAFE PRIORITY TEST RESULT")
    print("=" * 70)
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"total : {len(tests)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()