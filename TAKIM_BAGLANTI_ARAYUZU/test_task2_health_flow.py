from __future__ import annotations

from src.object_detection_model import ObjectDetectionModel


class DummyPrediction:
    def __init__(
        self,
        gt_x=None,
        gt_y=None,
        gt_z=None,
        frame_index=0,
    ):
        frame_name = f"frame_{frame_index:06d}"

        self.frame_url = f"/api/frames/{frame_name}/"
        self.image_url = f"/media/frames/{frame_name}.jpg"
        self.video_name = "TASK2_FLOW_TEST"

        self.gt_translation_x = gt_x
        self.gt_translation_y = gt_y
        self.gt_translation_z = gt_z

        self.translations = []

    def add_translation_object(self, translation):
        self.translations.append(translation)


def create_model():
    model = ObjectDetectionModel.__new__(
        ObjectDetectionModel
    )

    model.previous_translation = None
    model.last_translation = None
    model.translation_velocity = (0.0, 0.0, 0.0)
    model.translation_stale_frames = 0

    model.max_translation_stale_frames = 8
    model.translation_velocity_damping = 0.85
    model.max_translation_step = 25.0
    model.max_reliable_translation_jump = 10.0

    # CSV çakışması olmaması için bu testte CSV kullanmıyoruz.
    model.translation_readers = []
    model._translation_csv_ambiguity_logged = False

    return model


def extract_translation(prediction):
    if len(prediction.translations) != 1:
        raise AssertionError(
            "Tam olarak 1 translation bekleniyordu, "
            f"bulunan={len(prediction.translations)}"
        )

    result = prediction.translations[0]

    return (
        float(result.translation_x),
        float(result.translation_y),
        float(result.translation_z),
    )


def assert_close(
    actual,
    expected,
    tolerance=1e-6,
):
    if actual is None:
        raise AssertionError(
            f"Translation None geldi. Beklenen={expected}"
        )

    for actual_value, expected_value in zip(
        actual,
        expected,
    ):
        if abs(actual_value - expected_value) > tolerance:
            raise AssertionError(
                f"Beklenen={expected}, bulunan={actual}"
            )


def test_complete_health_flow():
    model = create_model()

    # Frame 0: İlk güvenilir GT.
    frame_0 = DummyPrediction(
        gt_x=10.0,
        gt_y=20.0,
        gt_z=30.0,
        frame_index=0,
    )

    model._add_task_2_translation(
        frame_0,
        health_status="1",
    )

    assert_close(
        extract_translation(frame_0),
        (10.0, 20.0, 30.0),
    )

    # Frame 1: İkinci GT ile hız öğrenilir.
    frame_1 = DummyPrediction(
        gt_x=12.0,
        gt_y=23.0,
        gt_z=34.0,
        frame_index=1,
    )

    model._add_task_2_translation(
        frame_1,
        health_status="1",
    )

    assert_close(
        extract_translation(frame_1),
        (12.0, 23.0, 34.0),
    )

    assert_close(
        model.translation_velocity,
        (2.0, 3.0, 4.0),
    )

    # Frame 2: health=0, ilk hareket tahmini.
    frame_2 = DummyPrediction(
        frame_index=2,
    )

    model._add_task_2_translation(
        frame_2,
        health_status="0",
    )

    assert_close(
        extract_translation(frame_2),
        (14.0, 26.0, 38.0),
    )

    # Frame 3: İkinci tahmin damping ile ilerler.
    frame_3 = DummyPrediction(
        frame_index=3,
    )

    model._add_task_2_translation(
        frame_3,
        health_status="0",
    )

    expected_frame_3 = (
        14.0 + 2.0 * 0.85,
        26.0 + 3.0 * 0.85,
        38.0 + 4.0 * 0.85,
    )

    assert_close(
        extract_translation(frame_3),
        expected_frame_3,
    )

    # Frame 4: Sağlıklı GT geri gelir.
    # Tahmin geçmişi yerine gerçek GT kabul edilmelidir.
    frame_4 = DummyPrediction(
        gt_x=16.0,
        gt_y=29.0,
        gt_z=42.0,
        frame_index=4,
    )

    model._add_task_2_translation(
        frame_4,
        health_status="1",
    )

    assert_close(
        extract_translation(frame_4),
        (16.0, 29.0, 42.0),
    )

    if model.translation_stale_frames != 0:
        raise AssertionError(
            "Yeni GT sonrasında stale frame sayısı "
            "sıfırlanmalıydı."
        )


def test_health_none_produces_no_output():
    model = create_model()

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
            "health_status=None iken translation "
            "eklenmemeliydi."
        )


def test_unknown_health_produces_no_output():
    model = create_model()

    prediction = DummyPrediction(
        gt_x=1.0,
        gt_y=2.0,
        gt_z=3.0,
    )

    model._add_task_2_translation(
        prediction,
        health_status="unexpected",
    )

    if prediction.translations:
        raise AssertionError(
            "Bilinmeyen health_status için translation "
            "eklenmemeliydi."
        )


def test_stale_history_expires():
    model = create_model()

    model._record_reliable_translation(
        (0.0, 0.0, 0.0)
    )
    model._record_reliable_translation(
        (1.0, 1.0, 1.0)
    )

    for frame_index in range(
        model.max_translation_stale_frames
    ):
        prediction = DummyPrediction(
            frame_index=frame_index,
        )

        model._add_task_2_translation(
            prediction,
            health_status="0",
        )

        if len(prediction.translations) != 1:
            raise AssertionError(
                "Yaş sınırı dolmadan translation "
                "üretilmeliydi."
            )

    expired_prediction = DummyPrediction(
        frame_index=100,
    )

    model._add_task_2_translation(
        expired_prediction,
        health_status="0",
    )

    # Mevcut üretim davranışı geçmiş bayatsa sıfır fallback üretir.
    assert_close(
        extract_translation(expired_prediction),
        (0.0, 0.0, 0.0),
    )


def test_jump_guard_in_full_flow():
    model = create_model()

    first = DummyPrediction(
        gt_x=0.0,
        gt_y=0.0,
        gt_z=0.0,
        frame_index=0,
    )

    model._add_task_2_translation(
        first,
        health_status="1",
    )

    jump = DummyPrediction(
        gt_x=100.0,
        gt_y=200.0,
        gt_z=300.0,
        frame_index=1,
    )

    model._add_task_2_translation(
        jump,
        health_status="1",
    )

    assert_close(
        model.translation_velocity,
        (0.0, 0.0, 0.0),
    )

    unhealthy = DummyPrediction(
        frame_index=2,
    )

    model._add_task_2_translation(
        unhealthy,
        health_status="0",
    )

    assert_close(
        extract_translation(unhealthy),
        (100.0, 200.0, 300.0),
    )


def main():
    tests = [
        test_complete_health_flow,
        test_health_none_produces_no_output,
        test_unknown_health_produces_no_output,
        test_stale_history_expires,
        test_jump_guard_in_full_flow,
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
    print("=" * 72)
    print("TASK 2 HEALTH FLOW INTEGRATION TEST")
    print("=" * 72)
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"total : {len(tests)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()