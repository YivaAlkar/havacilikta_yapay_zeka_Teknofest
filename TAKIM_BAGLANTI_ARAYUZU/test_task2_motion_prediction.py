from src.object_detection_model import ObjectDetectionModel


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

    return model


def assert_close(actual, expected, tolerance=1e-6):
    if len(actual) != len(expected):
        raise AssertionError(
            f"Boyut farklı: {actual}, {expected}"
        )

    for actual_value, expected_value in zip(
        actual,
        expected,
    ):
        if abs(actual_value - expected_value) > tolerance:
            raise AssertionError(
                f"Beklenen={expected}, bulunan={actual}"
            )


def test_velocity_is_learned():
    model = create_model()

    model._record_reliable_translation(
        (10.0, 20.0, 30.0)
    )
    model._record_reliable_translation(
        (12.0, 23.0, 34.0)
    )

    assert_close(
        model.translation_velocity,
        (2.0, 3.0, 4.0),
    )


def test_first_prediction_uses_velocity():
    model = create_model()

    model._record_reliable_translation(
        (10.0, 20.0, 30.0)
    )
    model._record_reliable_translation(
        (12.0, 23.0, 34.0)
    )

    prediction = (
        model._predict_short_term_translation()
    )

    assert_close(
        prediction,
        (14.0, 26.0, 38.0),
    )


def test_second_prediction_is_damped():
    model = create_model()

    model._record_reliable_translation(
        (10.0, 20.0, 30.0)
    )
    model._record_reliable_translation(
        (12.0, 23.0, 34.0)
    )

    first = model._predict_short_term_translation()
    second = model._predict_short_term_translation()

    assert_close(
        first,
        (14.0, 26.0, 38.0),
    )

    assert_close(
        second,
        (
            14.0 + 2.0 * 0.85,
            26.0 + 3.0 * 0.85,
            38.0 + 4.0 * 0.85,
        ),
    )


def test_prediction_expires():
    model = create_model()

    model._record_reliable_translation(
        (0.0, 0.0, 0.0)
    )
    model._record_reliable_translation(
        (1.0, 1.0, 1.0)
    )

    for _ in range(
        model.max_translation_stale_frames
    ):
        result = (
            model._predict_short_term_translation()
        )

        if result is None:
            raise AssertionError(
                "Tahmin erken sona erdi."
            )

    if (
        model._predict_short_term_translation()
        is not None
    ):
        raise AssertionError(
            "Yaş sınırından sonra None bekleniyordu."
        )


def test_large_velocity_is_clamped():
    model = create_model()

    model._record_reliable_translation(
        (0.0, 0.0, 0.0)
    )
    model._record_reliable_translation(
        (100.0, -100.0, 50.0)
    )

    assert_close(
        model.translation_velocity,
        (25.0, -25.0, 25.0),
    )

def test_normal_motion_updates_velocity():
    model = create_model()

    model._record_reliable_translation(
        (10.0, 20.0, 30.0)
    )
    model._record_reliable_translation(
        (12.0, 23.0, 34.0)
    )

    assert_close(
        model.translation_velocity,
        (2.0, 3.0, 4.0),
    )


def test_large_jump_resets_velocity():
    model = create_model()

    model._record_reliable_translation(
        (10.0, 20.0, 30.0)
    )

    result = model._record_reliable_translation(
        (100.0, 200.0, 300.0)
    )

    # Yeni koordinat reddedilmemeli.
    assert_close(
        result,
        (100.0, 200.0, 300.0),
    )

    assert_close(
        model.last_translation,
        (100.0, 200.0, 300.0),
    )

    # Ancak bu sıçramadan hız öğrenilmemeli.
    assert_close(
        model.translation_velocity,
        (0.0, 0.0, 0.0),
    )


def test_prediction_after_jump_does_not_run_away():
    model = create_model()

    model._record_reliable_translation(
        (0.0, 0.0, 0.0)
    )
    model._record_reliable_translation(
        (100.0, 200.0, 300.0)
    )

    prediction = (
        model._predict_short_term_translation()
    )

    # Velocity sıfırlandığı için son konum korunmalı.
    assert_close(
        prediction,
        (100.0, 200.0, 300.0),
    )


def test_velocity_recovers_after_jump():
    model = create_model()

    model._record_reliable_translation(
        (0.0, 0.0, 0.0)
    )

    # Sıçrama: velocity sıfırlanır.
    model._record_reliable_translation(
        (100.0, 100.0, 100.0)
    )

    assert_close(
        model.translation_velocity,
        (0.0, 0.0, 0.0),
    )

    # Sonraki normal hareketten yeniden hız öğrenilir.
    model._record_reliable_translation(
        (101.0, 102.0, 103.0)
    )

    assert_close(
        model.translation_velocity,
        (1.0, 2.0, 3.0),
    )

def main():
    tests = [
        test_velocity_is_learned,
        test_first_prediction_uses_velocity,
        test_second_prediction_is_damped,
        test_prediction_expires,
        test_normal_motion_updates_velocity,
        test_large_jump_resets_velocity,
        test_prediction_after_jump_does_not_run_away,
        test_velocity_recovers_after_jump,
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
    print("TASK 2 MOTION PREDICTION TEST")
    print("=" * 70)
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"total : {len(tests)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()