from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel
from src.reference_prediction import ReferencePrediction
from src_custom.reference_matcher import match_reference


def create_model_without_yolo() -> ObjectDetectionModel:
    model = ObjectDetectionModel.__new__(ObjectDetectionModel)
    return model


def create_prediction(
    frame_index: int = 1110,
) -> FramePredictions:
    frame_name = f"frame_{frame_index:06d}"

    return FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/{frame_name}.jpg",
        video_name="TASK3_INTEGRATION_TEST",
        gt_translation_x=None,
        gt_translation_y=None,
        gt_translation_z=None,
    )


def create_reference(
    reference_url: str = "/api/reference/1/",
    start_frame: int = 1000,
    end_frame: int = 1300,
) -> dict:
    return {
        "url": reference_url,
        "frame_start_image_url": (
            f"/media/frame_{start_frame:06d}.jpg"
        ),
        "frame_end_image_url": (
            f"/media/frame_{end_frame:06d}.jpg"
        ),
    }


def assert_bbox_valid(
    bbox,
    frame_width: int,
    frame_height: int,
) -> None:
    if bbox is None:
        raise AssertionError("BBox None geldi.")

    if len(bbox) != 4:
        raise AssertionError(
            f"BBox 4 elemanlı olmalı: {bbox}"
        )

    x1, y1, x2, y2 = map(float, bbox)

    if not (0 <= x1 < x2 < frame_width):
        raise AssertionError(
            f"X koordinatları geçersiz: {bbox}"
        )

    if not (0 <= y1 < y2 < frame_height):
        raise AssertionError(
            f"Y koordinatları geçersiz: {bbox}"
        )


def test_active_reference_adds_prediction():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=1110)

    reference_url = "/api/reference/positive/"
    active_refs = [
        create_reference(
            reference_url=reference_url,
            start_frame=1000,
            end_frame=1300,
        )
    ]

    ref_image_paths = {
        reference_url: "dummy_reference.jpg",
    }

    expected_bbox = (100, 120, 240, 300)

    with patch(
        "src.object_detection_model.match_reference",
        return_value=expected_bbox,
    ):
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=active_refs,
            ref_image_paths=ref_image_paths,
            frame_image_path="dummy_frame.jpg",
        )

    if len(prediction.reference_predictions) != 1:
        raise AssertionError(
            "Aktif ve pozitif referansta 1 prediction bekleniyordu, "
            f"bulunan={len(prediction.reference_predictions)}"
        )

    result = prediction.reference_predictions[0]

    actual_bbox = (
        result.top_left_x,
        result.top_left_y,
        result.bottom_right_x,
        result.bottom_right_y,
    )

    if actual_bbox != expected_bbox:
        raise AssertionError(
            f"BBox yanlış. Beklenen={expected_bbox}, "
            f"bulunan={actual_bbox}"
        )

    if result.reference_url != reference_url:
        raise AssertionError(
            "Reference URL yanlış."
        )

    if result.frame_url != prediction.frame_url:
        raise AssertionError(
            "Frame URL yanlış."
        )


def test_negative_match_adds_nothing():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=1110)

    reference_url = "/api/reference/negative/"

    with patch(
        "src.object_detection_model.match_reference",
        return_value=None,
    ):
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                )
            ],
            ref_image_paths={
                reference_url: "dummy_reference.jpg",
            },
            frame_image_path="dummy_frame.jpg",
        )

    if prediction.reference_predictions:
        raise AssertionError(
            "Negatif eşleşmede prediction eklenmemeliydi."
        )


def test_before_start_frame_is_skipped():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=999)

    reference_url = "/api/reference/range/"

    with patch(
        "src.object_detection_model.match_reference"
    ) as mocked_matcher:
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                    start_frame=1000,
                    end_frame=1300,
                )
            ],
            ref_image_paths={
                reference_url: "dummy_reference.jpg",
            },
            frame_image_path="dummy_frame.jpg",
        )

    mocked_matcher.assert_not_called()

    if prediction.reference_predictions:
        raise AssertionError(
            "Başlangıçtan önce prediction olmamalı."
        )


def test_start_frame_is_inclusive():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=1000)

    reference_url = "/api/reference/start/"

    with patch(
        "src.object_detection_model.match_reference",
        return_value=(10, 20, 50, 80),
    ) as mocked_matcher:
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                    start_frame=1000,
                    end_frame=1300,
                )
            ],
            ref_image_paths={
                reference_url: "dummy_reference.jpg",
            },
            frame_image_path="dummy_frame.jpg",
        )

    mocked_matcher.assert_called_once()

    if len(prediction.reference_predictions) != 1:
        raise AssertionError(
            "Başlangıç frame'i kapsayıcı olmalı."
        )


def test_end_frame_is_inclusive():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=1300)

    reference_url = "/api/reference/end/"

    with patch(
        "src.object_detection_model.match_reference",
        return_value=(10, 20, 50, 80),
    ) as mocked_matcher:
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                    start_frame=1000,
                    end_frame=1300,
                )
            ],
            ref_image_paths={
                reference_url: "dummy_reference.jpg",
            },
            frame_image_path="dummy_frame.jpg",
        )

    mocked_matcher.assert_called_once()

    if len(prediction.reference_predictions) != 1:
        raise AssertionError(
            "Bitiş frame'i kapsayıcı olmalı."
        )


def test_after_end_frame_is_skipped():
    model = create_model_without_yolo()
    prediction = create_prediction(frame_index=1301)

    reference_url = "/api/reference/after/"

    with patch(
        "src.object_detection_model.match_reference"
    ) as mocked_matcher:
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                    start_frame=1000,
                    end_frame=1300,
                )
            ],
            ref_image_paths={
                reference_url: "dummy_reference.jpg",
            },
            frame_image_path="dummy_frame.jpg",
        )

    mocked_matcher.assert_not_called()

    if prediction.reference_predictions:
        raise AssertionError(
            "Bitiş frame'inden sonra prediction olmamalı."
        )


def test_missing_reference_url_is_safe():
    model = create_model_without_yolo()
    prediction = create_prediction()

    broken_reference = create_reference()
    broken_reference.pop("url")

    model._add_task_3_reference_predictions(
        prediction=prediction,
        active_refs=[broken_reference],
        ref_image_paths={},
        frame_image_path="dummy_frame.jpg",
    )

    if prediction.reference_predictions:
        raise AssertionError(
            "URL olmayan referans atlanmalıydı."
        )


def test_missing_reference_path_is_safe():
    model = create_model_without_yolo()
    prediction = create_prediction()

    reference_url = "/api/reference/missing-path/"

    with patch(
        "src.object_detection_model.match_reference"
    ) as mocked_matcher:
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=[
                create_reference(
                    reference_url=reference_url,
                )
            ],
            ref_image_paths={},
            frame_image_path="dummy_frame.jpg",
        )

    mocked_matcher.assert_not_called()

    if prediction.reference_predictions:
        raise AssertionError(
            "Referans yolu yokken prediction olmamalı."
        )


def test_missing_frame_image_does_not_crash():
    result = match_reference(
        "does_not_exist_frame.jpg",
        "does_not_exist_reference.jpg",
    )

    if result is not None:
        raise AssertionError(
            "Eksik görüntülerde None bekleniyordu."
        )


def test_multiple_references_are_independent():
    model = create_model_without_yolo()
    prediction = create_prediction()

    first_url = "/api/reference/first/"
    second_url = "/api/reference/second/"
    third_url = "/api/reference/third/"

    active_refs = [
        create_reference(reference_url=first_url),
        create_reference(reference_url=second_url),
        create_reference(reference_url=third_url),
    ]

    ref_image_paths = {
        first_url: "first.jpg",
        second_url: "second.jpg",
        third_url: "third.jpg",
    }

    def fake_match(frame_path, reference_path):
        del frame_path

        if reference_path == "first.jpg":
            return (10, 20, 40, 60)

        if reference_path == "second.jpg":
            return None

        if reference_path == "third.jpg":
            return (100, 120, 180, 220)

        raise AssertionError(
            f"Beklenmeyen referans yolu: {reference_path}"
        )

    with patch(
        "src.object_detection_model.match_reference",
        side_effect=fake_match,
    ):
        model._add_task_3_reference_predictions(
            prediction=prediction,
            active_refs=active_refs,
            ref_image_paths=ref_image_paths,
            frame_image_path="dummy_frame.jpg",
        )

    if len(prediction.reference_predictions) != 2:
        raise AssertionError(
            "Üç referanstan iki pozitif sonuç bekleniyordu, "
            f"bulunan={len(prediction.reference_predictions)}"
        )

    returned_urls = {
        item.reference_url
        for item in prediction.reference_predictions
    }

    expected_urls = {
        first_url,
        third_url,
    }

    if returned_urls != expected_urls:
        raise AssertionError(
            f"URL kümesi yanlış. Beklenen={expected_urls}, "
            f"bulunan={returned_urls}"
        )


def test_reference_payload_is_correct():
    prediction = ReferencePrediction(
        reference_url="/api/reference/42/",
        frame_url="/api/frames/frame_001110/",
        top_left_x=10,
        top_left_y=20,
        bottom_right_x=50,
        bottom_right_y=80,
    )

    payload = prediction.create_payload()

    expected = {
        "reference": "/api/reference/42/",
        "frame": "/api/frames/frame_001110/",
        "top_left_x": "10",
        "top_left_y": "20",
        "bottom_right_x": "50",
        "bottom_right_y": "80",
    }

    if payload != expected:
        raise AssertionError(
            f"Payload yanlış.\nBeklenen={expected}\nBulunan={payload}"
        )


def test_real_matcher_with_synthetic_positive():
    with tempfile.TemporaryDirectory(
        prefix="task3_positive_"
    ) as temporary_directory:
        root = Path(temporary_directory)

        frame_path = root / "frame.jpg"
        reference_path = root / "reference.jpg"

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        # Referans olacak ayırt edici sentetik nesne.
        reference = np.zeros(
            (100, 140, 3),
            dtype=np.uint8,
        )

        cv2.rectangle(
            reference,
            (5, 5),
            (134, 94),
            (255, 255, 255),
            thickness=4,
        )
        cv2.line(
            reference,
            (10, 85),
            (125, 15),
            (255, 255, 255),
            thickness=3,
        )
        cv2.circle(
            reference,
            (70, 50),
            20,
            (255, 255, 255),
            thickness=3,
        )
        cv2.putText(
            reference,
            "R42",
            (35, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        target_x = 260
        target_y = 180

        frame[
            target_y:target_y + reference.shape[0],
            target_x:target_x + reference.shape[1],
        ] = reference

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(reference_path), reference)

        bbox = match_reference(
            frame_path,
            reference_path,
            debug=True,
        )

        assert_bbox_valid(
            bbox,
            frame_width=640,
            frame_height=480,
        )

        x1, y1, x2, y2 = bbox

        # Tam piksel eşleşmesi beklemek yerine makul yakınlık kontrolü.
        if abs(x1 - target_x) > 30:
            raise AssertionError(
                f"Pozitif eşleşme X konumu uzak: {bbox}"
            )

        if abs(y1 - target_y) > 30:
            raise AssertionError(
                f"Pozitif eşleşme Y konumu uzak: {bbox}"
            )


def test_real_matcher_with_synthetic_negative():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    reference = np.zeros(
        (100, 140, 3),
        dtype=np.uint8,
    )

    cv2.rectangle(
        reference,
        (5, 5),
        (134, 94),
        (255, 255, 255),
        thickness=4,
    )
    cv2.putText(
        reference,
        "NO_MATCH",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Frame tamamen farklı bir desen içeriyor.
    cv2.circle(
        frame,
        (320, 240),
        100,
        (255, 255, 255),
        thickness=8,
    )

    bbox = match_reference(
        frame,
        reference,
        debug=True,
    )

    if bbox is not None:
        raise AssertionError(
            f"Negatif sentetik örnekte None bekleniyordu: {bbox}"
        )


def main() -> None:
    tests = [
        test_active_reference_adds_prediction,
        test_negative_match_adds_nothing,
        test_before_start_frame_is_skipped,
        test_start_frame_is_inclusive,
        test_end_frame_is_inclusive,
        test_after_end_frame_is_skipped,
        test_missing_reference_url_is_safe,
        test_missing_reference_path_is_safe,
        test_missing_frame_image_does_not_crash,
        test_multiple_references_are_independent,
        test_reference_payload_is_correct,
        test_real_matcher_with_synthetic_positive,
        test_real_matcher_with_synthetic_negative,
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
    print("=" * 76)
    print("TASK 3 INTEGRATION TEST RESULT")
    print("=" * 76)
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"total : {len(tests)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()