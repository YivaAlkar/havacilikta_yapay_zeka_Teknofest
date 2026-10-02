from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


def main():
    test_image_path = "data/frames_rgb/frame_001110.jpg"

    prediction = FramePredictions(
        frame_url="/api/frames/frame_001110/",
        image_url="/media/frame_001110.jpg",
        video_name="local_rgb_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )

    model = ObjectDetectionModel("http://test/")

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=test_image_path,
    )

    print("Detected objects:", len(result.detected_objects))
    for obj in result.detected_objects:
        print(
            "OBJ:",
            "cls=", obj.cls,
            "landing=", obj.landing_status,
            "moving=", obj.moving_status,
            "bbox=",
            obj.top_left_x,
            obj.top_left_y,
            obj.bottom_right_x,
            obj.bottom_right_y,
        )

    print("Translations:", len(result.translations))
    for tr in result.translations:
        print("TR:", tr.translation_x, tr.translation_y, tr.translation_z)

    print("Reference predictions:", len(result.reference_predictions))


if __name__ == "__main__":
    main()