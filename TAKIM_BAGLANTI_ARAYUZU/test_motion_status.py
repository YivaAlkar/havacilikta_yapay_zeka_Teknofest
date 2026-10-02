from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


def run_frame(model, frame_name):
    frame_path = f"data/frames_rgb/{frame_name}.jpg"

    prediction = FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/{frame_name}.jpg",
        video_name="motion_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=frame_path,
    )

    print("\nFRAME:", frame_name)
    print("Detected objects:", len(result.detected_objects))

    for obj in result.detected_objects:
        print(
            "cls=", obj.cls,
            "moving=", obj.moving_status,
            "bbox=",
            round(float(obj.top_left_x), 1),
            round(float(obj.top_left_y), 1),
            round(float(obj.bottom_right_x), 1),
            round(float(obj.bottom_right_y), 1),
        )


def main():
    model = ObjectDetectionModel("http://test/")

    frames = [
        "frame_001110",
        "frame_001140",
        "frame_001170",
        "frame_001200",
    ]

    for frame_name in frames:
        run_frame(model, frame_name)


if __name__ == "__main__":
    main()