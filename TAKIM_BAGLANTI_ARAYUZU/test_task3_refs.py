from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


def main():
    model = ObjectDetectionModel("http://test/")

    prediction = FramePredictions(
        frame_url="/api/frames/frame_001110/",
        image_url="/media/frame_001110.jpg",
        video_name="task3_thermal_test",
        gt_translation_x=1.0,
        gt_translation_y=2.0,
        gt_translation_z=3.0,
    )

    active_refs = [
        {
            "url": "/media/references/Referans_Nesne_01.png",
            "frame_start_image_url": "/media/frame_001000.jpg",
            "frame_end_image_url": "/media/frame_001300.jpg",
        }
    ]

    ref_image_paths = {
        "/media/references/Referans_Nesne_01.png": "data/references_thermal/Referans_Nesne_01.png"
    }

    result = model.detect(
        prediction=prediction,
        health_status="1",
        active_refs=active_refs,
        ref_image_paths=ref_image_paths,
        frame_image_path="data/frames_thermal/frame_001110.jpg",
    )

    print("Detected objects:", len(result.detected_objects))
    print("Translations:", len(result.translations))
    print("Reference predictions:", len(result.reference_predictions))

    for rp in result.reference_predictions:
        print(
            "REF:",
            rp.reference_url,
            rp.frame_url,
            rp.top_left_x,
            rp.top_left_y,
            rp.bottom_right_x,
            rp.bottom_right_y,
        )


if __name__ == "__main__":
    main()