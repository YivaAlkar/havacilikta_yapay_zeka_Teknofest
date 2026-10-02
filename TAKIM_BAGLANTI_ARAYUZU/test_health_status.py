from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions


def make_prediction(frame_name, tx, ty, tz):
    return FramePredictions(
        frame_url=f"/api/frames/{frame_name}/",
        image_url=f"/media/{frame_name}.jpg",
        video_name="health_test",
        gt_translation_x=tx,
        gt_translation_y=ty,
        gt_translation_z=tz,
    )


def print_result(title, result):
    print("\n" + title)
    print("Translations:", len(result.translations))
    for tr in result.translations:
        print("TR:", tr.translation_x, tr.translation_y, tr.translation_z)


def main():
    model = ObjectDetectionModel("http://test/")
    frame_path = "data/frames_rgb/frame_001110.jpg"

    # 1) Sağlıklı GPS: GT gönderilmeli
    pred1 = make_prediction("frame_001110", 10.0, 20.0, -5.0)
    result1 = model.detect(
        prediction=pred1,
        health_status="1",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=frame_path,
    )
    print_result("CASE 1 - health_status=1", result1)

    # 2) GPS bozuk: son bilinen translation gönderilmeli
    pred2 = make_prediction("frame_001140", None, None, None)
    result2 = model.detect(
        prediction=pred2,
        health_status="0",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=frame_path,
    )
    print_result("CASE 2 - health_status=0", result2)

    # 3) health_status None: translation gönderilmemeli
    pred3 = make_prediction("frame_001170", 100.0, 200.0, -50.0)
    result3 = model.detect(
        prediction=pred3,
        health_status=None,
        active_refs=[],
        ref_image_paths={},
        frame_image_path=frame_path,
    )
    print_result("CASE 3 - health_status=None", result3)

    # 4) İlk frame direkt bozuk gelirse: 0,0,0 göndermeli
    model2 = ObjectDetectionModel("http://test/")
    pred4 = make_prediction("frame_001200", None, None, None)
    result4 = model2.detect(
        prediction=pred4,
        health_status="0",
        active_refs=[],
        ref_image_paths={},
        frame_image_path=frame_path,
    )
    print_result("CASE 4 - first frame health_status=0", result4)


if __name__ == "__main__":
    main()