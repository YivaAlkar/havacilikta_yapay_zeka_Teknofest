import cv2
from pathlib import Path

from src_custom.reference_matcher import match_reference


def draw_bbox(frame_path, bbox, output_path):
    image = cv2.imread(str(frame_path))

    if image is None:
        raise RuntimeError(f"Frame okunamadı: {frame_path}")

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            image,
            "REFERENCE MATCH",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_test(frame_path, ref_path, output_path):
    bbox = match_reference(
    frame_path,
    ref_path,
    debug=True,
    )

    print("\nFrame:", frame_path)
    print("Reference:", ref_path)
    print("BBox:", bbox)

    draw_bbox(frame_path, bbox, output_path)
    print("Kaydedildi:", output_path)


def main():
    tests = [
        (
            "data/frames_rgb/frame_001110.jpg",
            "data/references_rgb/Referans_Nesne_01.JPG",
            "outputs/tests/reference_match_rgb_01.jpg",
        ),
        (
            "data/frames_rgb/frame_001110.jpg",
            "data/references_rgb/Referans_Nesne_02.JPG",
            "outputs/tests/reference_match_rgb_02.jpg",
        ),
        (
            "data/frames_thermal/frame_001110.jpg",
            "data/references_thermal/Referans_Nesne_01.png",
            "outputs/tests/reference_match_thermal_01.jpg",
        ),
    ]

    for frame_path, ref_path, output_path in tests:
        run_test(frame_path, ref_path, output_path)


if __name__ == "__main__":
    main()