import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, step=30, max_frames=None):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Total frame: {total}, FPS: {fps}, Size: {width}x{height}")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved % 50 == 0:
                print(f"Kaydedilen frame: {saved}")

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Bitti. Kaydedilen toplam frame: {saved}")


if __name__ == "__main__":
    extract_frames(
        "data/raw/THYZ_2026_Ornek_Veri_1.MP4",
        "data/frames_rgb",
        step=30,
        max_frames=300
    )

    extract_frames(
        "data/raw/THYZ_2026_Ornek_Veri_2_Termal.MP4",
        "data/frames_thermal",
        step=30,
        max_frames=300
    )