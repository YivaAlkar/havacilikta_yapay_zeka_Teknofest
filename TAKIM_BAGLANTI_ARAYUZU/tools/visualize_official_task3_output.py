from pathlib import Path
import sys
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel


OUT_DIR = ROOT / "outputs" / "task3_official_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_bbox(frame_path: Path, bbox, label: str, out_path: Path):
    img = cv2.imread(str(frame_path))

    if img is None:
        print(f"[ERR] Frame okunamadı: {frame_path}")
        return False

    x1, y1, x2, y2 = [int(float(v)) for v in bbox]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        img,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), img)
    return True


def find_frame_from_api_path(api_path: str):
    """
    /api/frames/frame_001110/ gibi path'ten frame adını çıkarır.
    Sonra hem rgb hem thermal klasöründe arar.
    """
    parts = [p for p in api_path.replace("\\", "/").split("/") if p]
    frame_name = None

    for p in parts:
        if p.startswith("frame_"):
            frame_name = p
            break

    if frame_name is None:
        return None

    candidates = [
        ROOT / "data" / "frames_rgb" / f"{frame_name}.jpg",
        ROOT / "data" / "frames_rgb" / f"{frame_name}.png",
        ROOT / "data" / "frames_thermal" / f"{frame_name}.jpg",
        ROOT / "data" / "frames_thermal" / f"{frame_name}.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def main():
    print("[INFO] Official Task 3 output visualization başlıyor.")

    model = ObjectDetectionModel(evaluation_server_url="http://localhost")

    # test_task3_refs.py ile aynı mantık:
    frame_image_path = "/api/frames/frame_001110/"
    reference_image_paths = [
        "/media/references/Referans_Nesne_01.png",
    ]

    prediction = model.detect(
        frame_image_path=frame_image_path,
        health_status="1",
        reference_image_paths=reference_image_paths,
    )

    refs = getattr(prediction, "reference_predictions", [])

    print(f"[INFO] Reference predictions: {len(refs)}")

    frame_path = find_frame_from_api_path(frame_image_path)

    if frame_path is None:
        print(f"[ERR] Frame bulunamadı: {frame_image_path}")
        return

    print(f"[INFO] Frame local path: {frame_path}")

    for idx, ref in enumerate(refs):
        ref_path = getattr(ref, "reference_image_path", "ref")
        x1 = getattr(ref, "x1", None)
        y1 = getattr(ref, "y1", None)
        x2 = getattr(ref, "x2", None)
        y2 = getattr(ref, "y2", None)

        print(f"[REF {idx}] {ref_path} bbox={x1}, {y1}, {x2}, {y2}")

        if None in [x1, y1, x2, y2]:
            print("[WARN] bbox eksik, çizilmedi.")
            continue

        out_path = OUT_DIR / f"official_task3_ref_{idx}_{frame_path.stem}.jpg"

        ok = draw_bbox(
            frame_path=frame_path,
            bbox=(x1, y1, x2, y2),
            label=f"official_ref_{idx}",
            out_path=out_path,
        )

        if ok:
            print(f"[OK] Kaydedildi: {out_path}")

    print(f"[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()