from pathlib import Path
import sys
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_custom.reference_matcher import match_reference


FRAME_DIRS = [
    ROOT / "data" / "frames_rgb",
    ROOT / "data" / "frames_thermal",
]

REF_DIRS = [
    ROOT / "data" / "references_rgb",
    ROOT / "data" / "references_thermal",
]

OUT_DIR = ROOT / "outputs" / "task3_refs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Görsel okunamadı: {path}")
    return img


def normalize_bbox(result):
    if result is None:
        return None

    if isinstance(result, dict):
        result = result.get("bbox", None)
        if result is None:
            return None

    if isinstance(result, (list, tuple)) and len(result) >= 4:
        x1, y1, x2, y2 = result[:4]
        return int(x1), int(y1), int(x2), int(y2)

    return None


def call_matcher(frame_img, ref_img, frame_path: Path, ref_path: Path):
    attempts = [
        lambda: match_reference(frame_img, ref_img),
        lambda: match_reference(frame_img, ref_img, debug=False),
        lambda: match_reference(str(frame_path), str(ref_path)),
        lambda: match_reference(str(frame_path), str(ref_path), debug=False),
    ]

    last_error = None

    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_error = e
            continue

    raise last_error


def draw_bbox(img, bbox, label):
    x1, y1, x2, y2 = bbox
    out = img.copy()

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        out,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def get_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def main():
    print("[INFO] Task 3 reference visualization başlıyor.")

    total_matches = 0

    for frame_dir, ref_dir in zip(FRAME_DIRS, REF_DIRS):
        if not frame_dir.exists():
            print(f"[SKIP] Frame klasörü yok: {frame_dir}")
            continue

        if not ref_dir.exists():
            print(f"[SKIP] Referans klasörü yok: {ref_dir}")
            continue

        frame_files = get_images(frame_dir)
        ref_files = get_images(ref_dir)

        print(f"\n[INFO] Frame dir: {frame_dir}")
        print(f"[INFO] Frame count: {len(frame_files)}")
        print(f"[INFO] Ref count: {len(ref_files)}")

        if not frame_files or not ref_files:
            continue

        frame_files = frame_files[:30]

        mode_name = frame_dir.name.replace("frames_", "")

        for frame_path in frame_files:
            frame_img = read_image(frame_path)
            if frame_img is None:
                continue

            for ref_path in ref_files:
                ref_img = read_image(ref_path)
                if ref_img is None:
                    continue

                try:
                    result = call_matcher(frame_img, ref_img, frame_path, ref_path)
                except Exception as e:
                    print(
                        f"[ERR] matcher hata verdi: "
                        f"frame={frame_path.name}, ref={ref_path.name}, err={e}"
                    )
                    continue

                bbox = normalize_bbox(result)

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox

                if x2 <= x1 or y2 <= y1:
                    continue

                label = ref_path.stem
                drawn = draw_bbox(frame_img, bbox, label)

                out_name = f"{mode_name}_{frame_path.stem}__{ref_path.stem}.jpg"
                out_path = OUT_DIR / out_name

                cv2.imwrite(str(out_path), drawn)

                print(f"[MATCH] {out_path}")
                total_matches += 1

    print("\n" + "=" * 80)
    print(f"[DONE] Toplam görselleştirilen match: {total_matches}")
    print(f"[OUT] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()