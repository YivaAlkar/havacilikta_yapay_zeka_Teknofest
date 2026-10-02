from pathlib import Path
import sys
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAME_PATH = ROOT / "data" / "frames_thermal" / "frame_001110.jpg"
REF_DIR = ROOT / "data" / "references_thermal"

OUT_DIR = ROOT / "outputs" / "task3_debug_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_images(folder: Path):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(folder.glob(ext))
    return sorted(files)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Hem normal görüntü hem edge görüntüsü kullanacağız
    edges = cv2.Canny(gray, 40, 120)

    return gray, edges


def draw_box(img, bbox, label):
    x1, y1, x2, y2 = bbox
    out = img.copy()

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        out,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def template_candidates(frame_img, ref_img, top_k=8):
    frame_gray, frame_edges = preprocess(frame_img)
    ref_gray, ref_edges = preprocess(ref_img)

    fh, fw = frame_gray.shape[:2]
    rh0, rw0 = ref_gray.shape[:2]

    candidates = []

    # Referans nesne ölçüsü frame içinde değişmiş olabilir.
    # O yüzden farklı ölçekleri deniyoruz.
    scales = np.linspace(0.35, 2.50, 44)

    for scale in scales:
        rw = int(rw0 * scale)
        rh = int(rh0 * scale)

        if rw < 8 or rh < 8:
            continue

        if rw >= fw or rh >= fh:
            continue

        ref_gray_s = cv2.resize(ref_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        ref_edges_s = cv2.resize(ref_edges, (rw, rh), interpolation=cv2.INTER_AREA)

        try:
            res_gray = cv2.matchTemplate(frame_gray, ref_gray_s, cv2.TM_CCOEFF_NORMED)
            res_edge = cv2.matchTemplate(frame_edges, ref_edges_s, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue

        # İki skoru karıştırıyoruz.
        res = (0.55 * res_gray) + (0.45 * res_edge)

        for _ in range(top_k):
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            x1, y1 = max_loc
            x2, y2 = x1 + rw, y1 + rh

            candidates.append({
                "score": float(max_val),
                "scale": float(scale),
                "bbox": (x1, y1, x2, y2),
            })

            # Aynı yer tekrar gelmesin diye bölgeyi bastır
            sx1 = max(0, x1 - rw // 2)
            sy1 = max(0, y1 - rh // 2)
            sx2 = min(res.shape[1], x1 + rw // 2)
            sy2 = min(res.shape[0], y1 + rh // 2)
            res[sy1:sy2, sx1:sx2] = -1

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Çok benzer bbox'ları azalt
    filtered = []
    for cand in candidates:
        x1, y1, x2, y2 = cand["bbox"]
        keep = True

        for old in filtered:
            ox1, oy1, ox2, oy2 = old["bbox"]

            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = max(1, (x2 - x1) * (y2 - y1))
            area2 = max(1, (ox2 - ox1) * (oy2 - oy1))
            union = area1 + area2 - inter
            iou = inter / union

            if iou > 0.35:
                keep = False
                break

        if keep:
            filtered.append(cand)

        if len(filtered) >= 10:
            break

    return filtered


def main():
    frame_img = cv2.imread(str(FRAME_PATH))

    if frame_img is None:
        print(f"[ERR] Frame okunamadı: {FRAME_PATH}")
        return

    refs = get_images(REF_DIR)

    print(f"[FRAME] {FRAME_PATH}")
    print(f"[REF_COUNT] {len(refs)}")
    print(f"[OUT] {OUT_DIR}")

    for ref_path in refs:
        ref_img = cv2.imread(str(ref_path))

        if ref_img is None:
            print(f"[WARN] Ref okunamadı: {ref_path}")
            continue

        print("\n" + "=" * 80)
        print(f"[REF] {ref_path.name} shape={ref_img.shape}")

        candidates = template_candidates(frame_img, ref_img, top_k=5)

        for idx, cand in enumerate(candidates):
            bbox = cand["bbox"]
            score = cand["score"]
            scale = cand["scale"]

            print(f"[CAND {idx}] score={score:.4f} scale={scale:.2f} bbox={bbox}")

            drawn = draw_box(
                frame_img,
                bbox,
                f"{ref_path.stem} #{idx} {score:.2f}",
            )

            out_path = OUT_DIR / f"{ref_path.stem}_cand_{idx}_score_{score:.3f}.jpg"
            cv2.imwrite(str(out_path), drawn)

    print("\n[DONE]")


if __name__ == "__main__":
    main()