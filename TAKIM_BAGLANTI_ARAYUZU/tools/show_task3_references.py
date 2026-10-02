from pathlib import Path
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

REF_DIRS = [
    ROOT / "data" / "references_rgb",
    ROOT / "data" / "references_thermal",
]

OUT_DIR = ROOT / "outputs" / "task3_debug_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_images(folder: Path):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(folder.glob(ext))
    return sorted(files)


def make_tile(img, title, size=(260, 180)):
    tile_w, tile_h = size

    if img is None:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        return tile

    h, w = img.shape[:2]
    scale = min((tile_w - 20) / max(1, w), (tile_h - 45) / max(1, h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile[:] = (35, 35, 35)

    x = (tile_w - nw) // 2
    y = 35 + ((tile_h - 45 - nh) // 2)

    tile[y:y + nh, x:x + nw] = resized

    cv2.putText(
        tile,
        title[:28],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return tile


def main():
    all_tiles = []

    for ref_dir in REF_DIRS:
        refs = get_images(ref_dir)

        for ref_path in refs:
            img = cv2.imread(str(ref_path))
            title = f"{ref_dir.name}/{ref_path.name}"
            tile = make_tile(img, title)
            all_tiles.append(tile)

    if not all_tiles:
        print("[ERR] Referans bulunamadı.")
        return

    cols = 3
    rows = int(np.ceil(len(all_tiles) / cols))

    tile_h, tile_w = all_tiles[0].shape[:2]
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = (20, 20, 20)

    for i, tile in enumerate(all_tiles):
        r = i // cols
        c = i % cols
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

    out_path = OUT_DIR / "all_references_sheet.jpg"
    cv2.imwrite(str(out_path), sheet)

    print(f"[OK] Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()