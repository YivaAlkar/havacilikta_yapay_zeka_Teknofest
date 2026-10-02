from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "data" / "label_candidates_v2_auto"
OUT_ROOT = ROOT / "outputs" / "v2_candidate_contact_sheets"

FOLDERS = [
    "01_empty_or_missed",
    "04_human",
    "05_uap_uai",
    "06_suspicious_vehicle",
    "07_priority_review",
]

THUMB_W = 360
THUMB_H = 230
LABEL_H = 34
COLS = 3
ROWS = 4
PER_PAGE = COLS * ROWS
MARGIN = 16


def get_images(folder: Path):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(folder.glob(ext))
    return sorted(files)


def fit_image(img: Image.Image, width: int, height: int):
    img = img.convert("RGB")
    img.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "black")
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_pages(folder_name: str):
    src_dir = SRC_ROOT / folder_name
    files = get_images(src_dir)

    if not files:
        print(f"[WARN] Görsel yok: {src_dir}")
        return 0

    out_dir = OUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    page_w = MARGIN + COLS * (THUMB_W + MARGIN)
    page_h = MARGIN + ROWS * (THUMB_H + LABEL_H + MARGIN)

    total_pages = math.ceil(len(files) / PER_PAGE)

    for page_idx in range(total_pages):
        page_files = files[page_idx * PER_PAGE:(page_idx + 1) * PER_PAGE]
        sheet = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(sheet)

        for i, path in enumerate(page_files):
            row = i // COLS
            col = i % COLS

            x = MARGIN + col * (THUMB_W + MARGIN)
            y = MARGIN + row * (THUMB_H + LABEL_H + MARGIN)

            try:
                with Image.open(path) as img:
                    thumb = fit_image(img, THUMB_W, THUMB_H)
                sheet.paste(thumb, (x, y))
            except Exception as e:
                draw.rectangle((x, y, x + THUMB_W, y + THUMB_H), outline="red", width=3)
                draw.text((x + 8, y + 8), f"ERR: {e}", fill="red")

            label = path.stem
            draw.text((x, y + THUMB_H + 7), label, fill="black")

        out_path = out_dir / f"{folder_name}_page_{page_idx + 1:02d}.jpg"
        sheet.save(out_path, quality=92)

    print(f"[OK] {folder_name}: images={len(files)} pages={total_pages}")
    return total_pages


def main():
    print("[INFO] V2 aday contact sheet üretimi başlıyor.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    for folder_name in FOLDERS:
        total_pages += make_pages(folder_name)

    print("\n" + "=" * 80)
    print(f"[DONE] total_pages={total_pages}")
    print(f"[OUT] {OUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    