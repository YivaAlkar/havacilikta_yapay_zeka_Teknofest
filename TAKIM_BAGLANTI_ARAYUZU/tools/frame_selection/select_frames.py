from pathlib import Path
import math
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "rgb": ROOT / "data" / "frames_rgb",
    "thermal": ROOT / "data" / "frames_thermal",
}

OUTPUT_DIR = ROOT / "outputs" / "frame_selection"

THUMB_WIDTH = 320
THUMB_HEIGHT = 200
COLUMNS = 4
ROWS = 4
IMAGES_PER_SHEET = COLUMNS * ROWS

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_images(folder: Path):
    if not folder.exists():
        print(f"[ERROR] Klasör bulunamadı: {folder}")
        return []

    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images)


def read_image(image_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Görsel okunamadı: {image_path}")
        return None

    return image


def create_thumbnail(image, filename: str):
    canvas = np.zeros(
        (THUMB_HEIGHT + 35, THUMB_WIDTH, 3),
        dtype=np.uint8,
    )

    original_height, original_width = image.shape[:2]

    scale = min(
        THUMB_WIDTH / original_width,
        THUMB_HEIGHT / original_height,
    )

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    x_offset = (THUMB_WIDTH - new_width) // 2
    y_offset = (THUMB_HEIGHT - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    cv2.rectangle(
        canvas,
        (0, THUMB_HEIGHT),
        (THUMB_WIDTH - 1, THUMB_HEIGHT + 34),
        (35, 35, 35),
        thickness=-1,
    )

    cv2.putText(
        canvas,
        filename,
        (8, THUMB_HEIGHT + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def create_contact_sheets(source_name: str, source_folder: Path):
    images = get_images(source_folder)

    if not images:
        print(f"[WARNING] Görsel bulunamadı: {source_folder}")
        return

    output_folder = OUTPUT_DIR / source_name
    output_folder.mkdir(parents=True, exist_ok=True)

    sheet_count = math.ceil(len(images) / IMAGES_PER_SHEET)

    print()
    print("=" * 70)
    print(f"[SOURCE] {source_name}")
    print(f"[FOLDER] {source_folder}")
    print(f"[IMAGE COUNT] {len(images)}")
    print(f"[SHEET COUNT] {sheet_count}")
    print("=" * 70)

    sheet_height = ROWS * (THUMB_HEIGHT + 35)
    sheet_width = COLUMNS * THUMB_WIDTH

    for sheet_index in range(sheet_count):
        sheet = np.zeros(
            (sheet_height, sheet_width, 3),
            dtype=np.uint8,
        )

        start_index = sheet_index * IMAGES_PER_SHEET
        end_index = min(
            start_index + IMAGES_PER_SHEET,
            len(images),
        )

        page_images = images[start_index:end_index]

        for local_index, image_path in enumerate(page_images):
            image = read_image(image_path)

            if image is None:
                continue

            thumbnail = create_thumbnail(
                image,
                image_path.name,
            )

            row = local_index // COLUMNS
            column = local_index % COLUMNS

            y1 = row * (THUMB_HEIGHT + 35)
            y2 = y1 + THUMB_HEIGHT + 35

            x1 = column * THUMB_WIDTH
            x2 = x1 + THUMB_WIDTH

            sheet[y1:y2, x1:x2] = thumbnail

        output_path = output_folder / f"{source_name}_sheet_{sheet_index + 1:03d}.jpg"

        success = cv2.imwrite(
            str(output_path),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        if success:
            print(
                f"[OK] {output_path.name} "
                f"({start_index + 1}-{end_index})"
            )
        else:
            print(f"[ERROR] Kaydedilemedi: {output_path}")

    print(f"[DONE] Çıktı klasörü: {output_folder}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[START] Frame contact sheet oluşturuluyor...")
    print(f"[ROOT] {ROOT}")

    for source_name, source_folder in SOURCES.items():
        create_contact_sheets(
            source_name,
            source_folder,
        )

    print()
    print("=" * 70)
    print("[FINISHED]")
    print(f"Contact sheet klasörü: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()