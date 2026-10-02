from pathlib import Path
import random
import cv2


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "datasets" / "teknofest_v2"
IMAGES_DIR = DATASET_DIR / "train" / "images"
LABELS_DIR = DATASET_DIR / "train" / "labels"

OUT_DIR = ROOT / "outputs" / "v2_label_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

MAX_IMAGES = 24
RANDOM_SEED = 42


def get_images():
    files = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(IMAGES_DIR.glob(extension))

    return sorted(files)


def yolo_to_xyxy(parts, image_width, image_height):
    class_id = int(parts[0])

    center_x = float(parts[1])
    center_y = float(parts[2])
    box_width = float(parts[3])
    box_height = float(parts[4])

    x1 = int((center_x - box_width / 2.0) * image_width)
    y1 = int((center_y - box_height / 2.0) * image_height)
    x2 = int((center_x + box_width / 2.0) * image_width)
    y2 = int((center_y + box_height / 2.0) * image_height)

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return class_id, x1, y1, x2, y2


def main():
    print("[INFO] V2 label görsel kontrolü başlıyor.")

    images = get_images()

    if not images:
        raise FileNotFoundError(
            f"Görsel bulunamadı: {IMAGES_DIR}"
        )

    random.seed(RANDOM_SEED)

    if len(images) > MAX_IMAGES:
        selected_images = random.sample(images, MAX_IMAGES)
    else:
        selected_images = images

    selected_images = sorted(selected_images)

    print(f"[INFO] Toplam train görseli: {len(images)}")
    print(f"[INFO] Kontrol edilecek görsel: {len(selected_images)}")

    saved = 0

    for image_path in selected_images:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Görsel okunamadı: {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        if label_path.exists():
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id, x1, y1, x2, y2 = yolo_to_xyxy(
                    parts,
                    image_width,
                    image_height,
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"UNKNOWN_{class_id}",
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

                label_text = f"{class_id} {class_name}"

                cv2.putText(
                    image,
                    label_text,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        output_path = OUT_DIR / f"{image_path.stem}_labels.jpg"

        cv2.imwrite(str(output_path), image)

        saved += 1
        print(f"[OK] {output_path.name}")

    print("\n" + "=" * 80)
    print(f"[DONE] saved={saved}")
    print(f"[OUT] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    