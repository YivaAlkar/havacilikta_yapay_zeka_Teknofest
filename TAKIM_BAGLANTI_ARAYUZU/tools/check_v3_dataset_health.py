from pathlib import Path
from collections import Counter
import re


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "teknofest_v3_clean"
REPORT_PATH = ROOT / "outputs" / "v3_dataset_health_report.txt"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VALID_CLASS_IDS = {0, 1, 2, 3}

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


def get_image_files(folder: Path):
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_label_line(line: str, label_path: Path, line_number: int):
    parts = line.strip().split()

    if not parts:
        return None

    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except Exception as exc:
        raise ValueError(
            f"Sayisal donusum hatasi: {label_path}, "
            f"satir={line_number}, icerik={line!r}"
        ) from exc

    if class_id not in VALID_CLASS_IDS:
        raise ValueError(
            f"Gecersiz class id={class_id}: "
            f"{label_path}, satir={line_number}"
        )

    # Normal YOLO bbox:
    # class cx cy width height
    if len(coordinates) == 4:
        label_type = "bbox"

        center_x, center_y, box_width, box_height = coordinates

        if box_width <= 0 or box_height <= 0:
            raise ValueError(
                f"Sifir veya negatif bbox boyutu: "
                f"{label_path}, satir={line_number}"
            )

    # YOLO polygon / segmentation:
    # class x1 y1 x2 y2 x3 y3 ...
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        label_type = "polygon"

        xs = coordinates[0::2]
        ys = coordinates[1::2]

        polygon_width = max(xs) - min(xs)
        polygon_height = max(ys) - min(ys)

        if polygon_width <= 0 or polygon_height <= 0:
            raise ValueError(
                f"Sifir veya negatif polygon alani: "
                f"{label_path}, satir={line_number}"
            )

    else:
        raise ValueError(
            f"Desteklenmeyen label formati: "
            f"{label_path}, satir={line_number}, "
            f"koordinat_sayisi={len(coordinates)}"
        )

    for value in coordinates:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Koordinat 0-1 disinda: value={value}, "
                f"{label_path}, satir={line_number}"
            )

    return class_id, label_type


def main():
    lines = []
    errors = []
    warnings = []

    total_images = 0
    total_label_files = 0
    total_objects = 0
    total_empty_labels = 0

    global_class_counts = Counter()
    global_label_type_counts = Counter()

    lines.append("=" * 80)
    lines.append("TEKNOFEST V3 DATASET SAGLIK RAPORU")
    lines.append("=" * 80)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset bulunamadi: {DATASET}")

    for split in SPLITS:
        images_dir = DATASET / split / "images"
        labels_dir = DATASET / split / "labels"

        if not images_dir.exists():
            errors.append(f"Eksik images klasoru: {images_dir}")
            continue

        if not labels_dir.exists():
            errors.append(f"Eksik labels klasoru: {labels_dir}")
            continue

        images = get_image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt"))

        total_images += len(images)
        total_label_files += len(labels)

        image_by_stem = {image.stem: image for image in images}
        label_by_stem = {label.stem: label for label in labels}

        missing_labels = sorted(
            set(image_by_stem) - set(label_by_stem)
        )
        orphan_labels = sorted(
            set(label_by_stem) - set(image_by_stem)
        )

        split_class_counts = Counter()
        split_label_type_counts = Counter()
        split_objects = 0
        split_empty_labels = 0

        for stem in missing_labels:
            warnings.append(
                f"[{split}] Görsel var, label yok: "
                f"{image_by_stem[stem].name}"
            )

        for stem in orphan_labels:
            errors.append(
                f"[{split}] Label var, görsel yok: "
                f"{label_by_stem[stem].name}"
            )

        for label_path in labels:
            raw_lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            non_empty_lines = [
                line for line in raw_lines if line.strip()
            ]

            if not non_empty_lines:
                split_empty_labels += 1
                total_empty_labels += 1
                continue

            for line_number, line in enumerate(raw_lines, start=1):
                if not line.strip():
                    continue

                try:
                    result = parse_label_line(
                        line,
                        label_path,
                        line_number,
                    )

                    if result is None:
                        continue

                    class_id, label_type = result

                    split_class_counts[class_id] += 1
                    split_label_type_counts[label_type] += 1

                    global_class_counts[class_id] += 1
                    global_label_type_counts[label_type] += 1

                    split_objects += 1
                    total_objects += 1

                except Exception as exc:
                    errors.append(str(exc))

        lines.append("")
        lines.append(f"[{split.upper()}]")
        lines.append(f"Images: {len(images)}")
        lines.append(f"Label files: {len(labels)}")
        lines.append(f"Objects: {split_objects}")
        lines.append(f"Empty label files: {split_empty_labels}")
        lines.append(f"Images without labels: {len(missing_labels)}")
        lines.append(f"Labels without images: {len(orphan_labels)}")
        lines.append(
            f"Label types: {dict(split_label_type_counts)}"
        )

        for class_id in range(4):
            lines.append(
                f"Class {class_id} {CLASS_NAMES[class_id]}: "
                f"{split_class_counts[class_id]}"
            )

    duplicate_report = ROOT / "outputs" / "v3_duplicates.txt"

    if duplicate_report.exists():
        duplicate_lines = [
            line.strip()
            for line in duplicate_report.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if duplicate_lines:
            errors.append(
                f"Splitler arasi duplicate halen var: "
                f"{len(duplicate_lines)} grup"
            )
        else:
            lines.append("")
            lines.append("Cross-split duplicate: 0")
    else:
        warnings.append(
            "v3_duplicates.txt bulunamadi; duplicate kontrolu bilinmiyor."
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("GENEL TOPLAM")
    lines.append("=" * 80)
    lines.append(f"Images: {total_images}")
    lines.append(f"Label files: {total_label_files}")
    lines.append(f"Objects: {total_objects}")
    lines.append(f"Empty label files: {total_empty_labels}")
    lines.append(
        f"Label types: {dict(global_label_type_counts)}"
    )

    for class_id in range(4):
        lines.append(
            f"Class {class_id} {CLASS_NAMES[class_id]}: "
            f"{global_class_counts[class_id]}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("SONUC")
    lines.append("=" * 80)
    lines.append(f"Errors: {len(errors)}")
    lines.append(f"Warnings: {len(warnings)}")

    if errors:
        lines.append("")
        lines.append("[ERRORS]")
        lines.extend(errors)

    if warnings:
        lines.append("")
        lines.append("[WARNINGS]")
        lines.extend(warnings)

    lines.append("")

    if errors:
        lines.append("[FAILED] Dataset egitime hazir degil.")
    else:
        lines.append("[PASSED] Kritik dataset hatasi bulunmadi.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print(f"\n[REPORT] {REPORT_PATH}")


if __name__ == "__main__":
    main()