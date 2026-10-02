from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "teknofest_v1"
YAML_PATH = DATASET_DIR / "data.yaml"
BACKUP_DIR = DATASET_DIR / "_backup_before_class_remap"
MARKER_PATH = DATASET_DIR / ".class_remap_completed"

# Roboflow'un eski sırası:
# 0 Insan, 1 Tasit, 2 UAI, 3 UAP
#
# Yarışma için yeni sıra:
# 0 Tasit, 1 Insan, 2 UAP, 3 UAI
CLASS_REMAP = {
    0: 1,  # Insan -> Insan
    1: 0,  # Tasit -> Tasit
    2: 3,  # UAI -> UAI
    3: 2,  # UAP -> UAP
}

SPLITS = ("train", "valid", "test")


def backup_files() -> None:
    if BACKUP_DIR.exists():
        print(f"[INFO] Backup zaten var: {BACKUP_DIR}")
        return

    print(f"[BACKUP] {BACKUP_DIR}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(YAML_PATH, BACKUP_DIR / "data.yaml")

    for split in SPLITS:
        source = DATASET_DIR / split / "labels"
        destination = BACKUP_DIR / split / "labels"

        if source.exists():
            shutil.copytree(source, destination)


def remap_label_file(label_path: Path) -> tuple[int, int]:
    original_lines = label_path.read_text(encoding="utf-8").splitlines()
    output_lines = []

    object_count = 0
    changed_count = 0

    for line_number, line in enumerate(original_lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) < 5:
            raise ValueError(
                f"Geçersiz YOLO etiketi: {label_path}, satır {line_number}: {line}"
            )

        try:
            old_class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Sınıf ID tam sayı değil: {label_path}, satır {line_number}"
            ) from exc

        if old_class_id not in CLASS_REMAP:
            raise ValueError(
                f"Beklenmeyen sınıf ID: {old_class_id} "
                f"({label_path}, satır {line_number})"
            )

        new_class_id = CLASS_REMAP[old_class_id]
        parts[0] = str(new_class_id)

        output_lines.append(" ".join(parts))
        object_count += 1

        if new_class_id != old_class_id:
            changed_count += 1

    final_text = "\n".join(output_lines)

    if output_lines:
        final_text += "\n"

    label_path.write_text(final_text, encoding="utf-8")
    return object_count, changed_count


def remap_all_labels() -> None:
    total_files = 0
    total_objects = 0
    total_changed = 0

    for split in SPLITS:
        labels_dir = DATASET_DIR / split / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Label klasörü bulunamadı: {labels_dir}")

        label_files = sorted(labels_dir.glob("*.txt"))

        split_objects = 0
        split_changed = 0

        for label_path in label_files:
            object_count, changed_count = remap_label_file(label_path)
            split_objects += object_count
            split_changed += changed_count

        total_files += len(label_files)
        total_objects += split_objects
        total_changed += split_changed

        print(
            f"[OK] {split}: "
            f"label_files={len(label_files)}, "
            f"objects={split_objects}, "
            f"changed={split_changed}"
        )

    print()
    print("[LABEL SUMMARY]")
    print(f"files={total_files}")
    print(f"objects={total_objects}")
    print(f"changed={total_changed}")


def write_yaml() -> None:
    yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: Tasit
  1: Insan
  2: UAP
  3: UAI
"""

    YAML_PATH.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] data.yaml güncellendi: {YAML_PATH}")


def verify_dataset() -> None:
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    invalid_labels = []

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Görsel klasörü yok: {images_dir}")

        if not labels_dir.exists():
            raise FileNotFoundError(f"Label klasörü yok: {labels_dir}")

        image_count = sum(
            1
            for path in images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        label_files = list(labels_dir.glob("*.txt"))

        for label_path in label_files:
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue

                parts = line.split()

                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:5]]
                except (ValueError, IndexError):
                    invalid_labels.append(
                        f"{label_path}:{line_number} biçim hatası"
                    )
                    continue

                if class_id not in class_counts:
                    invalid_labels.append(
                        f"{label_path}:{line_number} class={class_id}"
                    )
                    continue

                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    invalid_labels.append(
                        f"{label_path}:{line_number} koordinat aralık dışında"
                    )

                class_counts[class_id] += 1

        print(
            f"[VERIFY] {split}: "
            f"images={image_count}, labels={len(label_files)}"
        )

    print()
    print("[CLASS COUNTS]")
    print(f"0 Tasit: {class_counts[0]}")
    print(f"1 Insan: {class_counts[1]}")
    print(f"2 UAP:   {class_counts[2]}")
    print(f"3 UAI:   {class_counts[3]}")

    if invalid_labels:
        print()
        print("[ERROR] Geçersiz etiketler:")
        for item in invalid_labels[:20]:
            print(item)

        raise RuntimeError(
            f"Toplam {len(invalid_labels)} geçersiz etiket bulundu."
        )

    print()
    print("[OK] Dataset doğrulaması başarılı.")


def main() -> None:
    print("=" * 70)
    print("[START] TEKNOFEST dataset sınıf dönüşümü")
    print(f"[DATASET] {DATASET_DIR}")
    print("=" * 70)

    if MARKER_PATH.exists():
        print("[STOP] Bu dataset daha önce dönüştürülmüş.")
        print(f"Marker: {MARKER_PATH}")
        sys.exit(0)

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"data.yaml bulunamadı: {YAML_PATH}")

    backup_files()
    remap_all_labels()
    write_yaml()
    verify_dataset()

    MARKER_PATH.write_text(
        "Class order converted to: 0 Tasit, 1 Insan, 2 UAP, 3 UAI\n",
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("[FINISHED]")
    print("Yeni sınıf sırası:")
    print("0 Tasit")
    print("1 Insan")
    print("2 UAP")
    print("3 UAI")
    print("=" * 70)


if __name__ == "__main__":
    main()