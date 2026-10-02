from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = PROJECT_ROOT / "datasets" / "teknofest_v4_human_clean"
TARGET = PROJECT_ROOT / "datasets" / "teknofest_v4_human_clean_fixed"

# Roboflow export:
# 0=Insan, 1=Tasit, 2=UAI, 3=UAP
#
# Yarışma:
# 0=Tasit, 1=Insan, 2=UAP, 3=UAI
CLASS_REMAP = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


def remap_label_file(path: Path) -> Counter:
    counts = Counter()
    new_lines = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        # YOLO detection: class + 4 değer = 5
        # YOLO polygon: class + x1 y1 x2 y2 ... = 7 veya daha fazla
        if len(parts) < 5:
            raise ValueError(
                f"{path}, satır {line_number}: "
                f"en az 5 değer bekleniyordu, bulunan={len(parts)}"
            )

        try:
            old_class = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{path}, satır {line_number}: sayısal olmayan değer var."
            ) from exc

        if old_class not in CLASS_REMAP:
            raise ValueError(
                f"{path}, satır {line_number}: "
                f"geçersiz class id={old_class}"
            )

        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(
                f"{path}, satır {line_number}: "
                "normalize koordinat 0-1 aralığı dışında."
            )

        new_class = CLASS_REMAP[old_class]
        parts[0] = str(new_class)

        # Polygon veya bbox koordinatlarının tamamını aynen korur.
        new_lines.append(" ".join(parts))
        counts[new_class] += 1

    path.write_text(
        "\n".join(new_lines) + ("\n" if new_lines else ""),
        encoding="utf-8",
    )

    return counts


def count_files(split: str) -> tuple[int, int]:
    image_dir = TARGET / split / "images"
    label_dir = TARGET / split / "labels"

    image_count = sum(
        1
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    label_count = len(list(label_dir.glob("*.txt")))

    return image_count, label_count


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Kaynak bulunamadı: {SOURCE}")

    if TARGET.exists():
        shutil.rmtree(TARGET)

    print(f"[KOPYALANIYOR] {SOURCE}")
    print(f"[HEDEF]       {TARGET}")

    shutil.copytree(SOURCE, TARGET)

    total_counts = Counter()

    for split in ("train", "valid", "test"):
        label_dir = TARGET / split / "labels"

        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Label klasörü bulunamadı: {label_dir}"
            )

        split_counts = Counter()

        for label_path in sorted(label_dir.glob("*.txt")):
            split_counts.update(remap_label_file(label_path))

        total_counts.update(split_counts)

        image_count, label_count = count_files(split)

        print()
        print(f"[{split.upper()}]")
        print(f"Images : {image_count}")
        print(f"Labels : {label_count}")
        print(f"Objects: {dict(sorted(split_counts.items()))}")

        if image_count != label_count:
            print(
                "UYARI: Görüntü ve label sayısı farklı. "
                "Boş annotation görselleri varsa bu normal olabilir."
            )

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

    yaml_path = TARGET / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print()
    print("=" * 72)
    print("V4 DATASET DÜZELTİLDİ")
    print("=" * 72)
    print(f"Dataset : {TARGET}")
    print(f"YAML    : {yaml_path}")
    print(f"Toplam nesne: {dict(sorted(total_counts.items()))}")
    print()
    print("Beklenen sınıflar:")
    print("0 = Tasit")
    print("1 = Insan")
    print("2 = UAP")
    print("3 = UAI")
    print()
    print("Kaynak dataset değiştirilmedi.")


if __name__ == "__main__":
    main()