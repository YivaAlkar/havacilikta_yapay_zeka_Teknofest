from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    PROJECT_ROOT
    / "datasets"
    / "teknofest_v4_human_clean_fixed"
)

TARGET = (
    PROJECT_ROOT
    / "datasets"
    / "teknofest_v4_human_clean_detect"
)


def polygon_to_bbox(
    coordinates: list[float],
) -> tuple[float, float, float, float]:
    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        raise ValueError(
            "Polygon en az 3 adet x,y noktası içermelidir."
        )

    xs = coordinates[0::2]
    ys = coordinates[1::2]

    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0.0 or height <= 0.0:
        raise ValueError("Polygon sıfır alanlı bbox üretti.")

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return center_x, center_y, width, height


def convert_label_file(
    path: Path,
) -> Counter:
    counts = Counter()
    output_lines: list[str] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"{path}, satır {line_number}: geçersiz değer."
            ) from exc

        if class_id not in {0, 1, 2, 3}:
            raise ValueError(
                f"{path}, satır {line_number}: "
                f"geçersiz class id={class_id}"
            )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                f"{path}, satır {line_number}: "
                "koordinatlar 0-1 dışında."
            )

        if len(values) == 4:
            center_x, center_y, width, height = values
            counts["bbox_kept"] += 1

        elif len(values) >= 6 and len(values) % 2 == 0:
            center_x, center_y, width, height = (
                polygon_to_bbox(values)
            )
            counts["polygon_converted"] += 1

        else:
            raise ValueError(
                f"{path}, satır {line_number}: "
                f"desteklenmeyen koordinat sayısı={len(values)}"
            )

        output_lines.append(
            f"{class_id} "
            f"{center_x:.8f} "
            f"{center_y:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

        counts[f"class_{class_id}"] += 1

    path.write_text(
        "\n".join(output_lines)
        + ("\n" if output_lines else ""),
        encoding="utf-8",
    )

    return counts


def verify_detection_labels() -> None:
    invalid_lines: list[str] = []

    for split in ("train", "valid", "test"):
        for label_path in sorted(
            (TARGET / split / "labels").glob("*.txt")
        ):
            for line_number, raw_line in enumerate(
                label_path.read_text(
                    encoding="utf-8"
                ).splitlines(),
                start=1,
            ):
                if not raw_line.strip():
                    continue

                if len(raw_line.split()) != 5:
                    invalid_lines.append(
                        f"{label_path}:"
                        f"{line_number}"
                    )

    if invalid_lines:
        raise RuntimeError(
            "5 sütunlu olmayan label kaldı:\n"
            + "\n".join(invalid_lines[:20])
        )


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Kaynak dataset bulunamadı: {SOURCE}"
        )

    if TARGET.exists():
        shutil.rmtree(TARGET)

    print(f"[KOPYALANIYOR] {SOURCE}")
    print(f"[HEDEF]       {TARGET}")

    shutil.copytree(SOURCE, TARGET)

    total_counts = Counter()

    for split in ("train", "valid", "test"):
        split_counts = Counter()
        label_dir = TARGET / split / "labels"

        for label_path in sorted(label_dir.glob("*.txt")):
            split_counts.update(
                convert_label_file(label_path)
            )

        total_counts.update(split_counts)

        print()
        print(f"[{split.upper()}]")
        print(
            f"Korunan bbox      : "
            f"{split_counts['bbox_kept']}"
        )
        print(
            f"Çevrilen polygon  : "
            f"{split_counts['polygon_converted']}"
        )
        print(
            "Sınıflar          : "
            f"0={split_counts['class_0']}, "
            f"1={split_counts['class_1']}, "
            f"2={split_counts['class_2']}, "
            f"3={split_counts['class_3']}"
        )

    verify_detection_labels()

    print()
    print("=" * 72)
    print("V4 DETECTION DATASET HAZIR")
    print("=" * 72)
    print(f"Dataset: {TARGET}")
    print(
        f"Korunan bbox toplam     : "
        f"{total_counts['bbox_kept']}"
    )
    print(
        f"Çevrilen polygon toplam : "
        f"{total_counts['polygon_converted']}"
    )
    print()
    print("Tüm dolu label satırları 5 sütunlu doğrulandı.")
    print("Kaynak dataset değiştirilmedi.")


if __name__ == "__main__":
    main()