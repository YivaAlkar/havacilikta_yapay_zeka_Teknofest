from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "teknofest_v3_clean"
SPLITS = ("train", "valid", "test")


def polygon_to_box(values):
    xs = values[0::2]
    ys = values[1::2]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return center_x, center_y, width, height


def main():
    converted_objects = 0
    bbox_objects = 0
    label_files = 0

    for split in SPLITS:
        labels_dir = DATASET / split / "labels"

        for label_path in sorted(labels_dir.glob("*.txt")):
            output_lines = []

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]

                # Normal detection etiketi:
                # class cx cy width height
                if len(values) == 4:
                    center_x, center_y, width, height = values
                    bbox_objects += 1

                # Polygon etiketi:
                # class x1 y1 x2 y2 x3 y3 ...
                elif len(values) >= 6 and len(values) % 2 == 0:
                    center_x, center_y, width, height = polygon_to_box(values)
                    converted_objects += 1

                else:
                    raise ValueError(
                        f"Desteklenmeyen format: {label_path}, "
                        f"satir={line_number}, koordinat={len(values)}"
                    )

                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Gecersiz kutu: {label_path}, satir={line_number}"
                    )

                output_lines.append(
                    f"{class_id} "
                    f"{center_x:.8f} "
                    f"{center_y:.8f} "
                    f"{width:.8f} "
                    f"{height:.8f}"
                )

            output_text = "\n".join(output_lines)

            if output_lines:
                output_text += "\n"

            label_path.write_text(output_text, encoding="utf-8")
            label_files += 1

    print("=" * 80)
    print(f"[DONE] label_files={label_files}")
    print(f"[DONE] existing_bbox_objects={bbox_objects}")
    print(f"[DONE] converted_polygon_objects={converted_objects}")
    print("=" * 80)


if __name__ == "__main__":
    main()