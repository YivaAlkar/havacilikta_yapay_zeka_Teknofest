from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "teknofest_v3_clean"
OUT_PATH = ROOT / "outputs" / "v3_duplicates.txt"


def main():
    groups = {}

    for split in ("train", "valid", "test"):
        images_dir = DATASET / split / "images"

        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            match = re.match(r"(frame_\d+)", image_path.name)

            if not match:
                continue

            frame_name = match.group(1)

            groups.setdefault(frame_name, []).append(
                (split, image_path.name)
            )

    duplicates = []

    for frame_name, items in sorted(groups.items()):
        splits = {split for split, _ in items}

        if len(splits) > 1:
            details = ", ".join(
                f"{split}:{filename}"
                for split, filename in items
            )

            duplicates.append(
                f"{frame_name} -> {details}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUT_PATH.write_text(
        "\n".join(duplicates),
        encoding="utf-8",
    )

    print(f"[DONE] duplicate groups: {len(duplicates)}")
    print(f"[OUT] {OUT_PATH}")


if __name__ == "__main__":
    main()