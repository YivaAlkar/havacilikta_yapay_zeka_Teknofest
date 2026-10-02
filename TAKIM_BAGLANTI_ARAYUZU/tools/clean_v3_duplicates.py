from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "teknofest_v3_clean"

SPLIT_PRIORITY = {
    "train": 0,
    "valid": 1,
    "test": 2,
}


def get_frame_name(path: Path):
    match = re.match(r"(frame_\d+)", path.name)

    if match:
        return match.group(1)

    return None


def find_label(image_path: Path):
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def main():
    groups = {}

    for split in ("train", "valid", "test"):
        images_dir = DATASET / split / "images"

        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            frame_name = get_frame_name(image_path)

            if frame_name is None:
                continue

            groups.setdefault(frame_name, []).append(
                {
                    "split": split,
                    "image": image_path,
                }
            )

    removed_images = 0
    removed_labels = 0
    duplicate_groups = 0

    for frame_name, items in sorted(groups.items()):
        present_splits = {item["split"] for item in items}

        if len(present_splits) <= 1:
            continue

        duplicate_groups += 1

        # test > valid > train önceliği
        keep_split = max(
            present_splits,
            key=lambda split: SPLIT_PRIORITY[split],
        )

        print(f"\n[DUPLICATE] {frame_name}")
        print(f"[KEEP SPLIT] {keep_split}")

        for item in items:
            split = item["split"]
            image_path = item["image"]

            if split == keep_split:
                print(f"[KEEP] {split}: {image_path.name}")
                continue

            label_path = find_label(image_path)

            if image_path.exists():
                image_path.unlink()
                removed_images += 1
                print(f"[DELETE IMAGE] {split}: {image_path.name}")

            if label_path.exists():
                label_path.unlink()
                removed_labels += 1
                print(f"[DELETE LABEL] {split}: {label_path.name}")

    print("\n" + "=" * 80)
    print(f"[DONE] duplicate_groups={duplicate_groups}")
    print(f"[DONE] removed_images={removed_images}")
    print(f"[DONE] removed_labels={removed_labels}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    