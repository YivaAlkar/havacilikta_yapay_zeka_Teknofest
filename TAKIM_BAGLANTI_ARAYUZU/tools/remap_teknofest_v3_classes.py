from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "teknofest_v3_clean"
BACKUP = ROOT / "datasets" / "teknofest_v3_clean_before_remap"
MAP = {0: 1, 1: 0, 2: 3, 3: 2}

if not DATASET.exists():
    raise FileNotFoundError(DATASET)

if not BACKUP.exists():
    shutil.copytree(DATASET, BACKUP)
    print(f"[OK] Backup: {BACKUP}")
else:
    print(f"[INFO] Backup zaten var: {BACKUP}")

counts = {0: 0, 1: 0, 2: 0, 3: 0}
file_count = 0

for split in ("train", "valid", "test"):
    labels_dir = DATASET / split / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    for label_path in sorted(labels_dir.glob("*.txt")):
        new_lines = []

        for line_no, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.strip().split()
            if not parts:
                continue

            old_id = int(parts[0])
            if old_id not in MAP:
                raise ValueError(
                    f"Beklenmeyen class id={old_id}: {label_path}, satir={line_no}"
                )

            new_id = MAP[old_id]
            parts[0] = str(new_id)
            counts[new_id] += 1
            new_lines.append(" ".join(parts))

        text = "\n".join(new_lines)
        if new_lines:
            text += "\n"

        label_path.write_text(text, encoding="utf-8")
        file_count += 1

dataset_path = DATASET.resolve().as_posix()
yaml_text = f"""path: {dataset_path}

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

(DATASET / "data.yaml").write_text(yaml_text, encoding="utf-8")

print(f"[OK] Label files remapped: {file_count}")
print(f"[COUNT] 0 Tasit: {counts[0]}")
print(f"[COUNT] 1 Insan: {counts[1]}")
print(f"[COUNT] 2 UAP: {counts[2]}")
print(f"[COUNT] 3 UAI: {counts[3]}")
print("[DONE] data.yaml ve class ID remap tamamlandi.")
