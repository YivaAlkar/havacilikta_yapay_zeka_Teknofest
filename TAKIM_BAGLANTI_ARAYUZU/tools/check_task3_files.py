from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    ROOT / "data" / "frames_rgb",
    ROOT / "data" / "frames_thermal",
    ROOT / "data" / "references_rgb",
    ROOT / "data" / "references_thermal",
]

for folder in FOLDERS:
    print("\n" + "=" * 80)
    print(folder)

    if not folder.exists():
        print("[YOK]")
        continue

    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(folder.glob(ext))

    files = sorted(files)
    print("count:", len(files))

    for path in files[:10]:
        img = cv2.imread(str(path))
        if img is None:
            print("[OKUNAMADI]", path.name)
            continue

        h, w = img.shape[:2]
        mean = img.mean()
        print(f"{path.name} shape={w}x{h} mean={mean:.1f}")