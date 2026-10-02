from pathlib import Path
import sys
from collections import Counter

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEST_PATH = ROOT / "models" / "best.pt"
FRAME_DIR = ROOT / "data" / "frames_rgb"

EXPECTED_MAPPING = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


def get_frames():
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(FRAME_DIR.glob(ext))
    return sorted(files)


def main():
    print("=" * 80)
    print("[CHECK] Custom model mapping kontrolü")
    print("=" * 80)

    print(f"[BEST_PATH] {BEST_PATH}")
    print(f"[BEST_EXISTS] {BEST_PATH.exists()}")

    if not BEST_PATH.exists():
        print("\n[STOP] models/best.pt yok.")
        print("Model eğitildikten sonra dosyayı şuraya koy:")
        print(BEST_PATH)
        print("\nSonra tekrar çalıştır:")
        print("python tools\\check_custom_model_mapping.py")
        return

    model = YOLO(str(BEST_PATH))

    print("\n[MODEL NAMES]")
    print(model.names)

    print("\n[EXPECTED COMPETITION MAPPING]")
    for cls_id, name in EXPECTED_MAPPING.items():
        print(f"{cls_id} -> {name}")

    model_class_ids = set(model.names.keys()) if isinstance(model.names, dict) else set(range(len(model.names)))

    print("\n[CLASS ID CHECK]")
    for cls_id in EXPECTED_MAPPING:
        if cls_id in model_class_ids:
            print(f"[OK] class id {cls_id} mevcut")
        else:
            print(f"[WARN] class id {cls_id} model.names içinde yok")

    invalid_ids = sorted([x for x in model_class_ids if x not in EXPECTED_MAPPING])
    if invalid_ids:
        print(f"[WARN] 0/1/2/3 dışında class id var: {invalid_ids}")
    else:
        print("[OK] Sadece 0/1/2/3 class id var gibi görünüyor.")

    frames = get_frames()
    print(f"\n[FRAME_COUNT] {len(frames)}")

    if not frames:
        print("[STOP] data/frames_rgb içinde frame yok.")
        return

    selected = frames[::30]
    print(f"[SELECTED_COUNT] {len(selected)}")

    class_counter = Counter()
    conf_values = []

    print("\n[PREDICTION SAMPLE]")
    for frame_path in selected:
        results = model.predict(
            source=str(frame_path),
            conf=0.25,
            iou=0.50,
            verbose=False,
        )[0]

        frame_counts = Counter()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_counter[cls_id] += 1
            frame_counts[cls_id] += 1
            conf_values.append(conf)

        if frame_counts:
            readable = {
                EXPECTED_MAPPING.get(k, f"UNKNOWN_{k}"): v
                for k, v in frame_counts.items()
            }
            print(f"{frame_path.name}: {readable}")
        else:
            print(f"{frame_path.name}: no detection")

    print("\n[SUMMARY]")
    if class_counter:
        for cls_id, count in sorted(class_counter.items()):
            name = EXPECTED_MAPPING.get(cls_id, f"UNKNOWN_{cls_id}")
            print(f"class {cls_id} ({name}): {count}")

        avg_conf = sum(conf_values) / max(1, len(conf_values))
        print(f"avg_conf: {avg_conf:.3f}")
        print(f"min_conf: {min(conf_values):.3f}")
        print(f"max_conf: {max(conf_values):.3f}")
    else:
        print("Hiç detection çıkmadı.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()

