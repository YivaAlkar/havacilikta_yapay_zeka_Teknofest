from pathlib import Path
import csv
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "data" / "frames_rgb"
MODEL_PATH = ROOT / "models" / "best.pt"
OUT_ROOT = ROOT / "data" / "label_candidates_v2_auto"
REPORT_PATH = OUT_ROOT / "selection_report.csv"

CLASS_NAMES = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}
INFERENCE_CONF = 0.20
LOW_CONF_LIMIT = 0.55
VERY_LOW_CONF_LIMIT = 0.35

LIMITS = {
    "empty": 60,
    "low_conf": 100,
    "crowded": 80,
    "human": 100,
    "uap_uai": 100,
    "suspicious_vehicle": 80,
}


def get_images(folder: Path):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(folder.glob(ext))
    return sorted(files)


def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def bbox_metrics(box, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area_ratio = (w * h) / max(1.0, float(frame_w * frame_h))
    aspect_ratio = w / max(1.0, h)
    return w, h, area_ratio, aspect_ratio


def main():
    print("[INFO] Dataset v2 otomatik aday seçimi başlıyor.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    frames = get_images(FRAME_DIR)
    if not frames:
        raise FileNotFoundError(f"RGB frame bulunamadı: {FRAME_DIR}")

    out_dirs = {
        "empty": OUT_ROOT / "01_empty_or_missed",
        "low_conf": OUT_ROOT / "02_low_confidence",
        "crowded": OUT_ROOT / "03_many_objects",
        "human": OUT_ROOT / "04_human",
        "uap_uai": OUT_ROOT / "05_uap_uai",
        "suspicious_vehicle": OUT_ROOT / "06_suspicious_vehicle",
        "priority": OUT_ROOT / "07_priority_review",
    }
    for folder in out_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    counters = {key: 0 for key in out_dirs}
    rows = []

    for index, frame_path in enumerate(frames, start=1):
        result = model.predict(
            source=str(frame_path), conf=INFERENCE_CONF, iou=0.50, verbose=False
        )[0]

        frame_h, frame_w = result.orig_shape
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        confs = []
        suspicious_vehicle = False
        total_objects = 0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in CLASS_NAMES:
                continue

            _, _, area_ratio, aspect_ratio = bbox_metrics(box, frame_w, frame_h)
            total_objects += 1
            class_counts[cls_id] += 1
            confs.append(conf)

            if cls_id == 0 and (
                conf < LOW_CONF_LIMIT
                or area_ratio > 0.035
                or aspect_ratio > 4.5
                or aspect_ratio < 0.30
            ):
                suspicious_vehicle = True

        min_conf = min(confs) if confs else None
        max_conf = max(confs) if confs else None
        avg_conf = (sum(confs) / len(confs)) if confs else None
        reasons = []

        if total_objects == 0:
            reasons.append("empty_or_possible_missed")
            if counters["empty"] < LIMITS["empty"]:
                safe_copy(frame_path, out_dirs["empty"])
                counters["empty"] += 1

        if confs and min_conf < LOW_CONF_LIMIT:
            reasons.append("low_confidence")
            if counters["low_conf"] < LIMITS["low_conf"]:
                safe_copy(frame_path, out_dirs["low_conf"])
                counters["low_conf"] += 1

        if total_objects >= 4:
            reasons.append("many_objects")
            if counters["crowded"] < LIMITS["crowded"]:
                safe_copy(frame_path, out_dirs["crowded"])
                counters["crowded"] += 1

        if class_counts[1] > 0:
            reasons.append("human")
            if counters["human"] < LIMITS["human"]:
                safe_copy(frame_path, out_dirs["human"])
                counters["human"] += 1

        if class_counts[2] > 0 or class_counts[3] > 0:
            reasons.append("uap_uai")
            if counters["uap_uai"] < LIMITS["uap_uai"]:
                safe_copy(frame_path, out_dirs["uap_uai"])
                counters["uap_uai"] += 1

        if suspicious_vehicle:
            reasons.append("suspicious_vehicle")
            if counters["suspicious_vehicle"] < LIMITS["suspicious_vehicle"]:
                safe_copy(frame_path, out_dirs["suspicious_vehicle"])
                counters["suspicious_vehicle"] += 1

        priority = (
            total_objects == 0
            or (min_conf is not None and min_conf < VERY_LOW_CONF_LIMIT)
            or class_counts[1] > 0
            or class_counts[2] > 0
            or class_counts[3] > 0
            or suspicious_vehicle
        )
        if priority:
            reasons.append("priority_review")
            safe_copy(frame_path, out_dirs["priority"])
            counters["priority"] += 1

        rows.append({
            "frame": frame_path.name,
            "total_objects": total_objects,
            "tasit": class_counts[0],
            "insan": class_counts[1],
            "uap": class_counts[2],
            "uai": class_counts[3],
            "min_conf": "" if min_conf is None else f"{min_conf:.4f}",
            "avg_conf": "" if avg_conf is None else f"{avg_conf:.4f}",
            "max_conf": "" if max_conf is None else f"{max_conf:.4f}",
            "reasons": "|".join(reasons),
        })

        if index % 25 == 0 or index == len(frames):
            print(f"[INFO] {index}/{len(frames)} frame işlendi.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "total_objects", "tasit", "insan", "uap", "uai",
            "min_conf", "avg_conf", "max_conf", "reasons"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("[DONE] Dataset v2 aday seçimi tamamlandı.")
    print(f"[OUT] {OUT_ROOT}")
    print(f"[REPORT] {REPORT_PATH}")
    for key, value in counters.items():
        print(f"[COUNT] {key}={value}")
    print("=" * 80)


if __name__ == "__main__":
    main()