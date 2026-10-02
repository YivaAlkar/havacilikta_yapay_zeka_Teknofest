from pathlib import Path
from collections import Counter
import csv
import time

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

FRAME_DIR = ROOT / "data" / "frames_rgb"

MODEL_PATHS = {
    "v1": ROOT / "models" / "best.pt",
    "v3": ROOT / "models" / "best_v3.pt",
}

OUT_DIR = ROOT / "outputs" / "v1_v3_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

CONFIDENCE = 0.40
IOU = 0.50


def get_frames():
    files = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(FRAME_DIR.glob(extension))

    return sorted(files)


def run_model(model_name, model_path, frames):
    print(f"\n[INFO] Model çalışıyor: {model_name}")
    print(f"[INFO] Path: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    model = YOLO(str(model_path))

    class_counts = Counter()
    detection_frames = 0
    total_objects = 0
    total_inference_ms = 0.0

    rows = []

    for index, frame_path in enumerate(frames, start=1):
        start_time = time.perf_counter()

        result = model.predict(
            source=str(frame_path),
            conf=CONFIDENCE,
            iou=IOU,
            device=0,
            verbose=False,
        )[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        total_inference_ms += elapsed_ms

        frame_counts = Counter()
        frame_confidences = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in CLASS_NAMES:
                continue

            frame_counts[class_id] += 1
            class_counts[class_id] += 1
            frame_confidences.append(confidence)
            total_objects += 1

        if sum(frame_counts.values()) > 0:
            detection_frames += 1

        rows.append(
            {
                "model": model_name,
                "frame": frame_path.name,
                "total": sum(frame_counts.values()),
                "tasit": frame_counts[0],
                "insan": frame_counts[1],
                "uap": frame_counts[2],
                "uai": frame_counts[3],
                "min_conf": (
                    f"{min(frame_confidences):.4f}"
                    if frame_confidences else ""
                ),
                "max_conf": (
                    f"{max(frame_confidences):.4f}"
                    if frame_confidences else ""
                ),
                "inference_ms": f"{elapsed_ms:.3f}",
            }
        )

        if index % 25 == 0 or index == len(frames):
            print(f"[INFO] {model_name}: {index}/{len(frames)}")

    average_ms = total_inference_ms / max(1, len(frames))

    summary = {
        "model": model_name,
        "frames": len(frames),
        "detection_frames": detection_frames,
        "empty_frames": len(frames) - detection_frames,
        "total_objects": total_objects,
        "tasit": class_counts[0],
        "insan": class_counts[1],
        "uap": class_counts[2],
        "uai": class_counts[3],
        "average_inference_ms": average_ms,
    }

    return summary, rows


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    frames = get_frames()

    if not frames:
        raise FileNotFoundError(f"Frame bulunamadı: {FRAME_DIR}")

    print(f"[INFO] Toplam frame: {len(frames)}")

    summaries = []
    all_rows = []

    for model_name, model_path in MODEL_PATHS.items():
        summary, rows = run_model(
            model_name,
            model_path,
            frames,
        )

        summaries.append(summary)
        all_rows.extend(rows)

    detail_path = OUT_DIR / "frame_by_frame.csv"
    summary_path = OUT_DIR / "summary.csv"

    write_csv(detail_path, all_rows)
    write_csv(summary_path, summaries)

    print("\n" + "=" * 90)
    print("V1 - V3 KARŞILAŞTIRMA")
    print("=" * 90)

    for summary in summaries:
        print(f"\nMODEL: {summary['model']}")
        print(f"frames: {summary['frames']}")
        print(f"detection_frames: {summary['detection_frames']}")
        print(f"empty_frames: {summary['empty_frames']}")
        print(f"total_objects: {summary['total_objects']}")
        print(f"Tasit: {summary['tasit']}")
        print(f"Insan: {summary['insan']}")
        print(f"UAP: {summary['uap']}")
        print(f"UAI: {summary['uai']}")
        print(
            "average_inference_ms: "
            f"{summary['average_inference_ms']:.3f}"
        )

    print("\n" + "=" * 90)
    print(f"[OUT] {OUT_DIR}")
    print(f"[CSV] {summary_path}")
    print(f"[CSV] {detail_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()