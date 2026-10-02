from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "V1_SAFE": PROJECT_ROOT / "models" / "best.pt",
    "FRIEND_YOLO11S_2CLASS": (
        PROJECT_ROOT / "models" / "best_friend_yolo11s_2class.pt"
    ),
}

IMAGE_SIZES = [640, 768, 960, 1024]
WARMUP_RUNS = 3
TIMED_RUNS = 10

EXPECTED_V1_CLASSES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

EXPECTED_FRIEND_CLASSES = {
    0: "tasit",
    1: "insan",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in names.items()
        }

    if isinstance(names, list):
        return {
            index: str(class_name)
            for index, class_name in enumerate(names)
        }

    return {}


def normalize_text(value: str) -> str:
    translation_table = str.maketrans(
        {
            "ı": "i",
            "İ": "i",
            "ş": "s",
            "Ş": "s",
            "ğ": "g",
            "Ğ": "g",
            "ü": "u",
            "Ü": "u",
            "ö": "o",
            "Ö": "o",
            "ç": "c",
            "Ç": "c",
        }
    )

    return value.translate(translation_table).strip().lower()


def class_order_matches(
    actual_names: dict[int, str],
    expected_names: dict[int, str],
) -> bool:
    if set(actual_names) != set(expected_names):
        return False

    for class_id, expected_name in expected_names.items():
        actual_name = actual_names[class_id]

        if normalize_text(actual_name) != normalize_text(expected_name):
            return False

    return True


def get_parameter_count(model: YOLO) -> int | None:
    inner_model = getattr(model, "model", None)

    if inner_model is None:
        return None

    try:
        return sum(parameter.numel() for parameter in inner_model.parameters())
    except Exception:
        return None


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_test_image() -> np.ndarray:
    """
    1920x1080 boyutunda sabit, tekrar üretilebilir sentetik görüntü.

    Bu görüntü doğruluk ölçmez.
    Yalnızca modellerin aynı girdi üzerindeki hızını ve kararlılığını ölçer.
    """
    rng = np.random.default_rng(seed=20260713)

    image = rng.integers(
        low=0,
        high=256,
        size=(1080, 1920, 3),
        dtype=np.uint8,
    )

    return image


def benchmark_model(
    model: YOLO,
    test_image: np.ndarray,
    image_size: int,
    device: str | int,
) -> dict[str, Any]:
    errors: list[str] = []
    times_ms: list[float] = []
    detection_counts: list[int] = []

    try:
        for _ in range(WARMUP_RUNS):
            model.predict(
                source=test_image,
                imgsz=image_size,
                conf=0.25,
                device=device,
                verbose=False,
            )

        synchronize_cuda()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(TIMED_RUNS):
            synchronize_cuda()
            start_time = time.perf_counter()

            results = model.predict(
                source=test_image,
                imgsz=image_size,
                conf=0.25,
                device=device,
                verbose=False,
            )

            synchronize_cuda()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            times_ms.append(elapsed_ms)

            if results and results[0].boxes is not None:
                detection_counts.append(len(results[0].boxes))
            else:
                detection_counts.append(0)

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    if not times_ms:
        return {
            "imgsz": image_size,
            "success": False,
            "runs": 0,
            "average_ms": None,
            "median_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
            "p95_ms": None,
            "estimated_2250_frames_minutes": None,
            "peak_gpu_memory_mb": None,
            "detection_counts": detection_counts,
            "errors": errors,
        }

    ordered_times = sorted(times_ms)
    p95_index = max(
        0,
        min(
            len(ordered_times) - 1,
            round(0.95 * (len(ordered_times) - 1)),
        ),
    )

    average_ms = statistics.mean(times_ms)

    peak_gpu_memory_mb = None

    if torch.cuda.is_available():
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )

    return {
        "imgsz": image_size,
        "success": True,
        "runs": len(times_ms),
        "average_ms": round(average_ms, 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "minimum_ms": round(min(times_ms), 3),
        "maximum_ms": round(max(times_ms), 3),
        "p95_ms": round(ordered_times[p95_index], 3),
        "estimated_2250_frames_minutes": round(
            average_ms * 2250 / 1000 / 60,
            3,
        ),
        "peak_gpu_memory_mb": (
            round(peak_gpu_memory_mb, 2)
            if peak_gpu_memory_mb is not None
            else None
        ),
        "detection_counts": detection_counts,
        "errors": errors,
    }


def inspect_model(
    label: str,
    model_path: Path,
    test_image: np.ndarray,
    device: str | int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "label": label,
        "path": str(model_path.relative_to(PROJECT_ROOT)),
        "exists": model_path.is_file(),
        "size_mb": None,
        "sha256": None,
        "load_success": False,
        "names": None,
        "class_count": None,
        "class_order_valid": None,
        "parameter_count": None,
        "benchmarks": [],
        "error": None,
    }

    if not model_path.is_file():
        report["error"] = "Model file not found."
        return report

    report["size_mb"] = round(
        model_path.stat().st_size / (1024 * 1024),
        2,
    )
    report["sha256"] = sha256_file(model_path)

    try:
        model = YOLO(str(model_path))
        names = normalize_names(model.names)

        report["load_success"] = True
        report["names"] = names
        report["class_count"] = len(names)
        report["parameter_count"] = get_parameter_count(model)

        if label == "V1_SAFE":
            report["class_order_valid"] = class_order_matches(
                names,
                EXPECTED_V1_CLASSES,
            )
        else:
            report["class_order_valid"] = class_order_matches(
                names,
                EXPECTED_FRIEND_CLASSES,
            )

        for image_size in IMAGE_SIZES:
            print(
                f"[BENCHMARK] {label} | imgsz={image_size}",
                flush=True,
            )

            benchmark = benchmark_model(
                model=model,
                test_image=test_image,
                image_size=image_size,
                device=device,
            )

            report["benchmarks"].append(benchmark)

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def print_model_summary(report: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(report["label"])
    print("=" * 78)
    print(f"Path             : {report['path']}")
    print(f"Exists           : {report['exists']}")
    print(f"Size MB          : {report['size_mb']}")
    print(f"Load success     : {report['load_success']}")
    print(f"Names            : {report['names']}")
    print(f"Class count      : {report['class_count']}")
    print(f"Class order valid: {report['class_order_valid']}")
    print(f"Parameters       : {report['parameter_count']}")
    print(f"SHA256           : {report['sha256']}")
    print(f"Error            : {report['error']}")

    if report["benchmarks"]:
        print()
        print(
            f"{'imgsz':<8}"
            f"{'avg ms':<12}"
            f"{'p95 ms':<12}"
            f"{'2250 dk':<12}"
            f"{'VRAM MB':<12}"
            f"{'başarı'}"
        )

        for benchmark in report["benchmarks"]:
            print(
                f"{str(benchmark['imgsz']):<8}"
                f"{str(benchmark['average_ms']):<12}"
                f"{str(benchmark['p95_ms']):<12}"
                f"{str(benchmark['estimated_2250_frames_minutes']):<12}"
                f"{str(benchmark['peak_gpu_memory_mb']):<12}"
                f"{benchmark['success']}"
            )


def main() -> None:
    output_directory = PROJECT_ROOT / "outputs" / "friend_model_analysis"
    output_directory.mkdir(parents=True, exist_ok=True)

    device: str | int = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("FRIEND MODEL TECHNICAL INSPECTION")
    print("=" * 78)
    print(f"Python      : {torch.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")
    print(
        "GPU         : "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
    )
    print(f"Device      : {device}")
    print()
    print("UYARI:")
    print("- Bu test doğruluk ölçmez.")
    print("- Sentetik görüntü üzerinde hız ve teknik uyumluluk ölçer.")
    print("- main.py çalıştırılmaz.")
    print("- Sunucu bağlantısı kurulmaz.")
    print("- models/best.pt değiştirilmez.")
    print()

    test_image = create_test_image()

    reports: list[dict[str, Any]] = []

    for label, model_path in MODELS.items():
        report = inspect_model(
            label=label,
            model_path=model_path,
            test_image=test_image,
            device=device,
        )
        reports.append(report)
        print_model_summary(report)

    final_report = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "test_image_shape": list(test_image.shape),
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "models": reports,
        "important_note": (
            "This benchmark measures technical compatibility and inference "
            "speed only. It does not measure object detection accuracy."
        ),
    }

    json_path = output_directory / "friend_model_technical_report.json"

    json_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("TEST COMPLETED")
    print("=" * 78)
    print(f"Report: {json_path}")
    print()
    print("Ana model değiştirilmedi.")
    print("main.py çalıştırılmadı.")
    print("Sunucu bağlantısı kurulmadı.")


if __name__ == "__main__":
    main()