from __future__ import annotations

import csv
import json
import shutil
import statistics
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path(
    r"C:\Users\muham\OneDrive\Masaüstü\HYZ_2025_Ornek_Veriler.zip"
)

VIDEO_MEMBER_NAME = "Ornek-Veri-1-RGB.MP4"

V1_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

FRIEND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_friend_yolo11s_2class.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "v1_friend_2025_comparison"
)

V1_OUTPUT_DIR = OUTPUT_ROOT / "v1"
FRIEND_OUTPUT_DIR = OUTPUT_ROOT / "friend"
SIDE_BY_SIDE_DIR = OUTPUT_ROOT / "side_by_side"
DISAGREEMENT_DIR = OUTPUT_ROOT / "disagreements"

SAMPLE_COUNT = 120

V1_IMGSZ = 640
FRIEND_IMGSZ = 1024

CONFIDENCE = 0.25

V1_ALLOWED_CLASSES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

FRIEND_ALLOWED_CLASSES = {
    0: "Tasit",
    1: "Insan",
}


def ensure_inputs_exist() -> None:
    required_paths = [
        ZIP_PATH,
        V1_MODEL_PATH,
        FRIEND_MODEL_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Eksik dosyalar:\n" + "\n".join(missing)
        )


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in names.items()
        }

    if isinstance(names, list):
        return {
            index: str(name)
            for index, name in enumerate(names)
        }

    return {}


def normalize_class_name(value: str) -> str:
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


def validate_model_classes(
    model_label: str,
    actual_names: dict[int, str],
    expected_names: dict[int, str],
) -> None:
    if set(actual_names) != set(expected_names):
        raise ValueError(
            f"{model_label} class ID'leri yanlış. "
            f"Beklenen={expected_names}, bulunan={actual_names}"
        )

    for class_id, expected_name in expected_names.items():
        actual_name = actual_names[class_id]

        if normalize_class_name(
            actual_name
        ) != normalize_class_name(expected_name):
            raise ValueError(
                f"{model_label} class sırası yanlış. "
                f"ID={class_id}, beklenen={expected_name}, "
                f"bulunan={actual_name}"
            )


def prepare_output_directories() -> None:
    for directory in (
        OUTPUT_ROOT,
        V1_OUTPUT_DIR,
        FRIEND_OUTPUT_DIR,
        SIDE_BY_SIDE_DIR,
        DISAGREEMENT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def extract_video_temporarily(
    archive: zipfile.ZipFile,
    temporary_directory: Path,
) -> Path:
    member_names = archive.namelist()

    matching_members = [
        name
        for name in member_names
        if Path(name).name.lower()
        == VIDEO_MEMBER_NAME.lower()
    ]

    if not matching_members:
        raise FileNotFoundError(
            f"ZIP içinde video bulunamadı: {VIDEO_MEMBER_NAME}"
        )

    archive_member = matching_members[0]
    output_path = temporary_directory / VIDEO_MEMBER_NAME

    print(f"[ÇIKARILIYOR] {archive_member}")
    print("Bu işlem birkaç dakika sürebilir.")

    with archive.open(archive_member, "r") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(
                source,
                target,
                length=8 * 1024 * 1024,
            )

    return output_path


def sample_frame_indexes(
    total_frames: int,
    sample_count: int,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError(
            f"Geçersiz video frame sayısı: {total_frames}"
        )

    actual_count = min(sample_count, total_frames)

    if actual_count == 1:
        return [0]

    indexes = {
        round(
            index * (total_frames - 1)
            / (actual_count - 1)
        )
        for index in range(actual_count)
    }

    return sorted(indexes)


def count_detections(
    result: Any,
    model_names: dict[int, str],
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return counts, detections

    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        coordinates = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        class_name = model_names.get(
            class_id,
            f"class_{class_id}",
        )

        counts[class_name] += 1

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "xyxy": coordinates,
            }
        )

    return counts, detections


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(
    model: YOLO,
    frame: Any,
    image_size: int,
    device: str | int,
) -> tuple[Any, float]:
    synchronize_cuda()
    start_time = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=image_size,
        conf=CONFIDENCE,
        device=device,
        verbose=False,
    )

    synchronize_cuda()
    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    if not results:
        raise RuntimeError(
            "YOLO predict boş sonuç döndürdü."
        )

    return results[0], elapsed_ms


def add_title(
    image: Any,
    title: str,
) -> Any:
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 55),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        output,
        title,
        (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def resize_for_comparison(
    image: Any,
    target_width: int = 960,
) -> Any:
    height, width = image.shape[:2]

    if width == target_width:
        return image

    scale = target_width / width
    target_height = max(1, round(height * scale))

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def normalized_two_class_counts(
    counts: Counter[str],
) -> dict[str, int]:
    result = {
        "tasit": 0,
        "insan": 0,
    }

    for class_name, count in counts.items():
        normalized = normalize_class_name(class_name)

        if normalized == "tasit":
            result["tasit"] += count
        elif normalized == "insan":
            result["insan"] += count

    return result


def models_disagree(
    v1_counts: Counter[str],
    friend_counts: Counter[str],
) -> bool:
    v1_two_class = normalized_two_class_counts(
        v1_counts
    )
    friend_two_class = normalized_two_class_counts(
        friend_counts
    )

    return v1_two_class != friend_two_class


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = round(0.95 * (len(ordered) - 1))
    return ordered[index]


def main() -> None:
    ensure_inputs_exist()
    prepare_output_directories()

    device: str | int = (
        0 if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("V1 / ARKADAŞ MODELİ GERÇEK RGB VİDEO KARŞILAŞTIRMASI")
    print("=" * 80)
    print(f"CUDA         : {torch.cuda.is_available()}")
    print(
        "GPU          : "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
    )
    print(f"Video        : {VIDEO_MEMBER_NAME}")
    print(f"Örnek sayısı : {SAMPLE_COUNT}")
    print()

    print("[MODEL] V1 yükleniyor...")
    v1_model = YOLO(str(V1_MODEL_PATH))
    v1_names = normalize_names(v1_model.names)

    validate_model_classes(
        "V1",
        v1_names,
        V1_ALLOWED_CLASSES,
    )

    print("[MODEL] Arkadaş modeli yükleniyor...")
    friend_model = YOLO(
        str(FRIEND_MODEL_PATH)
    )
    friend_names = normalize_names(
        friend_model.names
    )

    validate_model_classes(
        "FRIEND",
        friend_names,
        FRIEND_ALLOWED_CLASSES,
    )

    report_rows: list[dict[str, Any]] = []
    v1_times: list[float] = []
    friend_times: list[float] = []

    total_v1_counts: Counter[str] = Counter()
    total_friend_counts: Counter[str] = Counter()

    disagreement_count = 0

    with tempfile.TemporaryDirectory(
        prefix="hyz_2025_video_"
    ) as temporary_path:
        temporary_directory = Path(
            temporary_path
        )

        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as archive:
            video_path = extract_video_temporarily(
                archive,
                temporary_directory,
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Video açılamadı: {video_path}"
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        frame_indexes = sample_frame_indexes(
            total_frames,
            SAMPLE_COUNT,
        )

        print()
        print(f"Toplam frame : {total_frames}")
        print(f"FPS          : {fps}")
        print(f"Çözünürlük   : {width}x{height}")
        print(
            f"Seçilen frame: {len(frame_indexes)}"
        )
        print()

        for sample_number, frame_index in enumerate(
            frame_indexes,
            start=1,
        ):
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = capture.read()

            if not success or frame is None:
                print(
                    f"[UYARI] Frame okunamadı: "
                    f"{frame_index}"
                )
                continue

            print(
                f"[{sample_number:03d}/"
                f"{len(frame_indexes):03d}] "
                f"frame={frame_index}",
                flush=True,
            )

            v1_result, v1_elapsed_ms = run_model(
                v1_model,
                frame,
                V1_IMGSZ,
                device,
            )

            friend_result, friend_elapsed_ms = (
                run_model(
                    friend_model,
                    frame,
                    FRIEND_IMGSZ,
                    device,
                )
            )

            v1_counts, v1_detections = (
                count_detections(
                    v1_result,
                    v1_names,
                )
            )

            friend_counts, friend_detections = (
                count_detections(
                    friend_result,
                    friend_names,
                )
            )

            v1_times.append(v1_elapsed_ms)
            friend_times.append(
                friend_elapsed_ms
            )

            total_v1_counts.update(v1_counts)
            total_friend_counts.update(
                friend_counts
            )

            disagrees = models_disagree(
                v1_counts,
                friend_counts,
            )

            if disagrees:
                disagreement_count += 1

            v1_annotated = add_title(
                v1_result.plot(),
                (
                    f"V1 imgsz={V1_IMGSZ} | "
                    f"frame={frame_index} | "
                    f"{dict(v1_counts)}"
                ),
            )

            friend_annotated = add_title(
                friend_result.plot(),
                (
                    f"FRIEND imgsz={FRIEND_IMGSZ} | "
                    f"frame={frame_index} | "
                    f"{dict(friend_counts)}"
                ),
            )

            v1_resized = resize_for_comparison(
                v1_annotated
            )
            friend_resized = (
                resize_for_comparison(
                    friend_annotated
                )
            )

            side_by_side = cv2.hconcat(
                [
                    v1_resized,
                    friend_resized,
                ]
            )

            filename = (
                f"frame_{frame_index:06d}.jpg"
            )

            cv2.imwrite(
                str(V1_OUTPUT_DIR / filename),
                v1_annotated,
            )

            cv2.imwrite(
                str(
                    FRIEND_OUTPUT_DIR / filename
                ),
                friend_annotated,
            )

            cv2.imwrite(
                str(
                    SIDE_BY_SIDE_DIR / filename
                ),
                side_by_side,
            )

            if disagrees:
                cv2.imwrite(
                    str(
                        DISAGREEMENT_DIR
                        / filename
                    ),
                    side_by_side,
                )

            report_rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": (
                        frame_index / fps
                        if fps > 0
                        else None
                    ),
                    "v1_imgsz": V1_IMGSZ,
                    "friend_imgsz": (
                        FRIEND_IMGSZ
                    ),
                    "v1_inference_ms": (
                        v1_elapsed_ms
                    ),
                    "friend_inference_ms": (
                        friend_elapsed_ms
                    ),
                    "v1_counts": dict(v1_counts),
                    "friend_counts": dict(
                        friend_counts
                    ),
                    "v1_detection_count": len(
                        v1_detections
                    ),
                    "friend_detection_count": len(
                        friend_detections
                    ),
                    "models_disagree": disagrees,
                    "v1_detections": (
                        v1_detections
                    ),
                    "friend_detections": (
                        friend_detections
                    ),
                }
            )

        capture.release()

    csv_path = OUTPUT_ROOT / "comparison.csv"
    json_path = OUTPUT_ROOT / "comparison.json"
    summary_path = OUTPUT_ROOT / "summary.txt"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "frame_index",
                "time_seconds",
                "v1_detection_count",
                "friend_detection_count",
                "v1_tasit",
                "v1_insan",
                "friend_tasit",
                "friend_insan",
                "v1_inference_ms",
                "friend_inference_ms",
                "models_disagree",
            ]
        )

        for row in report_rows:
            v1_two = normalized_two_class_counts(
                Counter(row["v1_counts"])
            )
            friend_two = (
                normalized_two_class_counts(
                    Counter(
                        row["friend_counts"]
                    )
                )
            )

            writer.writerow(
                [
                    row["frame_index"],
                    row["time_seconds"],
                    row["v1_detection_count"],
                    row[
                        "friend_detection_count"
                    ],
                    v1_two["tasit"],
                    v1_two["insan"],
                    friend_two["tasit"],
                    friend_two["insan"],
                    row["v1_inference_ms"],
                    row["friend_inference_ms"],
                    row["models_disagree"],
                ]
            )

    final_report = {
        "zip_path": str(ZIP_PATH),
        "video_member": VIDEO_MEMBER_NAME,
        "sample_target": SAMPLE_COUNT,
        "processed_samples": len(report_rows),
        "confidence": CONFIDENCE,
        "v1_model": str(V1_MODEL_PATH),
        "friend_model": str(
            FRIEND_MODEL_PATH
        ),
        "v1_imgsz": V1_IMGSZ,
        "friend_imgsz": FRIEND_IMGSZ,
        "v1_class_names": v1_names,
        "friend_class_names": friend_names,
        "total_v1_counts": dict(
            total_v1_counts
        ),
        "total_friend_counts": dict(
            total_friend_counts
        ),
        "disagreement_frames": (
            disagreement_count
        ),
        "v1_average_ms": (
            statistics.fmean(v1_times)
            if v1_times
            else None
        ),
        "v1_p95_ms": percentile_95(v1_times),
        "friend_average_ms": (
            statistics.fmean(friend_times)
            if friend_times
            else None
        ),
        "friend_p95_ms": percentile_95(
            friend_times
        ),
        "frames": report_rows,
    }

    json_path.write_text(
        json.dumps(
            final_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 80,
        "V1 / ARKADAŞ MODELİ 2025 RGB KARŞILAŞTIRMA ÖZETİ",
        "=" * 80,
        "",
        f"İşlenen kare: {len(report_rows)}",
        (
            "Modellerin anlaşamadığı kare: "
            f"{disagreement_count}"
        ),
        "",
        f"V1 toplam tespit: {dict(total_v1_counts)}",
        (
            "Arkadaş toplam tespit: "
            f"{dict(total_friend_counts)}"
        ),
        "",
        (
            "V1 ortalama inference ms: "
            f"{final_report['v1_average_ms']}"
        ),
        (
            "V1 p95 inference ms: "
            f"{final_report['v1_p95_ms']}"
        ),
        (
            "Arkadaş ortalama inference ms: "
            f"{final_report['friend_average_ms']}"
        ),
        (
            "Arkadaş p95 inference ms: "
            f"{final_report['friend_p95_ms']}"
        ),
        "",
        (
            "Manuel kontrol klasörü: "
            f"{DISAGREEMENT_DIR}"
        ),
        "",
        "NOT:",
        (
            "Bu karşılaştırma ground truth içermez. "
            "Daha fazla tespit, otomatik olarak daha iyi model "
            "anlamına gelmez."
        ),
        (
            "Disagreement klasöründeki görüntüler insan gözüyle "
            "kontrol edilmelidir."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("KARŞILAŞTIRMA TAMAMLANDI")
    print("=" * 80)
    print(
        f"İşlenen kare               : "
        f"{len(report_rows)}"
    )
    print(
        f"Anlaşmazlık bulunan kare   : "
        f"{disagreement_count}"
    )
    print(
        f"V1 toplam                  : "
        f"{dict(total_v1_counts)}"
    )
    print(
        f"Arkadaş modeli toplam      : "
        f"{dict(total_friend_counts)}"
    )
    print(
        f"V1 ortalama ms             : "
        f"{final_report['v1_average_ms']}"
    )
    print(
        f"Arkadaş ortalama ms        : "
        f"{final_report['friend_average_ms']}"
    )
    print()
    print(f"Özet: {summary_path}")
    print(
        f"Kontrol edilecek görseller: "
        f"{DISAGREEMENT_DIR}"
    )
    print()
    print("Geçici video otomatik silindi.")
    print("Ana model değiştirilmedi.")
    print("main.py çalıştırılmadı.")
    print("Sunucu bağlantısı kurulmadı.")


if __name__ == "__main__":
    main()