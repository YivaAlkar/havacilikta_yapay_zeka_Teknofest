from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path(
    r"C:\Users\muham\OneDrive\Masaüstü\HYZ_2025_Ornek_Veriler.zip"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_friend_yolo11s_2class.pt"
)

VIDEO_NAMES = [
    "Ornek-Veri-1-RGB.MP4",
    "Ornek-Veri-2-RGB.MP4",
]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "friend_human_candidates"
)

IMAGES_DIR = OUTPUT_ROOT / "images"
LABELS_DIR = OUTPUT_ROOT / "labels"
PREVIEWS_DIR = OUTPUT_ROOT / "previews"

# Arkadaş modelindeki insan sınıfı.
FRIEND_HUMAN_CLASS_ID = 1

# Yeni 4 sınıflı yarışma datasetinde de insan sınıfı 1.
COMPETITION_HUMAN_CLASS_ID = 1

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 1024

# Her 5 kareden birini modelden geçirir.
SCAN_STRIDE = 5

# Bir insan adayı kaydedildikten sonra aynı videodan
# en az 30 frame geçmeden başka kare kaydetmez.
MIN_SAVED_FRAME_GAP = 30

# Aşırı küçük ve güvenilmez kutuları elemek için.
MIN_BOX_WIDTH_PX = 8
MIN_BOX_HEIGHT_PX = 8
MIN_BOX_AREA_PX = 100

# Bir videodan maksimum kaç aday kare kaydedilecek.
MAX_SAVED_FRAMES_PER_VIDEO = 300


def ensure_inputs() -> None:
    missing = []

    if not ZIP_PATH.is_file():
        missing.append(str(ZIP_PATH))

    if not MODEL_PATH.is_file():
        missing.append(str(MODEL_PATH))

    if missing:
        raise FileNotFoundError(
            "Eksik dosyalar:\n" + "\n".join(missing)
        )


def prepare_output_directories() -> None:
    for directory in (
        OUTPUT_ROOT,
        IMAGES_DIR,
        LABELS_DIR,
        PREVIEWS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


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


def validate_friend_model(model: YOLO) -> dict[int, str]:
    names = normalize_names(model.names)

    expected = {
        0: "tasit",
        1: "insan",
    }

    if set(names) != set(expected):
        raise ValueError(
            f"Arkadaş modeli class ID yapısı yanlış: {names}"
        )

    for class_id, expected_name in expected.items():
        actual_name = names[class_id]

        if normalize_class_name(actual_name) != expected_name:
            raise ValueError(
                f"Class sırası yanlış. ID={class_id}, "
                f"beklenen={expected_name}, bulunan={actual_name}"
            )

    return names


def find_archive_member(
    archive: zipfile.ZipFile,
    target_name: str,
) -> str:
    matches = [
        member
        for member in archive.namelist()
        if Path(member).name.lower() == target_name.lower()
    ]

    if not matches:
        raise FileNotFoundError(
            f"ZIP içinde video bulunamadı: {target_name}"
        )

    return matches[0]


def extract_video(
    archive: zipfile.ZipFile,
    member_name: str,
    output_path: Path,
) -> None:
    print(f"[ÇIKARILIYOR] {member_name}")
    print("Büyük RGB video olduğu için birkaç dakika sürebilir.")

    with archive.open(member_name, "r") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(
                source,
                target,
                length=8 * 1024 * 1024,
            )


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def valid_human_detections(
    result: Any,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if class_id != FRIEND_HUMAN_CLASS_ID:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        x1 = max(0.0, min(x1, frame_width - 1.0))
        y1 = max(0.0, min(y1, frame_height - 1.0))
        x2 = max(0.0, min(x2, frame_width - 1.0))
        y2 = max(0.0, min(y2, frame_height - 1.0))

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        if box_width < MIN_BOX_WIDTH_PX:
            continue

        if box_height < MIN_BOX_HEIGHT_PX:
            continue

        if box_area < MIN_BOX_AREA_PX:
            continue

        center_x = ((x1 + x2) / 2.0) / frame_width
        center_y = ((y1 + y2) / 2.0) / frame_height
        normalized_width = box_width / frame_width
        normalized_height = box_height / frame_height

        detections.append(
            {
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
                "yolo": [
                    COMPETITION_HUMAN_CLASS_ID,
                    center_x,
                    center_y,
                    normalized_width,
                    normalized_height,
                ],
                "width_px": box_width,
                "height_px": box_height,
                "area_px": box_area,
            }
        )

    return detections


def write_yolo_label(
    label_path: Path,
    detections: list[dict[str, Any]],
) -> None:
    lines = []

    for detection in detections:
        class_id, cx, cy, width, height = detection["yolo"]

        lines.append(
            f"{int(class_id)} "
            f"{cx:.8f} "
            f"{cy:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

    label_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def create_preview(
    frame: Any,
    detections: list[dict[str, Any]],
    video_name: str,
    frame_index: int,
) -> Any:
    preview = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in detection["xyxy"]
        ]

        confidence = detection["confidence"]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            preview,
            f"Insan {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    title = (
        f"{video_name} | frame={frame_index} | "
        f"insan={len(detections)}"
    )

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 50),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        preview,
        title,
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def process_video(
    model: YOLO,
    video_path: Path,
    original_video_name: str,
    device: str | int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))

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

    video_stem = Path(original_video_name).stem

    scanned_frames = 0
    frames_with_human = 0
    saved_frames = 0
    total_boxes = 0
    last_saved_frame = -10**9
    inference_times: list[float] = []
    manifest_rows: list[dict[str, Any]] = []

    print()
    print("=" * 80)
    print(f"VİDEO: {original_video_name}")
    print("=" * 80)
    print(f"Toplam frame : {total_frames}")
    print(f"FPS          : {fps}")
    print(f"Çözünürlük   : {width}x{height}")
    print(f"Tarama adımı : {SCAN_STRIDE}")
    print()

    frame_index = 0

    while frame_index < total_frames:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = capture.read()

        if not success or frame is None:
            frame_index += SCAN_STRIDE
            continue

        scanned_frames += 1

        synchronize_cuda()
        start = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            classes=[FRIEND_HUMAN_CLASS_ID],
            verbose=False,
        )

        synchronize_cuda()
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        inference_times.append(elapsed_ms)

        if not results:
            frame_index += SCAN_STRIDE
            continue

        detections = valid_human_detections(
            results[0],
            frame_width=width,
            frame_height=height,
        )

        if not detections:
            frame_index += SCAN_STRIDE
            continue

        frames_with_human += 1

        can_save = (
            frame_index - last_saved_frame
            >= MIN_SAVED_FRAME_GAP
        )

        if (
            can_save
            and saved_frames < MAX_SAVED_FRAMES_PER_VIDEO
        ):
            filename_stem = (
                f"{video_stem}_frame_{frame_index:06d}"
            )

            image_path = (
                IMAGES_DIR / f"{filename_stem}.jpg"
            )
            label_path = (
                LABELS_DIR / f"{filename_stem}.txt"
            )
            preview_path = (
                PREVIEWS_DIR / f"{filename_stem}.jpg"
            )

            cv2.imwrite(
                str(image_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )

            write_yolo_label(
                label_path,
                detections,
            )

            preview = create_preview(
                frame=frame,
                detections=detections,
                video_name=original_video_name,
                frame_index=frame_index,
            )

            cv2.imwrite(
                str(preview_path),
                preview,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    92,
                ],
            )

            saved_frames += 1
            total_boxes += len(detections)
            last_saved_frame = frame_index

            confidences = [
                detection["confidence"]
                for detection in detections
            ]

            manifest_rows.append(
                {
                    "video": original_video_name,
                    "frame_index": frame_index,
                    "time_seconds": (
                        frame_index / fps
                        if fps > 0
                        else None
                    ),
                    "image_file": image_path.name,
                    "label_file": label_path.name,
                    "preview_file": preview_path.name,
                    "human_count": len(detections),
                    "minimum_confidence": min(confidences),
                    "maximum_confidence": max(confidences),
                    "average_confidence": (
                        sum(confidences)
                        / len(confidences)
                    ),
                    "inference_ms": elapsed_ms,
                }
            )

            print(
                f"[KAYDEDİLDİ] frame={frame_index} | "
                f"insan={len(detections)} | "
                f"toplam={saved_frames}",
                flush=True,
            )

        if saved_frames >= MAX_SAVED_FRAMES_PER_VIDEO:
            print(
                f"[SINIR] {original_video_name} için "
                f"{MAX_SAVED_FRAMES_PER_VIDEO} aday kareye ulaşıldı."
            )
            break

        frame_index += SCAN_STRIDE

    capture.release()

    average_ms = (
        sum(inference_times) / len(inference_times)
        if inference_times
        else None
    )

    return {
        "video": original_video_name,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "scanned_frames": scanned_frames,
        "frames_with_human_before_gap_filter": frames_with_human,
        "saved_frames": saved_frames,
        "saved_human_boxes": total_boxes,
        "average_inference_ms": average_ms,
        "manifest_rows": manifest_rows,
    }


def main() -> None:
    ensure_inputs()
    prepare_output_directories()

    device: str | int = (
        0 if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("ARKADAŞ MODELİ İNSAN ADAYI ÇIKARMA")
    print("=" * 80)
    print(f"CUDA          : {torch.cuda.is_available()}")
    print(
        "GPU           : "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
    )
    print(f"Confidence    : {CONFIDENCE_THRESHOLD}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Scan stride   : {SCAN_STRIDE}")
    print(f"Save gap      : {MIN_SAVED_FRAME_GAP}")
    print()
    print("UYARI:")
    print("- Bunlar otomatik pseudo-label adaylarıdır.")
    print("- Elle kontrol edilmeden eğitime eklenmemelidir.")
    print("- Yalnız insan class ID 1 kaydedilir.")
    print()

    print("[MODEL] Arkadaş modeli yükleniyor...")
    model = YOLO(str(MODEL_PATH))
    names = validate_friend_model(model)

    print(f"Model sınıfları: {names}")

    video_reports: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        for video_name in VIDEO_NAMES:
            member_name = find_archive_member(
                archive,
                video_name,
            )

            with tempfile.TemporaryDirectory(
                prefix="friend_human_mining_"
            ) as temporary_directory:
                temporary_video_path = (
                    Path(temporary_directory)
                    / video_name
                )

                extract_video(
                    archive=archive,
                    member_name=member_name,
                    output_path=temporary_video_path,
                )

                report = process_video(
                    model=model,
                    video_path=temporary_video_path,
                    original_video_name=video_name,
                    device=device,
                )

                all_manifest_rows.extend(
                    report.pop("manifest_rows")
                )
                video_reports.append(report)

                print(
                    f"[TEMİZLENDİ] Geçici video silindi: "
                    f"{video_name}"
                )

    manifest_csv_path = OUTPUT_ROOT / "manifest.csv"
    report_json_path = OUTPUT_ROOT / "report.json"
    summary_path = OUTPUT_ROOT / "summary.txt"

    fieldnames = [
        "video",
        "frame_index",
        "time_seconds",
        "image_file",
        "label_file",
        "preview_file",
        "human_count",
        "minimum_confidence",
        "maximum_confidence",
        "average_confidence",
        "inference_ms",
    ]

    with manifest_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_manifest_rows)

    total_saved_frames = sum(
        report["saved_frames"]
        for report in video_reports
    )

    total_saved_boxes = sum(
        report["saved_human_boxes"]
        for report in video_reports
    )

    final_report = {
        "zip_path": str(ZIP_PATH),
        "model_path": str(MODEL_PATH),
        "model_names": names,
        "configuration": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "scan_stride": SCAN_STRIDE,
            "minimum_saved_frame_gap": MIN_SAVED_FRAME_GAP,
            "minimum_box_width_px": MIN_BOX_WIDTH_PX,
            "minimum_box_height_px": MIN_BOX_HEIGHT_PX,
            "minimum_box_area_px": MIN_BOX_AREA_PX,
            "maximum_saved_frames_per_video": (
                MAX_SAVED_FRAMES_PER_VIDEO
            ),
        },
        "videos": video_reports,
        "total_saved_frames": total_saved_frames,
        "total_saved_human_boxes": total_saved_boxes,
        "manual_review_required": True,
    }

    report_json_path.write_text(
        json.dumps(
            final_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 80,
        "ARKADAŞ MODELİ İNSAN ADAYI ÖZETİ",
        "=" * 80,
        "",
        f"Toplam kaydedilen kare: {total_saved_frames}",
        f"Toplam insan kutusu: {total_saved_boxes}",
        "",
    ]

    for report in video_reports:
        summary_lines.extend(
            [
                f"Video: {report['video']}",
                f"  Toplam frame: {report['total_frames']}",
                f"  Taranan frame: {report['scanned_frames']}",
                (
                    "  İnsan bulunan frame: "
                    f"{report['frames_with_human_before_gap_filter']}"
                ),
                f"  Kaydedilen frame: {report['saved_frames']}",
                f"  Kaydedilen insan kutusu: {report['saved_human_boxes']}",
                (
                    "  Ortalama inference ms: "
                    f"{report['average_inference_ms']}"
                ),
                "",
            ]
        )

    summary_lines.extend(
        [
            "Çıktı klasörleri:",
            f"  Görüntüler: {IMAGES_DIR}",
            f"  Etiketler : {LABELS_DIR}",
            f"  Önizleme  : {PREVIEWS_DIR}",
            "",
            "UYARI:",
            (
                "Bu kutular elle kontrol edilmeden dataset v4'e "
                "eklenmemelidir."
            ),
            (
                "Eksik insan kutuları tamamlanmalı, yanlış kutular "
                "silinmelidir."
            ),
        ]
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("İNSAN ADAYI ÇIKARMA TAMAMLANDI")
    print("=" * 80)
    print(f"Kaydedilen kare       : {total_saved_frames}")
    print(f"Kaydedilen insan kutusu: {total_saved_boxes}")
    print()
    print(f"Özet      : {summary_path}")
    print(f"Manifest  : {manifest_csv_path}")
    print(f"Önizlemeler: {PREVIEWS_DIR}")
    print()
    print("Ana model değiştirilmedi.")
    print("Geçici videolar silindi.")
    print("main.py çalıştırılmadı.")
    print("Sunucu bağlantısı kurulmadı.")


if __name__ == "__main__":
    main()