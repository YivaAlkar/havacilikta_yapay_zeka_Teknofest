from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "friend_human_candidates"
)

SOURCE_IMAGES = SOURCE_ROOT / "images"
SOURCE_LABELS = SOURCE_ROOT / "labels"
SOURCE_PREVIEWS = SOURCE_ROOT / "previews"
MANIFEST_PATH = SOURCE_ROOT / "manifest.csv"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "human_review_package"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"
OUTPUT_PREVIEWS = OUTPUT_ROOT / "previews"

# Aynı videoda seçilen iki görüntü arasında en az bu kadar frame olsun.
MIN_FRAME_GAP = 120

# Görsel hash farkı. Yüksek değer daha fazla çeşitlilik ister.
MIN_HASH_DISTANCE = 10

# Her videodan alınacak maksimum kare.
MAX_IMAGES_PER_VIDEO = 80

# Çok düşük güvenli bütün kareleri önceliksiz bırak.
MIN_MAX_CONFIDENCE = 0.40

# Bir karede aşırı sayıda kutu varsa manuel kontrolde özellikle yararlı olabilir,
# fakat hatalı kutu yoğunluğu da taşıyabilir.
MAX_ALLOWED_BOXES = 12


def prepare_directories() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for directory in (
        OUTPUT_IMAGES,
        OUTPUT_LABELS,
        OUTPUT_PREVIEWS,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {MANIFEST_PATH}"
        )

    rows: list[dict[str, Any]] = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "video": row["video"],
                    "frame_index": int(row["frame_index"]),
                    "image_file": row["image_file"],
                    "label_file": row["label_file"],
                    "preview_file": row["preview_file"],
                    "human_count": int(row["human_count"]),
                    "minimum_confidence": float(
                        row["minimum_confidence"]
                    ),
                    "maximum_confidence": float(
                        row["maximum_confidence"]
                    ),
                    "average_confidence": float(
                        row["average_confidence"]
                    ),
                }
            )

    return rows


def difference_hash(image_path: Path) -> np.ndarray:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Görüntü okunamadı: {image_path}"
        )

    resized = cv2.resize(
        image,
        (17, 16),
        interpolation=cv2.INTER_AREA,
    )

    return resized[:, 1:] > resized[:, :-1]


def hash_distance(
    first_hash: np.ndarray,
    second_hash: np.ndarray,
) -> int:
    return int(
        np.count_nonzero(
            first_hash != second_hash
        )
    )


def candidate_score(row: dict[str, Any]) -> float:
    """
    İnsan sayısı ve güveni yüksek karelere hafif öncelik verir.
    Bu skor doğruluk ölçüsü değildir.
    """
    return (
        min(row["human_count"], 8) * 2.0
        + row["maximum_confidence"] * 3.0
        + row["average_confidence"]
    )


def select_for_video(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_rows = [
        row
        for row in rows
        if row["maximum_confidence"]
        >= MIN_MAX_CONFIDENCE
        and 1 <= row["human_count"]
        <= MAX_ALLOWED_BOXES
    ]

    valid_rows.sort(
        key=lambda row: (
            row["frame_index"],
            -candidate_score(row),
        )
    )

    selected: list[dict[str, Any]] = []
    selected_hashes: list[np.ndarray] = []
    last_frame = -10**9

    for row in valid_rows:
        if len(selected) >= MAX_IMAGES_PER_VIDEO:
            break

        if (
            row["frame_index"] - last_frame
            < MIN_FRAME_GAP
        ):
            continue

        image_path = (
            SOURCE_IMAGES / row["image_file"]
        )

        if not image_path.is_file():
            continue

        current_hash = difference_hash(image_path)

        if selected_hashes:
            minimum_distance = min(
                hash_distance(
                    current_hash,
                    previous_hash,
                )
                for previous_hash in selected_hashes[-10:]
            )

            if minimum_distance < MIN_HASH_DISTANCE:
                continue

        selected.append(row)
        selected_hashes.append(current_hash)
        last_frame = row["frame_index"]

    return selected


def copy_selected_file(
    source_directory: Path,
    destination_directory: Path,
    filename: str,
) -> None:
    source = source_directory / filename
    destination = destination_directory / filename

    if not source.is_file():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {source}"
        )

    shutil.copy2(source, destination)


def count_label_boxes(label_path: Path) -> int:
    if not label_path.is_file():
        return 0

    return sum(
        1
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )


def main() -> None:
    prepare_directories()

    manifest_rows = load_manifest()

    grouped: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for row in manifest_rows:
        grouped[row["video"]].append(row)

    all_selected: list[dict[str, Any]] = []

    print("=" * 80)
    print("İNSAN REVIEW PAKETİ HAZIRLAMA")
    print("=" * 80)
    print(f"Toplam aday kare: {len(manifest_rows)}")
    print()

    for video_name, rows in sorted(grouped.items()):
        selected = select_for_video(rows)
        all_selected.extend(selected)

        print(
            f"{video_name}: "
            f"{len(rows)} aday -> "
            f"{len(selected)} seçildi"
        )

    copied_boxes = 0

    for row in all_selected:
        copy_selected_file(
            SOURCE_IMAGES,
            OUTPUT_IMAGES,
            row["image_file"],
        )

        copy_selected_file(
            SOURCE_LABELS,
            OUTPUT_LABELS,
            row["label_file"],
        )

        copy_selected_file(
            SOURCE_PREVIEWS,
            OUTPUT_PREVIEWS,
            row["preview_file"],
        )

        copied_boxes += count_label_boxes(
            OUTPUT_LABELS / row["label_file"]
        )

    selection_csv_path = (
        OUTPUT_ROOT / "selection_manifest.csv"
    )

    fieldnames = [
        "video",
        "frame_index",
        "image_file",
        "label_file",
        "preview_file",
        "human_count",
        "minimum_confidence",
        "maximum_confidence",
        "average_confidence",
    ]

    with selection_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_selected)

    report = {
        "source_candidate_images": len(
            manifest_rows
        ),
        "selected_images": len(all_selected),
        "selected_pseudo_label_boxes": (
            copied_boxes
        ),
        "configuration": {
            "minimum_frame_gap": MIN_FRAME_GAP,
            "minimum_hash_distance": (
                MIN_HASH_DISTANCE
            ),
            "maximum_images_per_video": (
                MAX_IMAGES_PER_VIDEO
            ),
            "minimum_max_confidence": (
                MIN_MAX_CONFIDENCE
            ),
            "maximum_allowed_boxes": (
                MAX_ALLOWED_BOXES
            ),
        },
        "manual_review_rules": [
            "Yanlış insan kutularını sil.",
            "Aynı insan üzerindeki tekrar kutularını sil.",
            "Eksik kalan bütün insanları kutula.",
            "Kutuları kişiyi kapsayacak şekilde düzelt.",
            "Diğer yarışma sınıfları varsa onları da doğru sınıfla etiketle.",
            "Preview dosyalarını eğitime yükleme; images klasörünü kullan.",
        ],
    }

    report_path = OUTPUT_ROOT / "report.json"
    summary_path = OUTPUT_ROOT / "README_REVIEW.txt"

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 80,
        "İNSAN REVIEW PAKETİ",
        "=" * 80,
        "",
        (
            "Kaynak aday kare: "
            f"{report['source_candidate_images']}"
        ),
        (
            "Seçilen kare: "
            f"{report['selected_images']}"
        ),
        (
            "Otomatik insan kutusu: "
            f"{report['selected_pseudo_label_boxes']}"
        ),
        "",
        "ROBOFLOW KONTROL KURALLARI:",
        "1. Yanlış kutuları sil.",
        "2. Aynı insan üzerindeki tekrar kutuları sil.",
        "3. Modelin kaçırdığı bütün insanları ekle.",
        "4. Çok geniş veya çok dar kutuları düzelt.",
        "5. Görüntüde Tasit/UAP/UAI varsa onları da etiketle.",
        "6. Eğitime previews değil images klasörünü yükle.",
        "",
        f"Images : {OUTPUT_IMAGES}",
        f"Labels : {OUTPUT_LABELS}",
        f"Preview: {OUTPUT_PREVIEWS}",
    ]

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("REVIEW PAKETİ TAMAMLANDI")
    print("=" * 80)
    print(
        f"Seçilen kare          : "
        f"{len(all_selected)}"
    )
    print(
        f"Otomatik insan kutusu : "
        f"{copied_boxes}"
    )
    print()
    print(f"Paket: {OUTPUT_ROOT}")
    print()
    print("Kaynak 526 kare değiştirilmedi.")
    print("Ana model değiştirilmedi.")
    print("main.py çalıştırılmadı.")


if __name__ == "__main__":
    main()