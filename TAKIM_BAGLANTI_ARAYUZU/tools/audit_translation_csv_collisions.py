from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_DIR = PROJECT_ROOT / "data" / "translations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "translation_collision_audit"

FIRST_COMPETITION_FRAMES = 2250


def parse_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def normalize_frame_name(value: Any) -> str:
    text = str(value).strip()
    name = Path(text.replace("\\", "/")).name
    return Path(name).stem


def frame_number(frame_name: str) -> int | None:
    normalized = normalize_frame_name(frame_name)

    if not normalized.startswith("frame_"):
        return None

    suffix = normalized.removeprefix("frame_")

    try:
        return int(suffix)
    except ValueError:
        return None


def detect_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(
            text[:8192],
            delimiters=",;\t|",
        ).delimiter
    except csv.Error:
        return ","


def load_translation_csv(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    delimiter = detect_delimiter(text)

    reader = csv.DictReader(
        text.splitlines(),
        delimiter=delimiter,
    )

    columns = list(reader.fieldnames or [])

    possible_frame_columns = [
        "frame_numbers",
        "frame_number",
        "frame",
        "image_url",
        "image_name",
    ]

    frame_column = next(
        (
            column
            for column in possible_frame_columns
            if column in columns
        ),
        None,
    )

    if frame_column is None:
        raise ValueError(
            f"Frame sütunu bulunamadı: {path.name}, columns={columns}"
        )

    required_translation_columns = [
        "translation_x",
        "translation_y",
        "translation_z",
    ]

    missing = [
        column
        for column in required_translation_columns
        if column not in columns
    ]

    if missing:
        raise ValueError(
            f"Eksik translation sütunları: {path.name}, missing={missing}"
        )

    values: dict[str, tuple[float, float, float]] = {}
    duplicate_frames: list[str] = []
    invalid_rows: list[int] = []

    for row_number, row in enumerate(reader, start=2):
        frame_name = normalize_frame_name(row.get(frame_column))

        tx = parse_float(row.get("translation_x"))
        ty = parse_float(row.get("translation_y"))
        tz = parse_float(row.get("translation_z"))

        if not frame_name or None in (tx, ty, tz):
            invalid_rows.append(row_number)
            continue

        if frame_name in values:
            duplicate_frames.append(frame_name)

        values[frame_name] = (tx, ty, tz)

    numeric_frames = [
        number
        for name in values
        if (number := frame_number(name)) is not None
    ]

    return {
        "path": path,
        "columns": columns,
        "frame_column": frame_column,
        "values": values,
        "duplicate_frames": sorted(set(duplicate_frames)),
        "invalid_rows": invalid_rows,
        "minimum_frame": min(numeric_frames) if numeric_frames else None,
        "maximum_frame": max(numeric_frames) if numeric_frames else None,
    }


def axis_statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "rmse": None,
        }

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "rmse": math.sqrt(
            statistics.fmean(value * value for value in values)
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(TRANSLATION_DIR.glob("*.csv"))

    print("=" * 80)
    print("TRANSLATION CSV ÇAKIŞMA AUDIT")
    print("=" * 80)
    print(f"CSV sayısı: {len(csv_paths)}")

    if len(csv_paths) < 2:
        raise RuntimeError(
            "Karşılaştırma için en az iki CSV gerekli."
        )

    reports = [load_translation_csv(path) for path in csv_paths]

    for index, report in enumerate(reports, start=1):
        print()
        print(f"[{index}] {report['path'].name}")
        print(f"Satır/frame sayısı : {len(report['values'])}")
        print(
            f"Frame aralığı      : "
            f"{report['minimum_frame']} -> {report['maximum_frame']}"
        )
        print(
            f"Tekrarlı frame     : "
            f"{len(report['duplicate_frames'])}"
        )
        print(
            f"Geçersiz satır     : "
            f"{len(report['invalid_rows'])}"
        )

    first = reports[0]
    second = reports[1]

    first_frames = set(first["values"])
    second_frames = set(second["values"])

    common_frames = sorted(
        first_frames & second_frames,
        key=lambda name: (
            frame_number(name)
            if frame_number(name) is not None
            else 10**12,
            name,
        ),
    )

    competition_common_frames = [
        name
        for name in common_frames
        if (
            frame_number(name) is not None
            and frame_number(name) < FIRST_COMPETITION_FRAMES
        )
    ]

    dx_values: list[float] = []
    dy_values: list[float] = []
    dz_values: list[float] = []
    euclidean_values: list[float] = []

    largest_differences: list[dict[str, Any]] = []

    for name in common_frames:
        first_value = first["values"][name]
        second_value = second["values"][name]

        dx = first_value[0] - second_value[0]
        dy = first_value[1] - second_value[1]
        dz = first_value[2] - second_value[2]

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        dx_values.append(dx)
        dy_values.append(dy)
        dz_values.append(dz)
        euclidean_values.append(distance)

        largest_differences.append(
            {
                "frame": name,
                "first": first_value,
                "second": second_value,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "euclidean_distance": distance,
            }
        )

    largest_differences.sort(
        key=lambda item: item["euclidean_distance"],
        reverse=True,
    )

    audit = {
        "load_order": [
            report["path"].name
            for report in reports
        ],
        "files": [
            {
                "name": report["path"].name,
                "columns": report["columns"],
                "frame_column": report["frame_column"],
                "frame_count": len(report["values"]),
                "minimum_frame": report["minimum_frame"],
                "maximum_frame": report["maximum_frame"],
                "duplicate_frame_count": len(
                    report["duplicate_frames"]
                ),
                "invalid_row_count": len(report["invalid_rows"]),
            }
            for report in reports
        ],
        "comparison": {
            "common_frame_count": len(common_frames),
            "common_frames_first_2250": len(
                competition_common_frames
            ),
            "only_in_first": len(first_frames - second_frames),
            "only_in_second": len(second_frames - first_frames),
            "dx_statistics": axis_statistics(dx_values),
            "dy_statistics": axis_statistics(dy_values),
            "dz_statistics": axis_statistics(dz_values),
            "euclidean_statistics": axis_statistics(
                euclidean_values
            ),
            "largest_differences": largest_differences[:20],
        },
        "current_code_risk": {
            "first_reader_name": first["path"].name,
            "first_match_wins": True,
            "likely_first_reader_used_for_overlapping_frames": (
                len(common_frames) > 0
            ),
        },
    }

    json_path = OUTPUT_DIR / "translation_collision_report.json"
    text_path = OUTPUT_DIR / "translation_collision_report.txt"

    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 80,
        "TRANSLATION CSV ÇAKIŞMA RAPORU",
        "=" * 80,
        "",
        "[YÜKLEME SIRASI]",
    ]

    for index, name in enumerate(audit["load_order"], start=1):
        lines.append(f"{index}. {name}")

    comparison = audit["comparison"]

    lines.extend(
        [
            "",
            "[ÇAKIŞMA]",
            f"Ortak frame sayısı: {comparison['common_frame_count']}",
            (
                "İlk 2250 frame içindeki ortak frame: "
                f"{comparison['common_frames_first_2250']}"
            ),
            f"Yalnız ilk CSV'de: {comparison['only_in_first']}",
            f"Yalnız ikinci CSV'de: {comparison['only_in_second']}",
            "",
            "[FARK İSTATİSTİKLERİ]",
            (
                "DX RMSE: "
                f"{comparison['dx_statistics']['rmse']}"
            ),
            (
                "DY RMSE: "
                f"{comparison['dy_statistics']['rmse']}"
            ),
            (
                "DZ RMSE: "
                f"{comparison['dz_statistics']['rmse']}"
            ),
            (
                "3B mesafe ortalaması: "
                f"{comparison['euclidean_statistics']['mean']}"
            ),
            (
                "3B mesafe maksimumu: "
                f"{comparison['euclidean_statistics']['maximum']}"
            ),
            "",
            "[MEVCUT KOD DAVRANIŞI]",
            (
                "İlk reader: "
                f"{audit['current_code_risk']['first_reader_name']}"
            ),
            "Ortak frame bulunursa ilk reader sonucu döndürülür.",
            "",
            f"JSON: {json_path}",
        ]
    )

    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("AUDIT TAMAMLANDI")
    print("=" * 80)
    print(f"Ortak frame sayısı              : {len(common_frames)}")
    print(
        "İlk 2250 içindeki ortak frame  : "
        f"{len(competition_common_frames)}"
    )
    print(
        "İlk eşleşmede kullanılacak CSV : "
        f"{first['path'].name}"
    )
    print(
        "3B fark ortalaması              : "
        f"{audit['comparison']['euclidean_statistics']['mean']}"
    )
    print(
        "3B fark maksimumu               : "
        f"{audit['comparison']['euclidean_statistics']['maximum']}"
    )
    print()
    print(f"Rapor: {text_path}")
    print()
    print("Üretim kodu değiştirilmedi.")
    print("CSV dosyaları değiştirilmedi.")
    print("main.py çalıştırılmadı.")


if __name__ == "__main__":
    main()