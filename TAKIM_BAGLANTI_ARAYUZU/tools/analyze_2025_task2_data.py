from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "example_data_audit"
    / "extracted_small_files"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task2_2025_data_analysis"

TEXT_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1254",
    "latin-1",
)

POSSIBLE_FRAME_COLUMNS = {
    "frame",
    "frame_id",
    "frame_number",
    "frame_no",
    "framenumber",
    "frameno",
    "image_id",
    "id",
    "index",
    "sira",
    "sıra",
    "kare",
    "kare_no",
    "kareno",
}


def normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{size_bytes} B"


def read_text_safely(path: Path) -> tuple[str, str]:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Dosya okunamadı: {path}",
    )


def detect_delimiter(sample: str) -> str:
    possible_delimiters = [",", ";", "\t", "|"]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters="".join(possible_delimiters),
        )
        return dialect.delimiter
    except csv.Error:
        counts = {
            delimiter: sample.count(delimiter)
            for delimiter in possible_delimiters
        }
        return max(counts, key=counts.get)


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def find_frame_column(columns: list[str]) -> str | None:
    normalized_columns = {
        normalize_text(column): column
        for column in columns
    }

    for candidate in POSSIBLE_FRAME_COLUMNS:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    for column in columns:
        normalized = normalize_text(column)

        if "frame" in normalized or "kare" in normalized:
            return column

    return None


def numeric_statistics(
    rows: list[dict[str, str]],
    columns: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for column in columns:
        values: list[float] = []

        for row in rows:
            value = parse_numeric(row.get(column))

            if value is not None:
                values.append(value)

        if not values:
            continue

        output[column] = {
            "numeric_count": len(values),
            "missing_or_non_numeric": len(rows) - len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    return output


def analyze_frame_sequence(
    rows: list[dict[str, str]],
    frame_column: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_column": frame_column,
        "numeric_frame_count": 0,
        "minimum_frame": None,
        "maximum_frame": None,
        "unique_frame_count": 0,
        "duplicate_frames": [],
        "missing_frames_preview": [],
        "strictly_increasing": None,
    }

    if frame_column is None:
        return result

    frame_values: list[int] = []

    for row in rows:
        number = parse_numeric(row.get(frame_column))

        if number is None:
            continue

        frame_values.append(int(round(number)))

    if not frame_values:
        return result

    counts = Counter(frame_values)

    duplicates = sorted(
        frame
        for frame, count in counts.items()
        if count > 1
    )

    unique_frames = sorted(set(frame_values))

    missing_frames: list[int] = []

    if len(unique_frames) >= 2:
        expected_frames = set(
            range(
                unique_frames[0],
                unique_frames[-1] + 1,
            )
        )
        missing_frames = sorted(
            expected_frames - set(unique_frames)
        )

    result.update(
        {
            "numeric_frame_count": len(frame_values),
            "minimum_frame": min(frame_values),
            "maximum_frame": max(frame_values),
            "unique_frame_count": len(unique_frames),
            "duplicate_frames": duplicates[:100],
            "missing_frames_preview": missing_frames[:100],
            "strictly_increasing": all(
                current > previous
                for previous, current in zip(
                    frame_values,
                    frame_values[1:],
                )
            ),
        }
    )

    return result


def analyze_csv(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.relative_to(INPUT_ROOT)),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
        "encoding": None,
        "delimiter": None,
        "columns": [],
        "row_count": 0,
        "first_rows": [],
        "last_rows": [],
        "numeric_statistics": {},
        "frame_analysis": {},
        "error": None,
    }

    try:
        text, encoding = read_text_safely(path)
        delimiter = detect_delimiter(text[:16384])

        reader = csv.DictReader(
            text.splitlines(),
            delimiter=delimiter,
        )

        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

        frame_column = find_frame_column(columns)

        report.update(
            {
                "encoding": encoding,
                "delimiter": (
                    "\\t" if delimiter == "\t" else delimiter
                ),
                "columns": columns,
                "row_count": len(rows),
                "first_rows": rows[:5],
                "last_rows": rows[-5:] if rows else [],
                "numeric_statistics": numeric_statistics(
                    rows,
                    columns,
                ),
                "frame_analysis": analyze_frame_sequence(
                    rows,
                    frame_column,
                ),
            }
        )

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    return report


def analyze_calibration_text(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.relative_to(INPUT_ROOT)),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
        "encoding": None,
        "line_count": 0,
        "content": None,
        "numeric_values": [],
        "error": None,
    }

    try:
        text, encoding = read_text_safely(path)
        lines = text.splitlines()

        numeric_values: list[float] = []

        for token in (
            text.replace(",", " ")
            .replace(";", " ")
            .replace("=", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace("(", " ")
            .replace(")", " ")
            .split()
        ):
            value = parse_numeric(token)

            if value is not None:
                numeric_values.append(value)

        report.update(
            {
                "encoding": encoding,
                "line_count": len(lines),
                "content": text,
                "numeric_values": numeric_values,
            }
        )

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    return report


def classify_csv_name(filename: str) -> dict[str, Any]:
    lowered = filename.lower()

    sample_number = None

    if "veri-1" in lowered:
        sample_number = 1
    elif "veri-2" in lowered:
        sample_number = 2

    modality = None

    if "rgb" in lowered:
        modality = "RGB"
    elif "termal" in lowered or "thermal" in lowered:
        modality = "THERMAL"

    return {
        "sample_number": sample_number,
        "modality": modality,
    }


def create_pair_comparisons(
    csv_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}

    for report in csv_reports:
        metadata = classify_csv_name(report["path"])

        sample_number = metadata["sample_number"]
        modality = metadata["modality"]

        if sample_number is None or modality is None:
            continue

        grouped.setdefault(sample_number, {})[modality] = report

    comparisons: list[dict[str, Any]] = []

    for sample_number, modalities in sorted(grouped.items()):
        rgb = modalities.get("RGB")
        thermal = modalities.get("THERMAL")

        comparison: dict[str, Any] = {
            "sample_number": sample_number,
            "rgb_found": rgb is not None,
            "thermal_found": thermal is not None,
            "rgb_row_count": rgb["row_count"] if rgb else None,
            "thermal_row_count": (
                thermal["row_count"] if thermal else None
            ),
            "row_count_difference": None,
            "same_columns": None,
            "rgb_columns": rgb["columns"] if rgb else None,
            "thermal_columns": (
                thermal["columns"] if thermal else None
            ),
        }

        if rgb and thermal:
            comparison["row_count_difference"] = (
                rgb["row_count"] - thermal["row_count"]
            )
            comparison["same_columns"] = (
                rgb["columns"] == thermal["columns"]
            )

        comparisons.append(comparison)

    return comparisons


def format_row(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def build_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("HYZ 2025 TASK 2 VERİ ANALİZİ")
    lines.append("=" * 100)
    lines.append(f"Tarih: {report['analysis_time']}")
    lines.append(f"Kaynak: {report['input_root']}")
    lines.append("")

    lines.append("[TRANSLATION CSV DOSYALARI]")
    lines.append("")

    for csv_report in report["csv_files"]:
        lines.append("-" * 100)
        lines.append(f"Dosya: {csv_report['path']}")
        lines.append(f"Boyut: {csv_report['size_human']}")
        lines.append(f"Encoding: {csv_report['encoding']}")
        lines.append(f"Delimiter: {csv_report['delimiter']}")
        lines.append(f"Satır sayısı: {csv_report['row_count']}")
        lines.append(f"Sütunlar: {csv_report['columns']}")
        lines.append(f"Hata: {csv_report['error']}")
        lines.append("")

        frame_analysis = csv_report["frame_analysis"]

        lines.append(
            f"Frame sütunu: "
            f"{frame_analysis.get('frame_column')}"
        )
        lines.append(
            f"Frame aralığı: "
            f"{frame_analysis.get('minimum_frame')} -> "
            f"{frame_analysis.get('maximum_frame')}"
        )
        lines.append(
            f"Tekil frame: "
            f"{frame_analysis.get('unique_frame_count')}"
        )
        lines.append(
            f"Tekrarlı frame: "
            f"{frame_analysis.get('duplicate_frames')}"
        )
        lines.append(
            f"Eksik frame önizleme: "
            f"{frame_analysis.get('missing_frames_preview')}"
        )
        lines.append(
            f"Kesin artan sıra: "
            f"{frame_analysis.get('strictly_increasing')}"
        )
        lines.append("")

        lines.append("İlk satırlar:")

        for row in csv_report["first_rows"]:
            lines.append(f"  {format_row(row)}")

        lines.append("")
        lines.append("Son satırlar:")

        for row in csv_report["last_rows"]:
            lines.append(f"  {format_row(row)}")

        lines.append("")
        lines.append("Sayısal sütun istatistikleri:")

        for column, statistics_data in (
            csv_report["numeric_statistics"].items()
        ):
            lines.append(
                f"  {column}: "
                f"count={statistics_data['numeric_count']}, "
                f"min={statistics_data['minimum']}, "
                f"max={statistics_data['maximum']}, "
                f"mean={statistics_data['mean']}, "
                f"median={statistics_data['median']}"
            )

        lines.append("")

    lines.append("=" * 100)
    lines.append("[RGB / THERMAL EŞLEŞME KARŞILAŞTIRMASI]")
    lines.append("")

    for comparison in report["pair_comparisons"]:
        lines.append(
            f"Örnek Veri {comparison['sample_number']}:"
        )
        lines.append(
            f"  RGB satır sayısı: "
            f"{comparison['rgb_row_count']}"
        )
        lines.append(
            f"  Thermal satır sayısı: "
            f"{comparison['thermal_row_count']}"
        )
        lines.append(
            f"  Satır farkı: "
            f"{comparison['row_count_difference']}"
        )
        lines.append(
            f"  Aynı sütun yapısı: "
            f"{comparison['same_columns']}"
        )
        lines.append("")

    lines.append("=" * 100)
    lines.append("[KALİBRASYON DOSYALARI]")
    lines.append("")

    for calibration_report in report["calibration_files"]:
        lines.append("-" * 100)
        lines.append(f"Dosya: {calibration_report['path']}")
        lines.append(
            f"Satır sayısı: "
            f"{calibration_report['line_count']}"
        )
        lines.append(
            f"Sayısal değer sayısı: "
            f"{len(calibration_report['numeric_values'])}"
        )
        lines.append(f"Hata: {calibration_report['error']}")
        lines.append("")
        lines.append("İçerik:")
        lines.append(calibration_report["content"] or "")
        lines.append("")

    lines.append("=" * 100)
    lines.append("[GÜVENLİK]")
    lines.append("- Videolar çıkarılmadı.")
    lines.append("- Üretim kodu değiştirilmedi.")
    lines.append("- Model dosyaları değiştirilmedi.")
    lines.append("- main.py çalıştırılmadı.")
    lines.append("- Sunucuya bağlanılmadı.")
    lines.append("- .env okunmadı.")

    return "\n".join(lines)


def main() -> None:
    print("=" * 80)
    print("HYZ 2025 TASK 2 VERİ ANALİZİ")
    print("=" * 80)
    print(f"Kaynak klasör: {INPUT_ROOT}")
    print()

    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Kaynak klasör bulunamadı: {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(
        path
        for path in INPUT_ROOT.rglob("*.csv")
        if path.is_file()
    )

    calibration_paths = sorted(
        path
        for path in INPUT_ROOT.rglob("*.txt")
        if path.is_file()
        and (
            "kalibrasyon" in path.name.lower()
            or "calibration" in path.name.lower()
        )
    )

    print(f"CSV sayısı        : {len(csv_paths)}")
    print(f"Kalibrasyon sayısı: {len(calibration_paths)}")
    print()

    csv_reports: list[dict[str, Any]] = []

    for path in csv_paths:
        print(f"[CSV] {path.name}")
        csv_reports.append(analyze_csv(path))

    calibration_reports: list[dict[str, Any]] = []

    for path in calibration_paths:
        print(f"[KALİBRASYON] {path.name}")
        calibration_reports.append(
            analyze_calibration_text(path)
        )

    report: dict[str, Any] = {
        "analysis_time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "input_root": str(INPUT_ROOT),
        "csv_count": len(csv_reports),
        "calibration_count": len(calibration_reports),
        "csv_files": csv_reports,
        "calibration_files": calibration_reports,
        "pair_comparisons": create_pair_comparisons(
            csv_reports
        ),
        "safety": {
            "videos_extracted": False,
            "production_code_modified": False,
            "models_modified": False,
            "main_executed": False,
            "server_connection_attempted": False,
            "env_read": False,
        },
    }

    json_path = OUTPUT_ROOT / "task2_2025_data_report.json"
    text_path = OUTPUT_ROOT / "task2_2025_data_report.txt"

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_path.write_text(
        build_text_report(report),
        encoding="utf-8",
    )

    error_count = sum(
        1
        for item in csv_reports + calibration_reports
        if item["error"] is not None
    )

    print()
    print("=" * 80)
    print("ANALİZ TAMAMLANDI")
    print("=" * 80)
    print(f"CSV analiz edildi        : {len(csv_reports)}")
    print(
        f"Kalibrasyon analiz edildi: "
        f"{len(calibration_reports)}"
    )
    print(f"Hata                     : {error_count}")
    print()
    print(f"Metin raporu: {text_path}")
    print(f"JSON raporu : {json_path}")
    print()
    print("Videolar çıkarılmadı.")
    print("Ana sistem değiştirilmedi.")
    print("main.py çalıştırılmadı.")
    print("Sunucu bağlantısı kurulmadı.")


if __name__ == "__main__":
    main()