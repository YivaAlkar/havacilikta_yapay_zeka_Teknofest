from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_DIRECTORIES = [
    PROJECT_ROOT / "data" / "translations",
    PROJECT_ROOT
    / "outputs"
    / "example_data_audit"
    / "extracted_small_files",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "task2_motion_rmse"
)

GAP_LENGTHS = (1, 2, 4, 8)

VELOCITY_DAMPING = 0.85
MAX_TRANSLATION_STEP = 25.0


def safe_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def detect_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(
            text[:8192],
            delimiters=",;\t|",
        ).delimiter
    except csv.Error:
        return ","


def parse_frame_number(value: Any) -> int | None:
    text = str(value).strip()
    stem = Path(text.replace("\\", "/")).stem

    if stem.startswith("frame_"):
        stem = stem.removeprefix("frame_")

    try:
        return int(stem)
    except ValueError:
        return None


def load_translation_csv(
    path: Path,
) -> list[tuple[int, tuple[float, float, float]]]:
    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    delimiter = detect_delimiter(text)

    reader = csv.DictReader(
        text.splitlines(),
        delimiter=delimiter,
    )

    fieldnames = set(reader.fieldnames or [])

    required = {
        "translation_x",
        "translation_y",
        "translation_z",
    }

    if not required.issubset(fieldnames):
        return []

    frame_column = next(
        (
            name
            for name in (
                "frame_numbers",
                "frame_number",
                "frame",
            )
            if name in fieldnames
        ),
        None,
    )

    if frame_column is None:
        return []

    values: dict[
        int,
        tuple[float, float, float],
    ] = {}

    for row in reader:
        frame = parse_frame_number(
            row.get(frame_column)
        )

        tx = safe_float(
            row.get("translation_x")
        )
        ty = safe_float(
            row.get("translation_y")
        )
        tz = safe_float(
            row.get("translation_z")
        )

        if frame is None or None in (tx, ty, tz):
            continue

        values[frame] = (tx, ty, tz)

    return sorted(values.items())


def clamp(value: float) -> float:
    return max(
        -MAX_TRANSLATION_STEP,
        min(MAX_TRANSLATION_STEP, value),
    )


def vector_error(
    predicted: tuple[float, float, float],
    actual: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    dx = predicted[0] - actual[0]
    dy = predicted[1] - actual[1]
    dz = predicted[2] - actual[2]

    distance = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    return dx, dy, dz, distance


def rmse(values: list[float]) -> float | None:
    if not values:
        return None

    return math.sqrt(
        statistics.fmean(
            value * value
            for value in values
        )
    )


def mean(values: list[float]) -> float | None:
    if not values:
        return None

    return statistics.fmean(values)


def evaluate_gap(
    sequence: list[
        tuple[int, tuple[float, float, float]]
    ],
    gap_length: int,
) -> dict[str, Any]:
    frame_map = dict(sequence)
    frame_numbers = sorted(frame_map)

    hold_dx: list[float] = []
    hold_dy: list[float] = []
    hold_dz: list[float] = []
    hold_distances: list[float] = []

    motion_dx: list[float] = []
    motion_dy: list[float] = []
    motion_dz: list[float] = []
    motion_distances: list[float] = []

    evaluated_windows = 0
    evaluated_points = 0

    for current_frame in frame_numbers:
        previous_frame = current_frame - 1

        if previous_frame not in frame_map:
            continue

        required_future = [
            current_frame + offset
            for offset in range(
                1,
                gap_length + 1,
            )
        ]

        if not all(
            frame in frame_map
            for frame in required_future
        ):
            continue

        previous = frame_map[previous_frame]
        current = frame_map[current_frame]

        velocity = (
            clamp(current[0] - previous[0]),
            clamp(current[1] - previous[1]),
            clamp(current[2] - previous[2]),
        )

        motion_prediction = current

        for age, future_frame in enumerate(
            required_future,
            start=1,
        ):
            actual = frame_map[future_frame]

            # Eski yöntem:
            # Son güvenilir değeri sabit tutar.
            hold_prediction = current

            hdx, hdy, hdz, hdistance = (
                vector_error(
                    hold_prediction,
                    actual,
                )
            )

            hold_dx.append(hdx)
            hold_dy.append(hdy)
            hold_dz.append(hdz)
            hold_distances.append(hdistance)

            # Yeni yöntem:
            # İlk kayıp frame'de tam hız,
            # sonraki frame'lerde damping.
            damping = (
                VELOCITY_DAMPING
                ** max(0, age - 1)
            )

            step = (
                clamp(velocity[0] * damping),
                clamp(velocity[1] * damping),
                clamp(velocity[2] * damping),
            )

            motion_prediction = (
                motion_prediction[0] + step[0],
                motion_prediction[1] + step[1],
                motion_prediction[2] + step[2],
            )

            mdx, mdy, mdz, mdistance = (
                vector_error(
                    motion_prediction,
                    actual,
                )
            )

            motion_dx.append(mdx)
            motion_dy.append(mdy)
            motion_dz.append(mdz)
            motion_distances.append(mdistance)

            evaluated_points += 1

        evaluated_windows += 1

    hold_rmse_3d = rmse(hold_distances)
    motion_rmse_3d = rmse(motion_distances)

    improvement_percent = None

    if (
        hold_rmse_3d is not None
        and motion_rmse_3d is not None
        and hold_rmse_3d > 0
    ):
        improvement_percent = (
            (
                hold_rmse_3d
                - motion_rmse_3d
            )
            / hold_rmse_3d
            * 100.0
        )

    return {
        "gap_length": gap_length,
        "evaluated_windows": evaluated_windows,
        "evaluated_points": evaluated_points,
        "hold_last": {
            "dx_rmse": rmse(hold_dx),
            "dy_rmse": rmse(hold_dy),
            "dz_rmse": rmse(hold_dz),
            "distance_mean": mean(
                hold_distances
            ),
            "distance_rmse": hold_rmse_3d,
        },
        "motion_prediction": {
            "dx_rmse": rmse(motion_dx),
            "dy_rmse": rmse(motion_dy),
            "dz_rmse": rmse(motion_dz),
            "distance_mean": mean(
                motion_distances
            ),
            "distance_rmse": motion_rmse_3d,
        },
        "improvement_percent": (
            improvement_percent
        ),
        "motion_is_better": (
            improvement_percent is not None
            and improvement_percent > 0
        ),
    }


def discover_csv_files() -> list[Path]:
    paths: list[Path] = []

    for directory in SEARCH_DIRECTORIES:
        if not directory.is_dir():
            continue

        paths.extend(
            directory.rglob("*.csv")
        )

    unique_paths = sorted(
        set(path.resolve() for path in paths)
    )

    return unique_paths


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_paths = discover_csv_files()

    print("=" * 80)
    print("TASK 2 HAREKET TAHMİNİ RMSE TESTİ")
    print("=" * 80)
    print(f"Bulunan CSV: {len(csv_paths)}")
    print()

    file_reports: list[dict[str, Any]] = []

    for path in csv_paths:
        sequence = load_translation_csv(path)

        if len(sequence) < 10:
            print(
                f"[ATLANDI] {path.name}: "
                "uygun translation verisi yok."
            )
            continue

        print("-" * 80)
        print(f"CSV: {path}")
        print(f"Frame sayısı: {len(sequence)}")

        evaluations = []

        for gap_length in GAP_LENGTHS:
            result = evaluate_gap(
                sequence,
                gap_length,
            )
            evaluations.append(result)

            hold_value = (
                result["hold_last"][
                    "distance_rmse"
                ]
            )
            motion_value = (
                result["motion_prediction"][
                    "distance_rmse"
                ]
            )
            improvement = result[
                "improvement_percent"
            ]

            print()
            print(
                f"Gap={gap_length} frame | "
                f"nokta={result['evaluated_points']}"
            )
            print(
                f"  Hold-last RMSE : "
                f"{hold_value}"
            )
            print(
                f"  Motion RMSE    : "
                f"{motion_value}"
            )
            print(
                f"  İyileşme       : "
                f"{improvement}%"
            )

        file_reports.append(
            {
                "path": str(path),
                "frame_count": len(sequence),
                "first_frame": sequence[0][0],
                "last_frame": sequence[-1][0],
                "evaluations": evaluations,
            }
        )

    report = {
        "configuration": {
            "gap_lengths": list(GAP_LENGTHS),
            "velocity_damping": (
                VELOCITY_DAMPING
            ),
            "maximum_translation_step": (
                MAX_TRANSLATION_STEP
            ),
        },
        "files": file_reports,
    }

    json_path = (
        OUTPUT_DIR
        / "task2_motion_rmse_report.json"
    )
    text_path = (
        OUTPUT_DIR
        / "task2_motion_rmse_report.txt"
    )

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 80,
        "TASK 2 HAREKET TAHMİNİ RMSE ÖZETİ",
        "=" * 80,
        "",
        (
            f"Damping: {VELOCITY_DAMPING}"
        ),
        (
            "Maksimum frame adımı: "
            f"{MAX_TRANSLATION_STEP}"
        ),
        "",
    ]

    for file_report in file_reports:
        lines.append(
            f"CSV: {file_report['path']}"
        )
        lines.append(
            f"Frame: {file_report['frame_count']}"
        )

        for evaluation in file_report[
            "evaluations"
        ]:
            lines.append(
                "  Gap "
                f"{evaluation['gap_length']}: "
                "hold="
                f"{evaluation['hold_last']['distance_rmse']} | "
                "motion="
                f"{evaluation['motion_prediction']['distance_rmse']} | "
                "iyileşme="
                f"{evaluation['improvement_percent']}%"
            )

        lines.append("")

    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RMSE TESTİ TAMAMLANDI")
    print("=" * 80)
    print(f"İşlenen CSV: {len(file_reports)}")
    print(f"Rapor: {text_path}")
    print()
    print("Üretim kodu değiştirilmedi.")
    print("CSV dosyaları değiştirilmedi.")
    print("Sunucuya bağlanılmadı.")


if __name__ == "__main__":
    main()