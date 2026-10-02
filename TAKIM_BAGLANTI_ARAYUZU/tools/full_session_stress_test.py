from __future__ import annotations
from collections import defaultdict

import argparse
import json
import time
import traceback
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


SESSION_DIR = (
    ROOT
    / "_images"
    / "THYZ_2026_Online_Yarisma_Test_Oturumu"
)

REFERENCES_JSON = SESSION_DIR / "references.json"
REFERENCES_DIR = SESSION_DIR / "references"
OUTPUT_DIR = ROOT / "outputs" / "tests"
REPORT_PATH = OUTPUT_DIR / "full_session_stress_report.txt"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 bütün frame'ler; pozitif sayı ilk N frame.",
    )

    parser.add_argument(
        "--progress",
        type=int,
        default=250,
        help="Kaç frame'de bir ilerleme yazdırılacağı.",
    )
    
    parser.add_argument(
        "--disable-task3",
        action="store_true",
        help="Task 3 referans eşleştirmesini kapatır.",
    )

    return parser.parse_args()


def get_frame_index(frame_path: Path) -> int:
    return int(frame_path.stem.split("_")[-1])


def load_references():
    with REFERENCES_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        references = json.load(file)

    reference_paths = {}

    for reference in references:
        reference_url = reference.get("url")
        order = reference.get("order")

        if not reference_url or order is None:
            continue

        reference_path = (
            REFERENCES_DIR
            / f"reference_{int(order)}.webp"
        )

        reference_paths[reference_url] = str(
            reference_path
        )

    return references, reference_paths


def create_prediction(
    frame_path: Path,
    frame_index: int,
):
    return FramePredictions(
        frame_url=(
            f"http://offline.test/frames/"
            f"{frame_index}/"
        ),
        image_url=(
            f"/THYZ_2026_Online_Yarisma_Test_Oturumu/"
            f"{frame_path.name}"
        ),
        video_name="FULL_SESSION_STRESS_TEST",
        gt_translation_x=float(frame_index),
        gt_translation_y=float(frame_index) * 0.5,
        gt_translation_z=100.0,
    )


def cuda_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024
        / 1024
    )

    return allocated, reserved


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = sorted(
        SESSION_DIR.glob("frame_*.webp")
    )

    if args.limit > 0:
        frames = frames[: args.limit]

    if not frames:
        raise RuntimeError(
            f"Frame bulunamadı: {SESSION_DIR}"
        )

    references, reference_paths = load_references()

    reference_meta_by_url = {
        reference["url"]: {
            "order": int(reference["order"]),
            "start": reference["frame_start_image_url"].split("/")[-1],
            "end": reference["frame_end_image_url"].split("/")[-1],
        }
        for reference in references
        if reference.get("url")
        and reference.get("order") is not None
        and reference.get("frame_start_image_url")
        and reference.get("frame_end_image_url")
    }
    print("=" * 80)
    print("FULL SESSION OFFLINE STRESS TEST")
    print("=" * 80)
    print(f"Frame sayısı     : {len(frames)}")
    print(f"Referans sayısı  : {len(references)}")
    print(f"CUDA             : {torch.cuda.is_available()}")
    print(f"Rapor            : {REPORT_PATH}")
    print("=" * 80)

    model = ObjectDetectionModel(
        "http://offline.test/"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()

    processed = 0
    failed = 0
    total_objects = 0
    total_translations = 0
    total_reference_predictions = 0

    reference_counts = defaultdict(int)
    reference_first_frame = {}
    reference_last_frame = {}
    reference_out_of_range = defaultdict(int)
    unknown_reference_urls = set()

    slowest_frame_seconds = 0.0
    slowest_frame_name = None
    errors = []

    for frame_path in frames:
        frame_started_at = time.perf_counter()
        frame_index = get_frame_index(frame_path)

        try:
            prediction = create_prediction(
                frame_path=frame_path,
                frame_index=frame_index,
            )

            result = model.detect(
                prediction=prediction,
                health_status="1",
                active_refs=[] if args.disable_task3 else references,
                ref_image_paths=reference_paths,
                frame_image_path=str(frame_path),
            )

            objects = getattr(
                result,
                "detected_objects",
                [],
            )

            translations = getattr(
                result,
                "translations",
                [],
            )

            reference_predictions = getattr(
                result,
                "reference_predictions",
                [],
            )

            total_objects += len(objects)
            total_translations += len(translations)
            total_reference_predictions += len(
                reference_predictions
            )

            for reference_prediction in reference_predictions:
                reference_url = getattr(
                    reference_prediction,
                    "reference_url",
                    None,
                )

                reference_meta = reference_meta_by_url.get(
                    reference_url
                )

                if reference_meta is None:
                    unknown_reference_urls.add(
                        str(reference_url)
                    )
                    continue

                reference_order = reference_meta["order"]
                frame_name = frame_path.name

                reference_counts[reference_order] += 1

                reference_first_frame.setdefault(
                    reference_order,
                    frame_name,
                )
                reference_last_frame[reference_order] = frame_name

                if not (
                    reference_meta["start"]
                    <= frame_name
                    <= reference_meta["end"]
                ):
                    reference_out_of_range[reference_order] += 1

            processed += 1

        except Exception as error:
            failed += 1

            error_message = (
                f"{frame_path.name}: "
                f"{type(error).__name__}: {error}"
            )

            errors.append(error_message)

            print(f"[ERROR] {error_message}")
            traceback.print_exc()

        frame_seconds = (
            time.perf_counter()
            - frame_started_at
        )

        if frame_seconds > slowest_frame_seconds:
            slowest_frame_seconds = frame_seconds
            slowest_frame_name = frame_path.name

        completed = processed + failed

        if (
            completed % args.progress == 0
            or completed == len(frames)
        ):
            elapsed = (
                time.perf_counter()
                - started_at
            )

            fps = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            allocated_mb, reserved_mb = (
                cuda_memory_mb()
            )

            print(
                f"[PROGRESS] {completed}/{len(frames)} "
                f"failed={failed} "
                f"fps={fps:.2f} "
                f"gpu_alloc={allocated_mb:.0f} MB "
                f"gpu_reserved={reserved_mb:.0f} MB"
            )

    total_seconds = (
        time.perf_counter()
        - started_at
    )

    total_attempted = processed + failed

    average_fps = (
        total_attempted / total_seconds
        if total_seconds > 0
        else 0.0
    )

    allocated_mb, reserved_mb = cuda_memory_mb()

    if torch.cuda.is_available():
        peak_allocated_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved()
            / 1024
            / 1024
        )
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    reference_detail_lines = [
        "",
        "REFERENCE PREDICTION DISTRIBUTION",
        "=" * 80,
    ]

    for reference in sorted(
        references,
        key=lambda item: int(item.get("order", 0)),
    ):
        reference_order = int(reference["order"])
        expected_start = reference[
            "frame_start_image_url"
        ].split("/")[-1]
        expected_end = reference[
            "frame_end_image_url"
        ].split("/")[-1]

        count = reference_counts[reference_order]
        first_frame = reference_first_frame.get(
            reference_order,
            "NONE",
        )
        last_frame = reference_last_frame.get(
            reference_order,
            "NONE",
        )
        outside_count = reference_out_of_range[
            reference_order
        ]

        reference_detail_lines.append(
            f"Ref {reference_order}: "
            f"count={count}, "
            f"first={first_frame}, "
            f"last={last_frame}, "
            f"expected={expected_start}..{expected_end}, "
            f"out_of_range={outside_count}"
        )

    reference_detail_lines.append(
        f"Unknown reference URL count: "
        f"{len(unknown_reference_urls)}"
    )

    report_lines = [
        "=" * 80,
        "FULL SESSION OFFLINE STRESS TEST REPORT",
        "=" * 80,
        f"Toplam frame              : {len(frames)}",
        f"Başarılı işlenen          : {processed}",
        f"Failed                    : {failed}",
        f"Toplam süre               : {total_seconds:.2f} saniye",
        f"Ortalama FPS              : {average_fps:.2f}",
        f"Toplam nesne              : {total_objects}",
        f"Toplam translation        : {total_translations}",
        (
            "Toplam referans prediction: "
            f"{total_reference_predictions}"
        ),
        (
            "En yavaş frame            : "
            f"{slowest_frame_name}"
        ),
        (
            "En yavaş frame süresi     : "
            f"{slowest_frame_seconds:.3f} saniye"
        ),
        (
            "GPU final allocated       : "
            f"{allocated_mb:.2f} MB"
        ),
        (
            "GPU final reserved        : "
            f"{reserved_mb:.2f} MB"
        ),
        (
            "GPU peak allocated        : "
            f"{peak_allocated_mb:.2f} MB"
        ),
        (
            "GPU peak reserved         : "
            f"{peak_reserved_mb:.2f} MB"
        ),
        "=" * 80,
    ]
    
    report_lines.extend(reference_detail_lines)

    if errors:
        report_lines.append("HATALAR:")

        for error in errors:
            report_lines.append(error)

        report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(report)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()