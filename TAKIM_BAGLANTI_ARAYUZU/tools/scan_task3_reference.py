from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src_custom.reference_matcher import (
    _bbox_is_safe,
    _clip_bbox,
    _preprocess_gray,
)


SESSION_DIRECTORY = (
    PROJECT_ROOT
    / "_images"
    / "THYZ_2026_Online_Yarisma_Test_Oturumu"
)

REFERENCES_JSON = SESSION_DIRECTORY / "references.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "tests"
    / "task3_real_reference_scan"
)

TOP_K_PER_REFERENCE = 15


def extract_frame_index(url: str) -> int:
    filename = Path(url).name
    return int(Path(filename).stem.split("_")[-1])


def local_path_from_url(url: str) -> Path:
    """
    JSON URL:
    /THYZ_2026_Online_Yarisma_Test_Oturumu/frame_000161.webp

    Yerel:
    _images/THYZ_2026_Online_Yarisma_Test_Oturumu/frame_000161.webp
    """
    return PROJECT_ROOT / "_images" / url.lstrip("/")


def calculate_best_template_match(
    frame_image,
    reference_image,
):
    frame_gray = _preprocess_gray(frame_image)
    reference_gray = _preprocess_gray(reference_image)

    frame_height, frame_width = frame_gray.shape[:2]
    reference_height, reference_width = reference_gray.shape[:2]

    best_result = None

    for scale in np.linspace(0.10, 1.20, 56):
        resized_width = int(reference_width * scale)
        resized_height = int(reference_height * scale)

        # Küçük ve güvenilmez eşleşmeleri engelle.
        if resized_width < 36 or resized_height < 20:
            continue

        if (
            resized_width >= frame_width
            or resized_height >= frame_height
        ):
            continue

        resized_reference = cv2.resize(
            reference_gray,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        try:
            result = cv2.matchTemplate(
                frame_gray,
                resized_reference,
                cv2.TM_CCOEFF_NORMED,
            )
        except cv2.error:
            continue

        _, score, _, location = cv2.minMaxLoc(result)

        x1, y1 = location
        x2 = x1 + resized_width
        y2 = y1 + resized_height

        bbox = _clip_bbox(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
        )

        if not _bbox_is_safe(bbox, frame_image.shape):
            continue

        candidate = {
            "score": float(score),
            "scale": float(scale),
            "bbox": bbox,
        }

        if (
            best_result is None
            or candidate["score"] > best_result["score"]
        ):
            best_result = candidate

    return best_result


def draw_result(
    frame_image,
    result,
    reference_order,
    frame_name,
):
    output = frame_image.copy()

    x1, y1, x2, y2 = result["bbox"]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
    )

    text = (
        f"REF {reference_order} | "
        f"{frame_name} | "
        f"score={result['score']:.4f} | "
        f"scale={result['scale']:.3f}"
    )

    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def scan_reference(reference: dict) -> None:
    order = int(reference["order"])

    start_index = extract_frame_index(
        reference["frame_start_image_url"]
    )
    end_index = extract_frame_index(
        reference["frame_end_image_url"]
    )

    reference_path = local_path_from_url(
        reference["image_url"]
    )

    reference_image = cv2.imread(
        str(reference_path)
    )

    if reference_image is None:
        print(
            f"[REF {order}] Referans okunamadı: "
            f"{reference_path}"
        )
        return

    output_directory = OUTPUT_ROOT / f"reference_{order}"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print()
    print("=" * 78)
    print(
        f"REFERENCE {order}: "
        f"{start_index:06d} → {end_index:06d}"
    )
    print(f"Image: {reference_path}")
    print("=" * 78)

    for frame_index in range(
        start_index,
        end_index + 1,
    ):
        frame_path = (
            SESSION_DIRECTORY
            / f"frame_{frame_index:06d}.webp"
        )

        if not frame_path.exists():
            continue

        frame_image = cv2.imread(
            str(frame_path)
        )

        if frame_image is None:
            continue

        result = calculate_best_template_match(
            frame_image,
            reference_image,
        )

        if result is None:
            continue

        results.append(
            {
                "frame_index": frame_index,
                "frame_name": frame_path.name,
                "frame_image": frame_image,
                **result,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if not results:
        print("Hiç aday üretilemedi.")
        return

    print("En yüksek adaylar:")

    for rank, result in enumerate(
        results[:TOP_K_PER_REFERENCE],
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{result['frame_name']} "
            f"score={result['score']:.4f} "
            f"scale={result['scale']:.3f} "
            f"bbox={result['bbox']}"
        )

        output_image = draw_result(
            result["frame_image"],
            result,
            order,
            result["frame_name"],
        )

        output_path = (
            output_directory
            / (
                f"{rank:02d}_"
                f"{result['score']:.4f}_"
                f"{result['frame_name']}"
            )
        )

        cv2.imwrite(
            str(output_path),
            output_image,
        )


def main():
    if not REFERENCES_JSON.exists():
        raise FileNotFoundError(
            f"references.json bulunamadı: {REFERENCES_JSON}"
        )

    with REFERENCES_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        references = json.load(file)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Referans sayısı: {len(references)}")
    print(f"Çıktı klasörü: {OUTPUT_ROOT}")

    for reference in references:
        scan_reference(reference)

    print()
    print("=" * 78)
    print("GERÇEK TASK 3 TARAMASI TAMAMLANDI")
    print("=" * 78)
    print(f"Sonuçlar: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()