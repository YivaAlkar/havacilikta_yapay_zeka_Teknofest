from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

SESSION_DIRECTORY = (
    PROJECT_ROOT
    / "_images"
    / "THYZ_2026_Online_Yarisma_Test_Oturumu"
)

REFERENCE_PATH = (
    SESSION_DIRECTORY
    / "references"
    / "reference_1.webp"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "tests"
    / "task3_yolo_candidates_ref1"
)

START_FRAME = 161
END_FRAME = 205

YOLO_CONFIDENCE = 0.12
TOP_K = 20


def clip_bbox(
    bbox,
    width: int,
    height: int,
):
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return x1, y1, x2, y2


def add_margin(
    bbox,
    width: int,
    height: int,
    margin_ratio: float = 0.12,
):
    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    margin_x = int(bbox_width * margin_ratio)
    margin_y = int(bbox_height * margin_ratio)

    return clip_bbox(
        (
            x1 - margin_x,
            y1 - margin_y,
            x2 + margin_x,
            y2 + margin_y,
        ),
        width,
        height,
    )


def largest_vehicle_crop(
    model,
    image,
):
    results = model.predict(
        source=image,
        conf=0.05,
        verbose=False,
    )

    best_bbox = None
    best_area = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())

            # Yarışma sınıf sırası:
            # 0 = Tasit
            if class_id != 0:
                continue

            bbox = tuple(
                map(
                    int,
                    box.xyxy[0].tolist(),
                )
            )

            x1, y1, x2, y2 = bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)

            if area > best_area:
                best_area = area
                best_bbox = bbox

    if best_bbox is None:
        return image, None

    height, width = image.shape[:2]

    best_bbox = add_margin(
        best_bbox,
        width,
        height,
        margin_ratio=0.08,
    )

    x1, y1, x2, y2 = best_bbox

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return image, None

    return crop, best_bbox


def preprocess_gray(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.equalizeHist(gray)

    return gray

def hsv_color_similarity(
    reference_image,
    candidate_image,
):
    """
    HSV renk histogramları üzerinden 0.0-1.0 arası benzerlik üretir.
    Referans sarı/bej biçerdöver olduğu için gri otomobilleri ayırmaya yardım eder.
    """
    if (
        reference_image is None
        or candidate_image is None
        or reference_image.size == 0
        or candidate_image.size == 0
    ):
        return 0.0

    reference_hsv = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2HSV,
    )
    candidate_hsv = cv2.cvtColor(
        candidate_image,
        cv2.COLOR_BGR2HSV,
    )

    reference_hist = cv2.calcHist(
        [reference_hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )

    candidate_hist = cv2.calcHist(
        [candidate_hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        reference_hist,
        reference_hist,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
    )

    cv2.normalize(
        candidate_hist,
        candidate_hist,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
    )

    correlation = cv2.compareHist(
        reference_hist,
        candidate_hist,
        cv2.HISTCMP_CORREL,
    )

    # Korelasyon teorik olarak negatif olabilir.
    return float(
        max(0.0, min(1.0, correlation))
    )




def sift_similarity(
    reference_crop,
    candidate_crop,
):
    reference_gray = preprocess_gray(
        reference_crop
    )
    candidate_gray = preprocess_gray(
        candidate_crop
    )

    sift = cv2.SIFT_create(
        nfeatures=2000,
    )

    reference_keypoints, reference_descriptors = (
        sift.detectAndCompute(
            reference_gray,
            None,
        )
    )

    candidate_keypoints, candidate_descriptors = (
        sift.detectAndCompute(
            candidate_gray,
            None,
        )
    )

    if (
        reference_descriptors is None
        or candidate_descriptors is None
        or len(reference_keypoints) < 6
        or len(candidate_keypoints) < 6
    ):
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    try:
        pairs = matcher.knnMatch(
            reference_descriptors,
            candidate_descriptors,
            k=2,
        )
    except cv2.error:
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    good_matches = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 4:
        return {
            "score": float(len(good_matches)),
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    source_points = np.float32(
        [
            reference_keypoints[
                match.queryIdx
            ].pt
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            candidate_keypoints[
                match.trainIdx
            ].pt
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    if homography is None or mask is None:
        return {
            "score": float(len(good_matches)),
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    inliers = int(mask.ravel().sum())

    inlier_ratio = (
        inliers
        / max(1, len(good_matches))
    )

    # İyi eşleşme ve geometrik tutarlılığı birlikte puanla.
    score = (
        inliers * 10.0
        + len(good_matches) * 2.0
        + inlier_ratio * 20.0
    )

    return {
        "score": float(score),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
    }


def draw_candidate(
    image,
    bbox,
    information,
    frame_name,
):
    output = image.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        5,
    )

    text = (
        f"{frame_name} "
        f"combined={information['combined_score']:.1f} "
        f"sift={information['score']:.1f} "
        f"yolo={information['confidence']:.2f} "
        f"color={information['color_similarity']:.2f} "
        f"inliers={information['inliers']} "
        f"ratio={information['inlier_ratio']:.2f}"
    )

    cv2.putText(
        output,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return output


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )

    reference_image = cv2.imread(
        str(REFERENCE_PATH)
    )

    if reference_image is None:
        raise RuntimeError(
            f"Referans okunamadı: {REFERENCE_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    # Referans görseli zaten yarışma sunucusunun verdiği hedef nesnedir.
    # YOLO yalnız tekerlek gibi küçük bir parçayı seçebildiği için
    # referans tarafında crop uygulamıyoruz.
    reference_crop = reference_image.copy()
    reference_bbox = None

    print(
        "Referans YOLO bbox: kullanılmadı"
    )
    print(
        "Referans tam görüntü shape:",
        reference_crop.shape,
    )

    reference_height, reference_width = (
    reference_crop.shape[:2]
    )

    reference_aspect_ratio = (
        reference_width
        / max(1, reference_height)
    )

    print(
        "Referans aspect ratio:",
        reference_aspect_ratio,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIRECTORY
            / "reference_vehicle_crop.jpg"
        ),
        reference_crop,
    )

    candidates = []

    for frame_index in range(
        START_FRAME,
        END_FRAME + 1,
    ):
        frame_path = (
            SESSION_DIRECTORY
            / f"frame_{frame_index:06d}.webp"
        )

        frame_image = cv2.imread(
            str(frame_path)
        )

        if frame_image is None:
            continue

        results = model.predict(
            source=frame_image,
            imgsz=1280,
            conf=0.01,
            iou=0.50,
            max_det=100,
            verbose=False,
        )

        height, width = frame_image.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            for box_index, box in enumerate(
                result.boxes
            ):
                class_id = int(
                    box.cls.item()
                )

                if class_id != 0:
                    continue

                confidence = float(
                    box.conf.item()
                )

                # Tanı sırasında 0.01 kullandık ancak final aday seçiminde
                # düşük güvenli kutular false positive üretiyor.
                if confidence < 0.75:
                    continue

                bbox = clip_bbox(
                    box.xyxy[0].tolist(),
                    width,
                    height,
                )

                bbox = add_margin(
                    bbox,
                    width,
                    height,
                    margin_ratio=0.20,
                )

                x1, y1, x2, y2 = bbox

                candidate_width = x2 - x1
                candidate_height = y2 - y1

                if candidate_width <= 0 or candidate_height <= 0:
                    continue

                candidate_area = (
                    candidate_width
                    * candidate_height
                )

                frame_area = width * height

                candidate_area_ratio = (
                    candidate_area
                    / max(1, frame_area)
                )

                # Minicik yanlış YOLO kutularını ele.
                # Gerçek biçerdöver frame alanının yaklaşık %1–2'sini kaplıyor.
                if candidate_area_ratio < 0.008:
                    continue

                # Aşırı büyük, anlamsız kutuları da ele.
                if candidate_area_ratio > 0.12:
                    continue

                candidate_aspect_ratio = (
                    candidate_width
                    / candidate_height
                )

                aspect_similarity = min(
                    candidate_aspect_ratio,
                    reference_aspect_ratio,
                ) / max(
                    candidate_aspect_ratio,
                    reference_aspect_ratio,
                )

                # Referans yatay biçerdöver.
                # Dikey otomobil kutularını SIFT'e göndermiyoruz.
                if aspect_similarity < 0.60:
                    continue

                crop = frame_image[
                    y1:y2,
                    x1:x2,
                ]

                if crop.size == 0:
                    continue

                similarity = sift_similarity(
                reference_crop,
                crop,
                )

                if similarity["good_matches"] < 30:
                    continue

                if similarity["inliers"] < 18:
                    continue

                if similarity["inlier_ratio"] < 0.50:
                    continue

                color_similarity = hsv_color_similarity(
                    reference_crop,
                    crop,
                )

                if color_similarity < 0.60:
                    continue

                combined_score = (
                    similarity["score"]
                    * aspect_similarity
                    * confidence
                    * (
                        0.35
                        + 0.65 * color_similarity
                    )
                )

                candidate = {
                    "frame_index": frame_index,
                    "frame_name": frame_path.name,
                    "box_index": box_index,
                    "bbox": bbox,
                    "confidence": confidence,
                    "frame_image": frame_image,
                    "candidate_area_ratio": candidate_area_ratio,
                    "aspect_similarity": aspect_similarity,
                    "color_similarity": color_similarity,
                    "combined_score": combined_score,
                    **similarity,
                }

                candidates.append(candidate)

                print(
                    f"{frame_path.name} "
                    f"bbox={bbox} "
                    f"yolo={confidence:.3f} "
                    f"sift={similarity['score']:.1f} "
                    f"good={similarity['good_matches']} "
                    f"inliers={similarity['inliers']} "
                    f"ratio={similarity['inlier_ratio']:.2f}"
                )

    candidates.sort(
        key=lambda item: (
            item["combined_score"],
            item["confidence"],
        ),
        reverse=True,
    )

    print()
    print("=" * 78)
    print("TOP YOLO + SIFT TASK 3 CANDIDATES")
    print("=" * 78)

    for rank, candidate in enumerate(
        candidates[:TOP_K],
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{candidate['frame_name']} "
            f"combined={candidate['combined_score']:.1f} "
            f"sift={candidate['score']:.1f} "
            f"aspect={candidate['aspect_similarity']:.2f} "
            f"color={candidate['color_similarity']:.2f} "
            f"yolo={candidate['confidence']:.3f} "
            f"inliers={candidate['inliers']} "
            f"ratio={candidate['inlier_ratio']:.2f} "
            f"bbox={candidate['bbox']}"
        )

        output = draw_candidate(
            candidate["frame_image"],
            candidate["bbox"],
            candidate,
            candidate["frame_name"],
        )

        output_path = (
            OUTPUT_DIRECTORY
            / (
                f"{rank:02d}_"
                f"score_{candidate['score']:.1f}_"
                f"{candidate['frame_name']}.jpg"
            )
        )

        cv2.imwrite(
            str(output_path),
            output,
        )

    print()
    print(
        "Sonuç klasörü:",
        OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()
