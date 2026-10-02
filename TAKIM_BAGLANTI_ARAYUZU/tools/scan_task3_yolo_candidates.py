from __future__ import annotations

import argparse
import json
import re
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

REFERENCES_JSON_PATH = SESSION_DIRECTORY / "references.json"

REFERENCE_PATH: Path | None = None
OUTPUT_DIRECTORY: Path | None = None

START_FRAME = 0
END_FRAME = 0

REFERENCE_NUMBER = 1

# Referans 1 iÃ§in 0.60 Ã§ok iyi Ã§alÄ±ÅŸtÄ±.
# DiÄŸer referanslar ilk taramada renk filtresi olmadan incelenecek.
MIN_COLOR_SIMILARITY = 0.60

def extract_frame_number(frame_url: str) -> int:
    match = re.search(r"frame_(\d+)", str(frame_url))

    if match is None:
        raise ValueError(
            f"Frame numarasi URL icinden okunamadi: {frame_url}"
        )

    return int(match.group(1))


def configure_reference(reference_number: int) -> None:
    global REFERENCE_PATH
    global OUTPUT_DIRECTORY
    global START_FRAME
    global END_FRAME
    global REFERENCE_NUMBER
    global MIN_COLOR_SIMILARITY

    if reference_number not in {1, 2, 3, 4, 5}:
        raise ValueError(
            f"Referans numarasi 1-5 arasinda olmali: {reference_number}"
        )

    if not REFERENCES_JSON_PATH.exists():
        raise FileNotFoundError(
            f"references.json bulunamadi: {REFERENCES_JSON_PATH}"
        )

    with REFERENCES_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        references_data = json.load(file)

    if isinstance(references_data, dict):
        references = references_data.get(
            "references",
            references_data.get("data", []),
        )
    else:
        references = references_data

    selected_reference = None

    for reference in references:
        try:
            order = int(reference.get("order"))
        except (TypeError, ValueError):
            continue

        if order == reference_number:
            selected_reference = reference
            break

    if selected_reference is None:
        raise ValueError(
            f"references.json icinde order={reference_number} bulunamadi."
        )

    image_url = selected_reference.get("image_url")
    frame_start_url = selected_reference.get("frame_start_image_url")
    frame_end_url = selected_reference.get("frame_end_image_url")

    if not image_url:
        raise ValueError(
            f"Referans {reference_number} image_url icermiyor."
        )

    if not frame_start_url or not frame_end_url:
        raise ValueError(
            f"Referans {reference_number} frame araligi eksik."
        )

    reference_filename = Path(str(image_url)).name

    REFERENCE_NUMBER = reference_number

    REFERENCE_PATH = (
        SESSION_DIRECTORY
        / "references"
        / reference_filename
    )

    OUTPUT_DIRECTORY = (
        PROJECT_ROOT
        / "outputs"
        / "tests"
        / f"task3_yolo_candidates_ref{reference_number}"
    )

    START_FRAME = extract_frame_number(frame_start_url)
    END_FRAME = extract_frame_number(frame_end_url)

    # Referans 1 iÃ§in doÄŸrulanmÄ±ÅŸ eÅŸik.
    # DiÄŸer referanslar Ã¶nce filtresiz taranacak.
    if reference_number == 1:
        MIN_COLOR_SIMILARITY = 0.60
        

    elif reference_number == 2:
        MIN_COLOR_SIMILARITY = 0.35

    else:
        MIN_COLOR_SIMILARITY = 0.0

    print("=" * 72)
    print(f"TASK 3 REFERENCE CONFIGURATION")
    print("=" * 72)
    print(f"Referans numarasi : {REFERENCE_NUMBER}")
    print(f"Referans dosyasi  : {REFERENCE_PATH}")
    print(f"Baslangic frame   : {START_FRAME}")
    print(f"Bitis frame       : {END_FRAME}")
    print(f"Renk alt siniri   : {MIN_COLOR_SIMILARITY:.2f}")
    print(f"Sonuc klasoru     : {OUTPUT_DIRECTORY}")
    print("=" * 72)

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

            # YarÄ±ÅŸma sÄ±nÄ±f sÄ±rasÄ±:
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
    HSV renk histogramlarÄ± Ã¼zerinden 0.0-1.0 arasÄ± benzerlik Ã¼retir.
    Referans sarÄ±/bej biÃ§erdÃ¶ver olduÄŸu iÃ§in gri otomobilleri ayÄ±rmaya yardÄ±m eder.
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

def yellow_vehicle_ratio(image):
    """
    RGB karede sarı/bej iş makinesi piksel oranını hesaplar.
    0.0-1.0 arasında değer döndürür.
    """
    if image is None or image.size == 0:
        return 0.0

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lower_yellow = np.array(
        [12, 55, 55],
        dtype=np.uint8,
    )

    upper_yellow = np.array(
        [42, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]

    return float(
        yellow_pixels / max(1, total_pixels)
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

    # Ä°yi eÅŸleÅŸme ve geometrik tutarlÄ±lÄ±ÄŸÄ± birlikte puanla.
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


def sift_localize_in_frame(
    reference_image,
    frame_image,
):
    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY,
    )

    frame_gray = cv2.cvtColor(
        frame_image,
        cv2.COLOR_BGR2GRAY,
    )

    sift = cv2.SIFT_create(
        nfeatures=5000,
    )

    reference_keypoints, reference_descriptors = (
        sift.detectAndCompute(
            reference_gray,
            None,
        )
    )

    frame_keypoints, frame_descriptors = (
        sift.detectAndCompute(
            frame_gray,
            None,
        )
    )

    empty_result = {
        "bbox": None,
        "score": 0.0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reason": "unknown",
    }

    if (
        reference_descriptors is None
        or frame_descriptors is None
        or len(reference_keypoints) < 4
        or len(frame_keypoints) < 4
    ):
        return empty_result

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        reference_descriptors,
        frame_descriptors,
        k=2,
    )

    good_matches = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        lowe_ratio = (
            0.72
            if REFERENCE_NUMBER == 3
            else 0.78
        )

        if first.distance < lowe_ratio * second.distance:

            good_matches.append(first)

    minimum_good_matches = (
        8
        if REFERENCE_NUMBER == 3
        else 5
    )

    if len(good_matches) < minimum_good_matches:
        empty_result["good_matches"] = len(good_matches)
        empty_result["reason"] = (
            f"good_matches_below_{minimum_good_matches}"
        )
        return empty_result

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
            frame_keypoints[
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
        empty_result["good_matches"] = len(good_matches)
        return empty_result

    inliers = int(mask.ravel().sum())

    inlier_ratio = (
        inliers
        / max(1, len(good_matches))
    )

    # Referans 3 uzaktaki ince futbol kalesi olduÄŸu iÃ§in
    # saÄŸlam homography bazen 6-7 inlier ile oluÅŸabiliyor.
    # DiÄŸer referanslarÄ±n mevcut gÃ¼venli eÅŸiÄŸini deÄŸiÅŸtirme.
    if REFERENCE_NUMBER == 3:
        minimum_inliers = 6
        minimum_inlier_ratio = 0.35
    elif REFERENCE_NUMBER in (4, 5):
        minimum_inliers = 4
        minimum_inlier_ratio = 0.25
    else:
        minimum_inliers = 8
        minimum_inlier_ratio = 0.35

    if (
        inliers < minimum_inliers
        or inlier_ratio < minimum_inlier_ratio
    ):
        return {
            "bbox": None,
            "score": 0.0,
            "good_matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": float(inlier_ratio),
            "reason": (
                f"inlier_filter_min_"
                f"{minimum_inliers}"
            ),
        }

    reference_height, reference_width = (
        reference_image.shape[:2]
    )

    reference_corners = np.float32(
        [
            [0, 0],
            [reference_width - 1, 0],
            [reference_width - 1, reference_height - 1],
            [0, reference_height - 1],
        ]
    ).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(
        reference_corners,
        homography,
    ).reshape(-1, 2)

    if not np.isfinite(transformed_corners).all():
        return empty_result

    x_values = transformed_corners[:, 0]
    y_values = transformed_corners[:, 1]

    frame_height, frame_width = frame_image.shape[:2]

    projected_bbox = clip_bbox(
        (
            int(np.floor(x_values.min())),
            int(np.floor(y_values.min())),
            int(np.ceil(x_values.max())),
            int(np.ceil(y_values.max())),
        ),
        frame_width,
        frame_height,
    )

    # Referans 3'te bÃ¼tÃ¼n referans kÃ¶ÅŸelerini homography ile dÃ¶nÃ¼ÅŸtÃ¼rmek,
    # kalenin Ã§ok kÃ¼Ã§Ã¼k gÃ¶rÃ¼nmesi nedeniyle aÅŸÄ±rÄ± bÃ¼yÃ¼k ve hatalÄ± kutular
    # oluÅŸturabiliyor. Bu nedenle yalnÄ±zca RANSAC tarafÄ±ndan doÄŸrulanan
    # inlier eÅŸleÅŸmelerinin frame Ã¼zerindeki daÄŸÄ±lÄ±mÄ±nÄ± kullan.
    if REFERENCE_NUMBER == 3:
        inlier_mask = mask.ravel().astype(bool)

        inlier_destination_points = (
            destination_points
            .reshape(-1, 2)[inlier_mask]
        )

        if len(inlier_destination_points) < minimum_inliers:
            return {
                "bbox": None,
                "score": 0.0,
                "good_matches": len(good_matches),
                "inliers": inliers,
                "inlier_ratio": float(inlier_ratio),
                "reason": "insufficient_inlier_points",
            }

        # RANSAC inlier noktalarÄ±nÄ±n bazÄ±larÄ± kare Ã¼zerinde birbirinden Ã§ok uzak
        # olabiliyor. Referans 3 iÃ§in birbirine yakÄ±n noktalarÄ±n en bÃ¼yÃ¼k
        # yerel kÃ¼mesini bul ve bbox'Ä± yalnÄ±z bu kÃ¼meden Ã¼ret.
        frame_diagonal = float(
            np.hypot(frame_width, frame_height)
        )
        cluster_radius = max(45.0, frame_diagonal * 0.055)

        largest_cluster = None

        for center_point in inlier_destination_points:
            distances = np.linalg.norm(
                inlier_destination_points - center_point,
                axis=1,
            )

            cluster = inlier_destination_points[
                distances <= cluster_radius
            ]

            if (
                largest_cluster is None
                or len(cluster) > len(largest_cluster)
            ):
                largest_cluster = cluster

        if largest_cluster is None or len(largest_cluster) < 4:
            return {
                "bbox": None,
                "score": 0.0,
                "good_matches": len(good_matches),
                "inliers": inliers,
                "inlier_ratio": float(inlier_ratio),
                "reason": "no_dense_inlier_cluster",
            }

        cluster_x_values = largest_cluster[:, 0]
        cluster_y_values = largest_cluster[:, 1]

        raw_x1 = float(cluster_x_values.min())
        raw_y1 = float(cluster_y_values.min())
        raw_x2 = float(cluster_x_values.max())
        raw_y2 = float(cluster_y_values.max())

        raw_width = raw_x2 - raw_x1
        raw_height = raw_y2 - raw_y1

        if raw_width < 10 or raw_height < 6:
            return {
                "bbox": None,
                "score": 0.0,
                "good_matches": len(good_matches),
                "inliers": inliers,
                "inlier_ratio": float(inlier_ratio),
                "reason": "dense_cluster_too_small",
            }

        padding_x = max(18, int(raw_width * 0.60))
        padding_y = max(15, int(raw_height * 0.60))

        bbox = clip_bbox(
            (
                int(np.floor(raw_x1)) - padding_x,
                int(np.floor(raw_y1)) - padding_y,
                int(np.ceil(raw_x2)) + padding_x,
                int(np.ceil(raw_y2)) + padding_y,
            ),
            frame_width,
            frame_height,
        )

    
    else:
        bbox = projected_bbox

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        return empty_result

    bbox_area_ratio = (
        bbox_width
        * bbox_height
        / max(1, frame_width * frame_height)
    )

    # Ã‡ok kÃ¼Ã§Ã¼k yanlÄ±ÅŸ homografileri ve neredeyse tÃ¼m ekranÄ± kaplayan
    # anlamsÄ±z dÃ¶nÃ¼ÅŸÃ¼mleri reddet.
    # Referans 3 iÃ§in projected bbox gÃ¼venilir deÄŸil.
    # Alan kontrolÃ¼ daha sonra inlier noktalarÄ±ndan Ã¼retilen gerÃ§ek bbox Ã¼zerinde yapÄ±lacak.
    if REFERENCE_NUMBER != 3:
        if bbox_area_ratio < 0.0005:
            return {
                "bbox": None,
                "score": 0.0,
                "good_matches": len(good_matches),
                "inliers": inliers,
                "inlier_ratio": float(inlier_ratio),
                "reason": f"bbox_too_small_{bbox_area_ratio:.6f}",
            }

        if bbox_area_ratio > 0.40:
            return {
                "bbox": None,
                "score": 0.0,
                "good_matches": len(good_matches),
                "inliers": inliers,
                "inlier_ratio": float(inlier_ratio),
                "reason": f"bbox_too_large_{bbox_area_ratio:.6f}",
            }

    score = (
        inliers * 10.0
        + len(good_matches) * 2.0
        + inlier_ratio * 20.0
    )

    return {
        "bbox": bbox,
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
            f"Model bulunamadÄ±: {MODEL_PATH}"
        )

    reference_image = cv2.imread(
        str(REFERENCE_PATH)
    )

    if reference_image is None:
        raise RuntimeError(
            f"Referans okunamadÄ±: {REFERENCE_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    # Referans gÃ¶rseli zaten yarÄ±ÅŸma sunucusunun verdiÄŸi hedef nesnedir.
    # YOLO yalnÄ±z tekerlek gibi kÃ¼Ã§Ã¼k bir parÃ§ayÄ± seÃ§ebildiÄŸi iÃ§in
    # referans tarafÄ±nda crop uygulamÄ±yoruz.
    reference_crop = reference_image.copy()
    reference_bbox = None

    print(
        "Referans YOLO bbox: kullanÄ±lmadÄ±"
    )
    print(
        "Referans tam gÃ¶rÃ¼ntÃ¼ shape:",
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

        # Referans 2-5 taÅŸÄ±t olmak zorunda deÄŸil.
        # Bu referanslarda YOLO taÅŸÄ±t kutusu yerine tÃ¼m karede
        # SIFT homografisi ile doÄŸrudan lokalizasyon yap.
        if REFERENCE_NUMBER in (3, 4):
            localization = sift_localize_in_frame(
                reference_crop,
                frame_image,
            )

            bbox = localization["bbox"]

            if bbox is None:
                if REFERENCE_NUMBER == 3:
                    print(
                        f"{frame_path.name} localization rejected "
                        f"reason={localization.get('reason', 'unknown')} "
                        f"good={localization.get('good_matches', 0)} "
                        f"inliers={localization.get('inliers', 0)} "
                        f"ratio={localization.get('inlier_ratio', 0.0):.2f}"
                    )
                continue

            x1, y1, x2, y2 = bbox

            candidate_width = x2 - x1
            candidate_height = y2 - y1

            if candidate_width <= 0 or candidate_height <= 0:
                continue

            candidate_area_ratio = (
                candidate_width
                * candidate_height
                / max(1, width * height)
            )

            candidate_aspect_ratio = (
                candidate_width
                / max(1, candidate_height)
            )

            aspect_similarity = min(
                candidate_aspect_ratio,
                reference_aspect_ratio,
            ) / max(
                candidate_aspect_ratio,
                reference_aspect_ratio,
            )

            # Referans 3'te homography bazen kalenin yalnÄ±z kÃ¼Ã§Ã¼k bir boru
            # parÃ§asÄ±nÄ± gÃ¼Ã§lÃ¼ eÅŸleÅŸme olarak dÃ¶ndÃ¼rebiliyor.
            # Referans 3 iÃ§in daha sÄ±kÄ± alan ve ÅŸekil filtresi uygula.
            if REFERENCE_NUMBER == 3:
                min_candidate_area_ratio = 0.0005
                max_candidate_area_ratio = 0.05
                min_aspect_similarity = 0.25
            else:
                min_candidate_area_ratio = 0.0005
                max_candidate_area_ratio = 0.40
                min_aspect_similarity = 0.25

            if (
                    candidate_area_ratio < min_candidate_area_ratio
                    or candidate_area_ratio > max_candidate_area_ratio
                    or aspect_similarity < min_aspect_similarity
                ):


                print(
                    f"{frame_path.name} geometry rejected "
                    f"bbox={bbox} "
                    f"area={candidate_area_ratio:.6f} "
                    f"aspect={aspect_similarity:.2f} "
                    f"good={localization['good_matches']} "
                    f"inliers={localization['inliers']} "
                    f"ratio={localization['inlier_ratio']:.2f}"
                )
                continue

            crop = frame_image[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                continue

            color_similarity = hsv_color_similarity(
                reference_crop,
                crop,
            )

            # Referans 3 beyaz metal kale ve yeÅŸil saha gÃ¶rÃ¼nÃ¼mÃ¼ne sahip.
            # DÃ¼ÅŸÃ¼k renk benzerlikli toprak/yol eÅŸleÅŸmelerini reddet.
            if REFERENCE_NUMBER == 3 and color_similarity < 0.20:
                print(
                    f"{frame_path.name} color rejected "
                    f"bbox={bbox} "
                    f"color={color_similarity:.2f} "
                    f"inliers={localization['inliers']} "
                    f"ratio={localization['inlier_ratio']:.2f}"
                )
                continue

            combined_score = (
                localization["score"]
                * (
                    0.50
                    + 0.50 * aspect_similarity
                )
            )

            candidate = {
                "frame_index": frame_index,
                "frame_name": frame_path.name,
                "box_index": 0,
                "bbox": bbox,
                "confidence": 1.0,
                "frame_image": frame_image,
                "candidate_area_ratio": candidate_area_ratio,
                "aspect_similarity": aspect_similarity,
                "color_similarity": color_similarity,
                "combined_score": combined_score,
                **localization,
            }

            candidates.append(candidate)

            print(
                f"{frame_path.name} "
                f"bbox={bbox} "
                f"sift={localization['score']:.1f} "
                f"good={localization['good_matches']} "
                f"inliers={localization['inliers']} "
                f"ratio={localization['inlier_ratio']:.2f}"
            )

            # Bu kare referans 2-5 yÃ¶ntemiyle iÅŸlendi.
            # Alttaki YOLO taÅŸÄ±t dÃ¶ngÃ¼sÃ¼ne girme.
            continue

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

                # TanÄ± sÄ±rasÄ±nda 0.01 kullandÄ±k ancak final aday seÃ§iminde
                # dÃ¼ÅŸÃ¼k gÃ¼venli kutular false positive Ã¼retiyor.
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

                # Minicik yanlÄ±ÅŸ YOLO kutularÄ±nÄ± ele.
                # GerÃ§ek biÃ§erdÃ¶ver frame alanÄ±nÄ±n yaklaÅŸÄ±k %1â€“2'sini kaplÄ±yor.
                if candidate_area_ratio < 0.008:
                    continue

                # AÅŸÄ±rÄ± bÃ¼yÃ¼k, anlamsÄ±z kutularÄ± da ele.
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

                # Referans yatay biÃ§erdÃ¶ver.
                # Dikey otomobil kutularÄ±nÄ± SIFT'e gÃ¶ndermiyoruz.
                if aspect_similarity < 0.60:
                    continue

                crop = frame_image[
                    y1:y2,
                    x1:x2,
                ]

                if crop.size == 0:
                    continue
                
                # Ref 5 termal referans, yarışma karesi ise RGB.
                # Bu nedenle SIFT ve referans renk benzerliği güvenilir değil.
                # YOLO taşıt kutusu + sarı iş makinesi oranı kullanılır.
                if REFERENCE_NUMBER == 5:
                    yellow_ratio = yellow_vehicle_ratio(
                        crop
                    )

                    # Küçük otomobilleri ve sarı olmayan kutuları reddet.
                    if candidate_area_ratio < 0.006:
                        continue

                    if candidate_area_ratio > 0.10:
                        continue

                    if yellow_ratio < 0.08:
                        continue

                    # Biçerdöver genelde kare veya hafif yatay görünür.
                    if candidate_aspect_ratio < 0.55:
                        continue

                    if candidate_aspect_ratio > 2.20:
                        continue

                    combined_score = (
                        confidence
                        * (
                            1.0
                            + 5.0 * yellow_ratio
                        )
                        * (
                            1.0
                            + 4.0 * candidate_area_ratio
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
                        "aspect_similarity": 1.0,
                        "color_similarity": yellow_ratio,
                        "combined_score": combined_score,
                        "score": combined_score,
                        "good_matches": 0, 
                        "inliers": 0,
                        "inlier_ratio": 0.0,
                    }

                    candidates.append(candidate)

                    print(
                        f"{frame_path.name} "
                        f"bbox={bbox} "
                        f"yolo={confidence:.3f} "
                        f"yellow={yellow_ratio:.3f} "
                        f"area={candidate_area_ratio:.4f}"
                    )

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

                if color_similarity < MIN_COLOR_SIMILARITY:
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
        "SonuÃ§ klasÃ¶rÃ¼:",
        OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 3 YOLO + SIFT reference candidate scanner"
    )

    parser.add_argument(
        "--reference",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Taranacak referans numarasi",
    )

    arguments = parser.parse_args()

    configure_reference(arguments.reference)
    main()
