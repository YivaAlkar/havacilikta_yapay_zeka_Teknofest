import cv2
import numpy as np


def _clip_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return float(x1), float(y1), float(x2), float(y2)

def _is_reasonable_bbox(bbox, width, height, min_area_ratio=0.0005, max_area_ratio=0.25):
    """
    Çok küçük veya çok büyük bbox'ları eler.
    max_area_ratio=0.25 demek: görüntünün %25'inden büyük kutuları reddet.
    """
    if bbox is None:
        return False

    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    image_area = width * height

    if image_area <= 0:
        return False

    ratio = box_area / image_area

    if ratio < min_area_ratio:
        return False

    if ratio > max_area_ratio:
        return False

    return True

def match_reference_orb(frame_path, reference_path, min_matches=10):
    """
    ORB ile referans görseli frame içinde arar.
    İyi eşleşme bulursa bbox döndürür.
    Bulamazsa None döndürür.
    """
    frame = cv2.imread(str(frame_path))
    ref = cv2.imread(str(reference_path))

    if frame is None or ref is None:
        return None

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

    if des_ref is None or des_frame is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_ref, des_frame)

    if len(matches) < min_matches:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[: max(min_matches, int(len(matches) * 0.25))]

    if len(good_matches) < min_matches:
        return None

    frame_points = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])

    x, y, w, h = cv2.boundingRect(frame_points)

    H, W = frame.shape[:2]
    bbox = _clip_bbox(x, y, x + w, y + h, W, H)

    if not _is_reasonable_bbox(bbox, W, H):
        return None

    return bbox


def match_reference_template(frame_path, reference_path, scales=None, threshold=0.45):
    """
    Template matching ile referansı farklı ölçeklerde frame içinde arar.
    Skor yeterliyse bbox döndürür.
    """
    frame = cv2.imread(str(frame_path))
    ref = cv2.imread(str(reference_path))

    if frame is None or ref is None:
        return None

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    H, W = frame_gray.shape[:2]

    if scales is None:
        scales = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]

    best_score = -1
    best_bbox = None

    for scale in scales:
        new_w = int(ref_gray.shape[1] * scale)
        new_h = int(ref_gray.shape[0] * scale)

        if new_w < 10 or new_h < 10:
            continue

        if new_w >= W or new_h >= H:
            continue

        resized_ref = cv2.resize(ref_gray, (new_w, new_h))

        result = cv2.matchTemplate(frame_gray, resized_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            x1, y1 = max_loc
            x2, y2 = x1 + new_w, y1 + new_h
            best_score = max_val
            best_bbox = (x1, y1, x2, y2)

    if best_bbox is None or best_score < threshold:
        return None

    bbox = _clip_bbox(*best_bbox, W, H)

    if not _is_reasonable_bbox(bbox, W, H):
        return None

    return bbox

def match_reference(frame_path, reference_path):
    """
    Ana fonksiyon.
    Önce ORB dener, olmazsa template matching dener.
    """
    bbox = match_reference_orb(frame_path, reference_path)

    if bbox is not None:
        return bbox

    bbox = match_reference_template(frame_path, reference_path)

    return bbox