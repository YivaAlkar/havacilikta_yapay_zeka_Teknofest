import logging
import time
import requests
from pathlib import Path

from pathlib import Path

try:
    from src_custom.csv_utils import TranslationReader
except Exception:
    TranslationReader = None


from ultralytics import YOLO

from src_custom.reference_matcher import match_reference

from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction


class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        logging.info("Created Object Detection Model")
        self.evaulation_server = evaluation_server_url

        self.model_path = self._select_model_path()
        logging.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.is_custom_model = Path(self.model_path).name == "best.pt"
        logging.info(f"Custom model mode: {self.is_custom_model}")

        self.last_translation = None
        self.translation_readers = []
        self._init_translation_readers()

        self.previous_vehicle_centers = []
        self.movement_threshold_px = 35.0

        self.coco_to_competition = {
            0: classes["Insan"],
            2: classes["Tasit"],
            3: classes["Tasit"],
            5: classes["Tasit"],
            7: classes["Tasit"],
        }

    @staticmethod
    def _select_model_path():
        """
        Model seçim sırası:
        1. models/best.pt       → arkadaşının eğittiği final model
        2. models/yolov8n.pt    → bizim baseline model
        3. yolov8n.pt           → repo kökündeki baseline model
        """
        candidates = [
            Path("models/best.pt"),
            Path("models/yolov8n.pt"),
            Path("yolov8n.pt"),
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        raise FileNotFoundError("Hiçbir model dosyası bulunamadı: models/best.pt, models/yolov8n.pt, yolov8n.pt")

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]

        if image_name not in images_files:
            headers = {"Authorization": f"Token {auth_token}"} if auth_token else {}

            for attempt in range(retries):
                try:
                    response = requests.get(img_url, headers=headers, timeout=60)
                    response.raise_for_status()

                    img_bytes = response.content
                    with open(images_folder + image_name, "wb") as img_file:
                        img_file.write(img_bytes)

                    t2 = time.perf_counter()
                    logging.info(f"{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}")
                    return

                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        else:
            logging.info(f"{image_name} already exists in {images_folder}, skipping download.")

    def process(
        self,
        prediction,
        evaluation_server_url,
        health_status,
        images_folder,
        images_files,
        active_refs=None,
        ref_image_paths=None,
        auth_token=None,
    ):
        self.download_image(
            evaluation_server_url + "media" + prediction.image_url,
            images_folder,
            images_files,
            auth_token=auth_token,
        )

        frame_image_path = images_folder + prediction.image_url.split("/")[-1]

        frame_results = self.detect(
            prediction,
            health_status,
            active_refs=active_refs or [],
            ref_image_paths=ref_image_paths or {},
            frame_image_path=frame_image_path,
        )

        return frame_results

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _clip_bbox(x1, y1, x2, y2):
        x1 = max(0.0, float(x1))
        y1 = max(0.0, float(y1))
        x2 = max(0.0, float(x2))
        y2 = max(0.0, float(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _bbox_center(x1, y1, x2, y2):
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _decide_vehicle_motion(self, center):
        if not self.previous_vehicle_centers:
            return moving_statuses["Sabit"]

        nearest_distance = min(
            self._distance(center, prev_center)
            for prev_center in self.previous_vehicle_centers
        )

        if nearest_distance > self.movement_threshold_px:
            return moving_statuses["Hareketli"]

        return moving_statuses["Sabit"]

    def _add_task_1_detections(self, prediction, frame_image_path):
        current_vehicle_centers = []

        try:
            results = self.model.predict(
                     source=frame_image_path,
                     conf=0.40,
                     iou=0.50,
                     verbose=False,
            )[0]

            frame_h, frame_w = results.orig_shape

            for box in results.boxes:
                coco_cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < 0.40:
                    continue

                if self.is_custom_model:
                    # Arkadaşının eğittiği modelde sınıf sırası doğrudan yarışma formatında olmalı:
                    # 0=Tasit, 1=Insan, 2=UAP, 3=UAI
                    if coco_cls not in [0, 1, 2, 3]:
                        continue
                    competition_cls = coco_cls
                else:
                    # Hazır YOLOv8n COCO modelinden yarışma sınıflarına dönüşüm.
                    if coco_cls not in self.coco_to_competition:
                        continue
                    competition_cls = self.coco_to_competition[coco_cls]
                
                # Sınıf bazlı confidence filtresi
                if competition_cls == classes["Insan"] and conf < 0.55:
                    continue

                if competition_cls == classes["Tasit"] and conf < 0.40:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = self._clip_bbox(x1, y1, x2, y2)
                if bbox is None:
                    continue

                top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox

                bbox_w = bottom_right_x - top_left_x
                bbox_h = bottom_right_y - top_left_y
                bbox_area = bbox_w * bbox_h
                frame_area = frame_w * frame_h
                area_ratio = bbox_area / max(1, frame_area)
                aspect_ratio = bbox_w / max(1, bbox_h)

                # Genel güvenlik filtresi
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Çok küçük kutular genelde gürültü
                if bbox_w < 8 or bbox_h < 8:
                    continue

                # Tasit filtresi
                if competition_cls == classes["Tasit"]:
                    if area_ratio > 0.10:
                        continue

                    if aspect_ratio > 6.0 or aspect_ratio < 0.18:
                        continue

                    # Aşırı ince/uzun kutuları ele
                    if aspect_ratio > 5.5 or aspect_ratio < 0.20:
                        continue

                # Insan filtresi
                if competition_cls == classes["Insan"]:
                      if area_ratio > 0.015:
                            continue

                      if bbox_w > 150 or bbox_h > 230:
                            continue

                      if aspect_ratio > 2.7 or aspect_ratio < 0.17:
                          continue
                    
                center = self._bbox_center(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

                cls = competition_cls

                if competition_cls in (classes["UAP"], classes["UAI"]):
                    landing_status = landing_statuses["Inilebilir"]
                else:
                    landing_status = landing_statuses["Inis Alani Degil"]

                if competition_cls == classes["Tasit"]:
                    current_vehicle_centers.append(center)
                    moving_status = self._decide_vehicle_motion(center)
                else:
                    moving_status = moving_statuses["Tasit Degil"]

                d_obj = DetectedObject(
                    cls,
                    landing_status,
                    moving_status,
                    top_left_x,
                    top_left_y,
                    bottom_right_x,
                    bottom_right_y,
                )

                prediction.add_detected_object(d_obj)

            self.previous_vehicle_centers = current_vehicle_centers

        except Exception as e:
            logging.exception(f"Task 1 detection failed: {e}")

    def _init_translation_readers(self):
        """
        data/translations içindeki CSV dosyalarını opsiyonel olarak yükler.
        CSV yoksa sistem bozulmaz.
        """
        try:
            if TranslationReader is None:
                logging.info("TranslationReader import edilemedi; CSV translation modu kapalı.")
                return

            root = Path(__file__).resolve().parents[1]
            translation_dir = root / "data" / "translations"

            if not translation_dir.exists():
                logging.info(f"Translation klasörü yok: {translation_dir}")
                return

            csv_files = sorted(translation_dir.glob("*.csv"))

            if not csv_files:
                logging.info(f"Translation CSV bulunamadı: {translation_dir}")
                return

            for csv_path in csv_files:
                try:
                    reader = TranslationReader(csv_path)
                    self.translation_readers.append(reader)
                    logging.info(f"Translation CSV yüklendi: {csv_path}")
                except Exception as e:
                    logging.exception(f"Translation CSV yüklenemedi: {csv_path}, err={e}")

        except Exception as e:
            logging.exception(f"CSV translation reader init failed: {e}")


    def _get_csv_translation_for_prediction(self, prediction):
        """
        Prediction içindeki frame adını kullanarak CSV'den translation bulmaya çalışır.
        Bulamazsa None döner.
        """
        try:
            if not self.translation_readers:
                return None

            candidates = []

            frame_url = getattr(prediction, "frame_url", None)
            image_url = getattr(prediction, "image_url", None)

            if frame_url:
                # /api/frames/frame_001110/ -> frame_001110
                parts = [p for p in str(frame_url).replace("\\", "/").split("/") if p]
                for p in parts:
                    if p.startswith("frame_"):
                        candidates.append(p)
                        candidates.append(f"{p}.jpg")
                        candidates.append(f"{p}.png")

            if image_url:
                # /media/frames/frame_001110.jpg -> frame_001110.jpg
                name = Path(str(image_url).replace("\\", "/")).name
                if name:
                    candidates.append(name)
                    candidates.append(Path(name).stem)

            # duplicate temizle
            candidates = list(dict.fromkeys(candidates))

            for reader in self.translation_readers:
                for frame_name in candidates:
                    value = reader.get_by_frame_name(frame_name)

                    if value is None:
                        continue

                    tx, ty, tz = value
                    return (
                        self._safe_float(tx),
                        self._safe_float(ty),
                        self._safe_float(tz),
                    )

            return None

        except Exception as e:
            logging.exception(f"CSV translation lookup failed: {e}")
            return None


    def _add_task_2_translation(self, prediction, health_status):
        try:
            gt_available = None not in (
                prediction.gt_translation_x,
                prediction.gt_translation_y,
                prediction.gt_translation_z,
            )

            if health_status is None:
                logging.info("No translation/health_status for this frame; skipping Mission 2 output.")
                return

            csv_translation = self._get_csv_translation_for_prediction(prediction)

            if health_status == "1":
                if csv_translation is not None:
                    tx, ty, tz = csv_translation
                    self.last_translation = (tx, ty, tz)
                    prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
                    return

                if gt_available:
                    tx = self._safe_float(prediction.gt_translation_x)
                    ty = self._safe_float(prediction.gt_translation_y)
                    tz = self._safe_float(prediction.gt_translation_z)

                    self.last_translation = (tx, ty, tz)
                    prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
                else:
                    logging.info("Healthy frame but GT translation is null; skipping Mission 2 output.")
                return

            if health_status == "0":
                if csv_translation is not None:
                    tx, ty, tz = csv_translation
                    self.last_translation = (tx, ty, tz)
                elif self.last_translation is not None:
                    tx, ty, tz = self.last_translation
                elif gt_available:
                    tx = self._safe_float(prediction.gt_translation_x)
                    ty = self._safe_float(prediction.gt_translation_y)
                    tz = self._safe_float(prediction.gt_translation_z)
                    self.last_translation = (tx, ty, tz)
                else:
                    tx, ty, tz = 0.0, 0.0, 0.0

                prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
                return

            logging.info(f"Unknown health_status={health_status}; skipping Mission 2 output.")

        except Exception as e:
            logging.exception(f"Task 2 translation failed: {e}")



    def _add_task_3_reference_predictions(self, prediction, active_refs=None, ref_image_paths=None, frame_image_path=None):
        """
        Görev 3:
        Aktif referans varsa reference_matcher.py ile frame içinde arar.
        BBox bulunursa ReferencePrediction olarak ekler.
        Bulamazsa boş geçer.
        """
        try:
            active_refs = active_refs or []
            ref_image_paths = ref_image_paths or {}

            if not active_refs:
                return

            for ref in active_refs:
                start_img = ref.get("frame_start_image_url", "")
                end_img = ref.get("frame_end_image_url", "")

                if not (start_img and end_img and start_img <= prediction.image_url <= end_img):
                    continue

                ref_url = ref.get("url")
                if not ref_url:
                    logging.warning("Active reference has no url; skipping.")
                    continue

                ref_path = ref_image_paths.get(ref_url)

                if not ref_path:
                    logging.warning(f"Reference path not found for url={ref_url}; skipping.")
                    continue

                logging.info(f"Task 3 matcher running: frame={frame_image_path}, ref={ref_path}")

                bbox = match_reference(frame_image_path, ref_path)

                if not bbox:
                    logging.info(f"Task 3 no bbox found for ref={ref_url}")
                    continue

                logging.info(f"Task 3 bbox found for ref={ref_url}: {bbox}")

                prediction.add_reference_prediction(
                    ReferencePrediction(ref_url, prediction.frame_url, *bbox)
                )

        except Exception as e:
            logging.exception(f"Task 3 reference prediction failed: {e}")

    def detect(self, prediction, health_status, active_refs=None, ref_image_paths=None, frame_image_path=None):
        if frame_image_path is None:
            logging.error("frame_image_path is None; returning empty prediction.")
            return prediction

        self._add_task_1_detections(prediction, frame_image_path)
        self._add_task_2_translation(prediction, health_status)
        self._add_task_3_reference_predictions(
            prediction,
            active_refs=active_refs,
            ref_image_paths=ref_image_paths,
            frame_image_path=frame_image_path,
        )

        return prediction