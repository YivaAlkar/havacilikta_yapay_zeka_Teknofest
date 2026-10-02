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

        # Task 2 kısa süreli hareket tahmini durumu.
        self.previous_translation = None
        self.translation_velocity = (0.0, 0.0, 0.0)
        self.translation_stale_frames = 0

        # Kesinti sırasında en fazla kaç frame tahmin üretileceği.
        self.max_translation_stale_frames = 8

        # Tahmin ilerledikçe hızın küçülmesi için damping.
        self.translation_velocity_damping = 0.85

        # Tek frame'de olağan dışı büyük tahmin sıçramasını sınırlar.
        self.max_translation_step = 25.0

        # İki güvenilir ölçüm arasındaki 3B mesafe bu eşiği aşarsa
        # yeni koordinat kabul edilir fakat bu sıçramadan hız öğrenilmez.
        self.max_reliable_translation_jump = 10.0

        # Birden fazla CSV aynı frame adlarını içeriyorsa yanlış CSV'nin
        # sessizce kullanılmasını engellemek için uyarıyı yalnızca bir kez yazdırır.
        self._translation_csv_ambiguity_logged = False

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
    def download_image(
        img_url,
        images_folder,
        images_files,
        retries=3,
        initial_wait_time=0.1,
        auth_token=None,
    ):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]
        image_path = images_folder + image_name

        # Listede görünse bile dosyanın gerçekten mevcut ve boş olmadığını doğrula.
        if image_name in images_files:
            try:
                with open(image_path, "rb") as existing_file:
                    if existing_file.read(1):
                        logging.info(
                            f"{image_name} already exists in {images_folder}, "
                            "skipping download."
                        )
                        return True
            except OSError:
                logging.warning(
                    f"{image_name} was listed but could not be read; "
                    "download will be retried."
                )

        headers = {"Authorization": f"Token {auth_token}"} if auth_token else {}

        for attempt in range(retries):
            try:
                response = requests.get(img_url, headers=headers, timeout=60)
                response.raise_for_status()

                img_bytes = response.content
                if not img_bytes:
                    raise requests.exceptions.RequestException(
                        "Downloaded response body is empty."
                    )

                with open(image_path, "wb") as img_file:
                    img_file.write(img_bytes)

                # Yazılan dosyanın gerçekten boş olmadığını doğrula.
                with open(image_path, "rb") as saved_file:
                    if not saved_file.read(1):
                        raise OSError("Downloaded image file is empty.")

                t2 = time.perf_counter()
                logging.info(
                    f"{img_url} - Download Finished in {t2 - t1} "
                    f"seconds to {image_path}"
                )
                return True

            except (requests.exceptions.RequestException, OSError) as e:
                logging.error(
                    f"Download failed for {img_url} on attempt "
                    f"{attempt + 1}/{retries}: {e}"
                )

                if attempt < retries - 1:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2

        logging.error(
            f"Failed to download image from {img_url} after {retries} attempts."
        )
        return False

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
        download_ok = self.download_image(
            evaluation_server_url + "media" + prediction.image_url,
            images_folder,
            images_files,
            auth_token=auth_token,
        )

        frame_image_path = images_folder + prediction.image_url.split("/")[-1]

        if not download_ok:
            logging.error(
                f"Frame image could not be downloaded; prediction will not be "
                f"created or submitted: {prediction.image_url}"
            )
            return None

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
                    (competition_cls,),
                    landing_status,
                    moving_status,
                    top_left_x,
                    top_left_y,
                    bottom_right_x,
                    bottom_right_y,
                )
                setattr(d_obj, "confidence", conf)
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
            Prediction içindeki frame adını kullanarak CSV'den translation bulur.

            Güvenlik kuralı:
            - CSV yoksa None döner.
            - Yalnızca bir CSV reader varsa onu kullanır.
            - Birden fazla CSV varsa video/modalite eşleşmesi kesin olmadığı için
            ilk eşleşeni körlemesine kullanmaz ve None döner.

            Böylece aynı frame adlarını içeren farklı rota CSV'lerinin yanlışlıkla
            birbirinin yerine kullanılması engellenir.
            """
            try:
                if not self.translation_readers:
                    return None

                if len(self.translation_readers) != 1:
                    if not self._translation_csv_ambiguity_logged:
                        logging.warning(
                            "Birden fazla translation CSV yüklü (%d adet). "
                            "Video/modalite eşleşmesi kesin olmadığı için CSV "
                            "translation devre dışı bırakıldı.",
                            len(self.translation_readers),
                        )
                        self._translation_csv_ambiguity_logged = True

                    return None

                candidates = []

                frame_url = getattr(prediction, "frame_url", None)
                image_url = getattr(prediction, "image_url", None)

                if frame_url:
                    # /api/frames/frame_001110/ -> frame_001110
                    parts = [
                        part
                        for part in str(frame_url).replace("\\", "/").split("/")
                        if part
                    ]

                    for part in parts:
                        if part.startswith("frame_"):
                            candidates.append(part)
                            candidates.append(f"{part}.jpg")
                            candidates.append(f"{part}.png")

                if image_url:
                    # /media/frames/frame_001110.jpg -> frame_001110.jpg
                    name = Path(
                        str(image_url).replace("\\", "/")
                    ).name

                    if name:
                        candidates.append(name)
                        candidates.append(Path(name).stem)

                candidates = list(dict.fromkeys(candidates))
                reader = self.translation_readers[0]

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
                logging.exception(
                    f"CSV translation lookup failed: {e}"
                )
                return None

    
    def _record_reliable_translation(self, translation):
        """
        GT veya kesin CSV gibi güvenilir bir translation geldiğinde
        geçmişi ve kısa süreli hız tahminini günceller.

        Ani sıçrama koruması:
        - Yeni koordinat her durumda kabul edilir.
        - Önceki konuma göre 3B sıçrama aşırı büyükse bu ölçümden
          hız öğrenilmez.
        - Böylece tek bozuk ölçüm sonraki tahminleri zehirlemez.
        """
        tx, ty, tz = (
            self._safe_float(translation[0]),
            self._safe_float(translation[1]),
            self._safe_float(translation[2]),
        )

        current = (tx, ty, tz)

        if self.last_translation is not None:
            previous = self.last_translation

            raw_velocity = (
                current[0] - previous[0],
                current[1] - previous[1],
                current[2] - previous[2],
            )

            jump_distance = (
                raw_velocity[0] ** 2
                + raw_velocity[1] ** 2
                + raw_velocity[2] ** 2
            ) ** 0.5

            self.previous_translation = previous

            if (
                jump_distance
                > self.max_reliable_translation_jump
            ):
                logging.warning(
                    "Task 2 translation jump detected: "
                    "distance=%.6f threshold=%.6f. "
                    "Position accepted, velocity reset.",
                    jump_distance,
                    self.max_reliable_translation_jump,
                )

                # Yeni konumu kabul ediyoruz ancak sıçramadan hız
                # öğrenmeyerek sonraki tahminleri koruyoruz.
                self.translation_velocity = (
                    0.0,
                    0.0,
                    0.0,
                )

            else:
                self.translation_velocity = tuple(
                    max(
                        -self.max_translation_step,
                        min(
                            self.max_translation_step,
                            value,
                        ),
                    )
                    for value in raw_velocity
                )

        self.last_translation = current
        self.translation_stale_frames = 0

        return current


    def _predict_short_term_translation(self):
        """
        Son güvenilir translation ve ölçülen hız kullanılarak kısa süreli
        sabit-hız tahmini yapar. Eski değer sınırsız taşınmaz.
        """
        if self.last_translation is None:
            return None

        if (
            self.translation_stale_frames
            >= self.max_translation_stale_frames
        ):
            return None

        next_age = self.translation_stale_frames + 1
        damping = (
            self.translation_velocity_damping
            ** max(0, next_age - 1)
        )

        step = tuple(
            max(
                -self.max_translation_step,
                min(
                    self.max_translation_step,
                    velocity * damping,
                ),
            )
            for velocity in self.translation_velocity
        )

        predicted = (
            self.last_translation[0] + step[0],
            self.last_translation[1] + step[1],
            self.last_translation[2] + step[2],
        )

        self.last_translation = predicted
        self.translation_stale_frames = next_age

        return predicted

    def _add_task_2_translation(self, prediction, health_status):
        """
        Task 2 güvenli translation akışı.

        health_status == "1":
        1. Sunucudan gelen GT
        2. Kesin ve tek CSV
        3. Veri yoksa çıktı gönderme

        health_status == "0":
         1. Kesin ve tek CSV
         2. Son iki güvenilir değerden kısa süreli hareket tahmini
         3. GT mevcutsa fallback
         4. Hiç veri yoksa 0, 0, 0

        Kısa süreli tahmin en fazla max_translation_stale_frames kadar
        sürer; eski translation sınırsız taşınmaz.
        """
        try:
            gt_available = None not in (
                prediction.gt_translation_x,
                prediction.gt_translation_y,
                prediction.gt_translation_z,
            )

            if health_status is None:
                logging.info(
                    "No health_status for this frame; "
                    "skipping Mission 2 output."
                )
                return

            if health_status == "1":
                if gt_available:
                    translation = self._record_reliable_translation(
                        (
                            prediction.gt_translation_x,
                            prediction.gt_translation_y,
                            prediction.gt_translation_z,
                        )
                    )

                    prediction.add_translation_object(
                        DetectedTranslation(*translation)
                    )
                    return

                csv_translation = (
                    self._get_csv_translation_for_prediction(
                        prediction
                    )
                )

                if csv_translation is not None:
                    translation = self._record_reliable_translation(
                        csv_translation
                    )

                    prediction.add_translation_object(
                        DetectedTranslation(*translation)
                    )
                    return

                logging.info(
                    "Healthy frame has no GT and no unambiguous CSV; "
                    "skipping Mission 2 output."
                )
                return

            if health_status == "0":
                csv_translation = (
                    self._get_csv_translation_for_prediction(
                        prediction
                    )
                )

                if csv_translation is not None:
                    translation = self._record_reliable_translation(
                        csv_translation
                    )

                else:
                    translation = (
                        self._predict_short_term_translation()
                    )

                    if translation is None and gt_available:
                        translation = (
                            self._record_reliable_translation(
                                (
                                    prediction.gt_translation_x,
                                    prediction.gt_translation_y,
                                    prediction.gt_translation_z,
                                )
                            )
                        )

                    if translation is None:
                        logging.warning(
                            "Translation history is missing or stale; "
                            "using zero fallback."
                        )
                        translation = (0.0, 0.0, 0.0)

                prediction.add_translation_object(
                    DetectedTranslation(*translation)
                )
                return

            logging.info(
                "Unknown health_status=%s; "
                "skipping Mission 2 output.",
                health_status,
            )

        except Exception as exc:
            logging.exception(
                "Task 2 translation failed: %s",
                exc,
            )



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