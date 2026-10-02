from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection_model import ObjectDetectionModel


def main():
    print("=" * 80)
    print("[CHECK] Model loader kontrolü")
    print("=" * 80)

    models_dir = ROOT / "models"
    best_path = models_dir / "best.pt"
    local_yolo_path = models_dir / "yolov8n.pt"

    print(f"[ROOT] {ROOT}")
    print(f"[MODELS_DIR] {models_dir}")
    print(f"[BEST_PT_EXISTS] {best_path.exists()} -> {best_path}")
    print(f"[LOCAL_YOLO_EXISTS] {local_yolo_path.exists()} -> {local_yolo_path}")

    model = ObjectDetectionModel("http://test/")

    print("\n[MODEL OBJECT]")
    print(f"type(model.model): {type(model.model)}")

    if hasattr(model, "model_path"):
        print(f"model.model_path: {model.model_path}")
    else:
        print("model.model_path: attribute yok")

    if hasattr(model, "is_custom_model"):
        print(f"model.is_custom_model: {model.is_custom_model}")
    else:
        print("model.is_custom_model: attribute yok")

    try:
        names = model.model.names
        print(f"model.names: {names}")
    except Exception as e:
        print(f"model.names okunamadı: {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()