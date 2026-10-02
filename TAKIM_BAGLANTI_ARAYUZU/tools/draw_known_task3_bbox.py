from pathlib import Path
import cv2


ROOT = Path(__file__).resolve().parents[1]

FRAME_PATH = ROOT / "data" / "frames_thermal" / "frame_001110.jpg"
OUT_DIR = ROOT / "outputs" / "task3_refs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "thermal_frame_001110_known_bbox.jpg"

BBOX = (175, 369, 206, 385)

img = cv2.imread(str(FRAME_PATH))

if img is None:
    print(f"[ERR] Frame okunamadı: {FRAME_PATH}")
    raise SystemExit(1)

x1, y1, x2, y2 = BBOX

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.putText(
    img,
    "Ref",
    (x1, max(25, y1 - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.imwrite(str(OUT_PATH), img)

print(f"[OK] Kaydedildi: {OUT_PATH}")