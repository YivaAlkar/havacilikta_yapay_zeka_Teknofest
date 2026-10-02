from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

frames_dir = Path("data/frames_rgb")
frames = sorted(frames_dir.glob("*.jpg"))

found = []

for i, frame_path in enumerate(frames):
    result = model(str(frame_path), verbose=False)[0]
    count = len(result.boxes)

    if count > 0:
        found.append((frame_path.name, count))
        print(frame_path.name, "tespit:", count)

    if i % 50 == 0:
        print("Tarandı:", i)

print("\nToplam tespitli frame:", len(found))
print("İlk 20:", found[:20])