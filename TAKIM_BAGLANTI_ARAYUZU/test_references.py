import cv2
from pathlib import Path


def check_references(folder):
    folder = Path(folder)
    print("\nKlasör:", folder)

    image_paths = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.png")))

    print("Referans sayısı:", len(image_paths))

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            print("OKUNAMADI:", path.name)
            continue

        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        print(f"{path.name} -> width={w}, height={h}, channels={channels}")

        # İlk görseli çıktı olarak kaydet
        out_dir = Path("outputs/tests")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"check_{path.stem}.jpg"
        cv2.imwrite(str(out_path), img)


check_references("data/references_rgb")
check_references("data/references_thermal")