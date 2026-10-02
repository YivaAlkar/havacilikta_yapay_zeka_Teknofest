from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "rgb": {
        "source": ROOT / "data" / "frames_rgb",
        "destination": ROOT / "data" / "label_candidates" / "rgb",
        "selection_file": ROOT / "tools" / "frame_selection" / "rgb_selected.txt",
    },
    "thermal": {
        "source": ROOT / "data" / "frames_thermal",
        "destination": ROOT / "data" / "label_candidates" / "thermal",
        "selection_file": ROOT / "tools" / "frame_selection" / "thermal_selected.txt",
    },
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_frame_number(value: str):
    value = value.strip()

    if not value or value.startswith("#"):
        return None

    numbers = re.findall(r"\d+", value)

    if not numbers:
        return None

    return int(numbers[-1])


def read_selected_numbers(selection_file: Path):
    if not selection_file.exists():
        print(f"[ERROR] Seçim dosyası bulunamadı: {selection_file}")
        return []

    selected_numbers = []

    for line in selection_file.read_text(encoding="utf-8").splitlines():
        number = extract_frame_number(line)

        if number is not None:
            selected_numbers.append(number)

    return list(dict.fromkeys(selected_numbers))


def build_source_index(source_folder: Path):
    index = {}

    if not source_folder.exists():
        print(f"[ERROR] Kaynak klasör bulunamadı: {source_folder}")
        return index

    for path in source_folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        frame_number = extract_frame_number(path.stem)

        if frame_number is not None:
            index[frame_number] = path

    return index


def copy_selected(source_name: str, settings: dict):
    source_folder = settings["source"]
    destination_folder = settings["destination"]
    selection_file = settings["selection_file"]

    destination_folder.mkdir(parents=True, exist_ok=True)

    selected_numbers = read_selected_numbers(selection_file)
    source_index = build_source_index(source_folder)

    print()
    print("=" * 70)
    print(f"[SOURCE] {source_name}")
    print(f"[SOURCE FILE COUNT] {len(source_index)}")
    print(f"[SELECTED] {len(selected_numbers)}")
    print("=" * 70)

    copied = 0
    missing = 0

    for frame_number in selected_numbers:
        source_path = source_index.get(frame_number)

        if source_path is None:
            print(f"[MISSING] frame number: {frame_number}")
            missing += 1
            continue

        destination_path = destination_folder / source_path.name
        shutil.copy2(source_path, destination_path)

        print(f"[COPIED] {source_path.name}")
        copied += 1

    print()
    print(f"[SUMMARY] {source_name}")
    print(f"copied={copied}")
    print(f"missing={missing}")
    print(f"destination={destination_folder}")


def main():
    print("[START] Seçilmiş eğitim kareleri kopyalanıyor...")

    for source_name, settings in CONFIG.items():
        copy_selected(source_name, settings)

    print()
    print("[FINISHED]")


if __name__ == "__main__":
    main()