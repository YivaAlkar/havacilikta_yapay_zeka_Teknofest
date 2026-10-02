from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "outputs"
    / "v1_v3_comparison"
    / "frame_by_frame.csv"
)

OUT_PATH = (
    ROOT
    / "outputs"
    / "v1_v3_comparison"
    / "important_differences.txt"
)


def to_int(value):
    return int(value or 0)


def main():
    rows_by_frame = {}

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame = row["frame"]
            model = row["model"]

            rows_by_frame.setdefault(frame, {})[model] = row

    differences = []

    for frame, models in rows_by_frame.items():
        if "v1" not in models or "v3" not in models:
            continue

        v1 = models["v1"]
        v3 = models["v3"]

        v1_total = to_int(v1["total"])
        v3_total = to_int(v3["total"])

        v1_tasit = to_int(v1["tasit"])
        v3_tasit = to_int(v3["tasit"])

        v1_insan = to_int(v1["insan"])
        v3_insan = to_int(v3["insan"])

        v1_uap = to_int(v1["uap"])
        v3_uap = to_int(v3["uap"])

        v1_uai = to_int(v1["uai"])
        v3_uai = to_int(v3["uai"])

        score = (
            abs(v1_total - v3_total) * 3
            + abs(v1_tasit - v3_tasit) * 4
            + abs(v1_insan - v3_insan) * 5
            + abs(v1_uap - v3_uap) * 4
            + abs(v1_uai - v3_uai) * 4
        )

        if score == 0:
            continue

        differences.append(
            {
                "score": score,
                "frame": frame,
                "v1_total": v1_total,
                "v3_total": v3_total,
                "v1_tasit": v1_tasit,
                "v3_tasit": v3_tasit,
                "v1_insan": v1_insan,
                "v3_insan": v3_insan,
                "v1_uap": v1_uap,
                "v3_uap": v3_uap,
                "v1_uai": v1_uai,
                "v3_uai": v3_uai,
            }
        )

    differences.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    lines = []

    for item in differences[:40]:
        lines.append(
            f"{item['frame']} | "
            f"score={item['score']} | "
            f"V1 total={item['v1_total']} "
            f"T={item['v1_tasit']} "
            f"I={item['v1_insan']} "
            f"UAP={item['v1_uap']} "
            f"UAI={item['v1_uai']} | "
            f"V3 total={item['v3_total']} "
            f"T={item['v3_tasit']} "
            f"I={item['v3_insan']} "
            f"UAP={item['v3_uap']} "
            f"UAI={item['v3_uai']}"
        )

    OUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"[DONE] different frames={len(differences)}")
    print(f"[OUT] {OUT_PATH}")


if __name__ == "__main__":
    main()