import pandas as pd


class TranslationReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.frame_map = {
            row["frame_numbers"]: (
                float(row["translation_x"]),
                float(row["translation_y"]),
                float(row["translation_z"]),
            )
            for _, row in self.df.iterrows()
        }

    def get_by_frame_name(self, frame_name):
        return self.frame_map.get(frame_name)

    def get_by_frame_index(self, frame_index):
        frame_name = f"frame_{int(frame_index):06d}"
        return self.get_by_frame_name(frame_name)