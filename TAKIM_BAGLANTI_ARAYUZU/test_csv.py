import pandas as pd


def load_and_show(csv_path):
    print("\nCSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    print(df.head())
    print(df.tail())


load_and_show("data/translations/THYZ_2026_Ornek_Veri_1_translation.csv")
load_and_show("data/translations/THYZ_2026_Ornek_Veri_2_Termal_translation.csv")