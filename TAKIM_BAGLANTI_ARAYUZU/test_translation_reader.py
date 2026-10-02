from src_custom.csv_utils import TranslationReader


rgb_reader = TranslationReader("data/translations/THYZ_2026_Ornek_Veri_1_translation.csv")
thermal_reader = TranslationReader("data/translations/THYZ_2026_Ornek_Veri_2_Termal_translation.csv")

print("RGB frame_000000:", rgb_reader.get_by_frame_name("frame_000000"))
print("RGB frame_000030:", rgb_reader.get_by_frame_name("frame_000030"))
print("RGB index 120:", rgb_reader.get_by_frame_index(120))

print("Thermal frame_000000:", thermal_reader.get_by_frame_name("frame_000000"))
print("Thermal frame_000030:", thermal_reader.get_by_frame_name("frame_000030"))
print("Thermal index 120:", thermal_reader.get_by_frame_index(120))