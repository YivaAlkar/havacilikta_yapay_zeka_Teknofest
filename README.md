#  TEKNOFEST 2026 · Havacılıkta Yapay Zeka Yarışması

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF)
![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-E30A17)

---

## 🎯 Görevler

| # | Görev | Metrik | Durum |
|---|-------|--------|-------|
| 1 | **Nesne Tespiti**: taşıt / insan / iniş alanı tespiti, hareket ve iniş durumu | mAP (IoU 0.5) | ✅ Çalışıyor (YOLO tabanlı) |
| 2 | **Konum Kestirimi**: GPS sağlıksızken (x, y, z) tahmini | RMSE | ✅ Çalışıyor |
| 3 | **Referans Nesne Tespiti**: ilk kez görülen referans nesneyi bulma | mAP (IoU 0.25) | 🟡 Güvenli baseline (geliştirilecek) |

### Görev 1 – Nesne Tespiti
- Ultralytics **YOLO** modeli, `model.predict` ile çalışır.
- **Sınıf bazlı güven eşikleri**: `Tasit ≥ 0.40`, `Insan ≥ 0.55`
- Büyük bina/çatı ve anormal boyutlu insan **yanlış pozitifleri** filtrelenir.
- Kareler arası hareket analiziyle **hareketli / hareketsiz** durumu belirlenir.
- `models/best.pt` varsa özel model, yoksa `yolov8n.pt` otomatik seçilir.

### Görev 2 – Konum Kestirimi
- `health_status = 1` → sunucudan gelen translation kullanılır.
- `health_status = 0` → son bilinen konum / hareket tahmini kullanılır.
- `health_status = None` → translation eklenmez.
- İlk kare sağlıksızsa ve veri yoksa `(0, 0, 0)` fallback'i devreye girer.

### Görev 3 – Referans Nesne
- ORB + şablon eşleştirme ve YOLO adayları ile eşleştirme.
- **Bilinçli tercih:** güven düşükse yanlış kutu göndermek yerine boş döner.

---

## 📁 Proje Yapısı

```
.
├── Kamera_Kalibrasyon/            # Resmi kamera kalibrasyon parametreleri
├── TAKIM_BAGLANTI_ARAYUZU/
│   ├── main.py                    # Sunucuya bağlanıp kareleri işleyen ana akış
│   ├── src/                       # Resmi arayüz + modelimiz (object_detection_model.py)
│   ├── src_custom/                # Bizim eklediklerimiz: reference_matcher, csv_utils
│   ├── tools/                     # Veri seti hazırlama, görselleştirme, stres testi
│   ├── config/example.env         # Giriş bilgisi şablonu
│   ├── test_*.py                  # Birim / entegrasyon testleri
│   └── requirements.txt
├── RESMI_README.md                # Yarışmanın resmi açıklaması
└── README.md
```

---

## 🚀 Kurulum

```bash
conda create -n teknofest_yarisma python=3.11
conda activate teknofest_yarisma
pip install -r TAKIM_BAGLANTI_ARAYUZU/requirements.txt
pip install ultralytics torch torchvision   # GPU için CUDA'lı PyTorch önerilir
```

**Giriş bilgileri:** `TAKIM_BAGLANTI_ARAYUZU/config/example.env` dosyasını `config/.env` olarak kopyalayıp doldurun:

```text
TEAM_NAME=takim_kullanici_adiniz
PASSWORD=takim_sifreniz
EVALUATION_SERVER_URL="https://....../"
SESSION_NAME=
```

> 🔒 `.env` dosyası `.gitignore` ile dışarıda tutulur, **asla** GitHub'a yüklemeyin.

**Model dosyaları** (`*.pt`) boyutları yüzünden depoda yoktur. Kendi modelinizi `TAKIM_BAGLANTI_ARAYUZU/models/best.pt` yoluna koyun.

---

## ▶️ Çalıştırma

```bash
cd TAKIM_BAGLANTI_ARAYUZU
python main.py
```

> ⚠️ Sunucu her kare için **yalnızca bir kez** tahmin kabul eder (tekrar gönderimde `406`). Puanlanan oturumda `main.py`'yi modelin hazır olduğundan emin olmadan çalıştırmayın; önce test oturumunda deneyin.

## 🧪 Testler

```bash
cd TAKIM_BAGLANTI_ARAYUZU
python test_local_model.py
python test_motion_status.py
python test_health_status.py
python test_reference_matcher.py
python test_task3_refs.py
python tools/run_all_tests.py
```

---

## 🗺️ Yol Haritası

- [ ] Özel eğitilmiş model ile sınıf eşlemesi doğrulaması (`0 Tasit · 1 Insan · 2 UAP · 3 UAI`) ve eşiklerin yeniden ayarı
- [ ] Görev 3 için model tabanlı, daha güçlü referans eşleştirme
- [ ] Görev 2 CSV okuyucusunun resmi akışa daha temiz bağlanması
- [ ] Offline / bağlantı kopması stres testlerinin genişletilmesi

---

## 👥 Takım

**UluGazi Takımı** · TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması

## 🙏 Kaynaklar

- [TEKNOFEST Resmi Yarışma Deposu](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
