@echo off
setlocal

call conda activate teknofest_yarisma
if errorlevel 1 (
    echo [HATA] Conda ortami acilamadi.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [KONTROL] Python ve kritik dosyalar kontrol ediliyor...

python -m py_compile main.py src\object_detection_model.py src_custom\reference_matcher.py
if errorlevel 1 (
    echo [HATA] Python syntax kontrolu basarisiz.
    pause
    exit /b 1
)

if not exist "models\best.pt" (
    echo [HATA] models\best.pt bulunamadi.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [HATA] .env bulunamadi.
    pause
    exit /b 1
)

echo [KONTROL] Model yukleme testi...

python -c "from src.object_detection_model import ObjectDetectionModel; m=ObjectDetectionModel('http://offline.test/'); assert m.model is not None; print('[OK] Model:', m.model_path)"
if errorlevel 1 (
    echo [HATA] Model yuklenemedi.
    pause
    exit /b 1
)

echo.
echo [UYARI] Bir sonraki adim gercek yarisma sunucusuna baglanir.
choice /C EH /N /M "Devam edilsin mi? [E=Evet, H=Hayir]: "

if errorlevel 2 (
    echo Baslatma iptal edildi.
    exit /b 0
)

echo.
echo [BASLATILIYOR] Yarışma istemcisi aciliyor...
python main.py

if errorlevel 1 (
    echo.
    echo [HATA] Program hata koduyla kapandi.
    pause
    exit /b 1
)

echo.
echo [TAMAMLANDI] Program normal sekilde kapandi.
pause