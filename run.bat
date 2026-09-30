@echo off
chcp 65001 >nul
setlocal

rem Göreli yollar (data/, static/live_cache) için proje kökünden çalıştır
cd /d "%~dp0"

rem Varsa sanal ortamı etkinleştir
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

where python >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python bulunamadı. Python 3.10+ kurup PATH'e ekleyin.
    pause
    exit /b 1
)

python -c "import flask, flask_cors, cv2, numpy, PIL, onnxruntime, huggingface_hub, pyproj, tifffile" >nul 2>nul
if errorlevel 1 (
    echo Eksik bağımlılıklar yükleniyor...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [HATA] Bağımlılıklar yüklenemedi.
        pause
        exit /b 1
    )
)

echo Atlas GeoChange başlatılıyor: http://127.0.0.1:5000
echo Durdurmak için Ctrl+C
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

python app.py

pause
