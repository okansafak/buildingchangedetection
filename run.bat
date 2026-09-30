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

rem Port başka bir uygulamada (ör. Docker) kullanılıyorsa istekler o uygulamaya gidebilir;
rem PORT verilmemişse 5000'den başlayarak ilk boş portu seç
if not defined PORT (
    set PORT=5000
    call :find_free_port
)
if errorlevel 1 (
    echo [HATA] 5000-5100 arasında boş port bulunamadı.
    pause
    exit /b 1
)

echo Atlas GeoChange başlatılıyor: http://127.0.0.1:%PORT%
echo Durdurmak için Ctrl+C
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%PORT%"

python app.py

pause
exit /b

:find_free_port
rem netstat -ano hem IPv4 hem IPv6 dinleyicilerini listeler
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if errorlevel 1 exit /b 0
echo Port %PORT% kullanımda, sonraki port deneniyor...
set /a PORT+=1
if %PORT% gtr 5100 exit /b 1
goto :find_free_port
