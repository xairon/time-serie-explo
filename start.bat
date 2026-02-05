@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================
echo  Time-Serie-Explo Launcher
echo ================================================

:: Check for backend marker
if exist ".venv\.backend" (
    set /p BACKEND=<.venv\.backend
) else (
    echo No environment configured. Running deploy.py...
    python deploy.py
    if exist ".venv\.backend" (
        set /p BACKEND=<.venv\.backend
    ) else (
        echo ERROR: Deployment failed.
        pause
        exit /b 1
    )
)

echo.
echo Backend: %BACKEND%
echo.

:: Activate oneAPI for XPU
if "%BACKEND%"=="xpu" (
    echo Activating Intel oneAPI...
    if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
        call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul 2>&1
    ) else if exist "C:\Program Files\Intel\oneAPI\setvars.bat" (
        call "C:\Program Files\Intel\oneAPI\setvars.bat" > nul 2>&1
    ) else (
        echo WARNING: oneAPI not found. XPU may not work.
    )
)

:: Verify torch installation
echo.
echo Checking PyTorch installation...
.venv\Scripts\python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not working. Try: python deploy.py --force-rebuild
    pause
    exit /b 1
)

:: Check device availability
.venv\Scripts\python check_gpu.py
echo.

:: Launch services
echo ================================================
echo  Launching Services
echo ================================================
echo MLflow:    http://localhost:5000
echo Streamlit: http://localhost:8501
echo.

start "MLflow" /min cmd /c ".venv\Scripts\python -m mlflow server --host 0.0.0.0 --port 5000"
timeout /t 2 /nobreak > nul
start "Streamlit" cmd /c ".venv\Scripts\python -m streamlit run dashboard/training/Home.py --server.port 8501"

echo.
echo Services started. Press any key to exit this window.
pause > nul
