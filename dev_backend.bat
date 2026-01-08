@echo off
SETLOCAL
cd /d "%~dp0"

echo ==========================================
echo   Junon Time Series - Local Dev Backend
echo ==========================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: 2. Check/Create Venv
if not exist "venv" (
    echo [INFO] Creating venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Using existing venv.
)

:: 3. Activate Venv
call venv\Scripts\activate.bat

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install Dependencies
echo [INFO] Installing dependencies (this may take a while first time)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 6. Setup Local Environment Variables if needed
:: (Config defaults to localhost:5433/postgres/postgres/postgres_default_pass_2024 which matches dev)

:: 7. Start Backend
echo [INFO] Starting Uvicorn...
cd backend
python -m uvicorn app.main:app --reload --port 8000 --host 127.0.0.1
