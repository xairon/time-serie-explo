@echo off
echo ================================================
echo SETUP JUNON TIME SERIES - WINDOWS
echo ================================================
echo.

REM Vérifier si .venv existe et demander de le supprimer manuellement
if exist .venv (
    echo ATTENTION: Un environnement .venv existe deja.
    echo Pour eviter les conflits, veuillez le supprimer manuellement:
    echo.
    echo    rmdir /s /q .venv
    echo.
    echo Puis relancez ce script.
    echo.
    pause
    exit /b 1
)

REM Installer uv si pas deja fait
echo [1/3] Installation de uv...
powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
echo.

REM Créer l'environnement et installer les dépendances (sans PyTorch)
echo [2/3] Creation environnement + installation dependances...
echo (utilise Python 3.12 pour compatibilite PyTorch CUDA)
uv sync --extra notebooks --python 3.12
echo.

REM Installer PyTorch avec CUDA (GPU NVIDIA)
echo [3/3] Installation PyTorch avec support GPU...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo ================================================
echo Installation terminee!
echo ================================================
echo.
echo Verification GPU:
.venv\Scripts\python.exe -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
echo.
echo Dans VSCode:
echo 1. Ouvrir notebooks/1_train_baselines.ipynb
echo 2. Cliquer sur "Select Kernel" en haut a droite
echo 3. Choisir Python: .venv\Scripts\python.exe
echo 4. Run All
echo.
echo ================================================
pause
