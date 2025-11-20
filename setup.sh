#!/bin/bash

echo "================================================"
echo "SETUP JUNON TIME SERIES - LINUX/MAC"
echo "================================================"
echo ""

# Vérifier si .venv existe et le supprimer
if [ -d ".venv" ]; then
    echo "ATTENTION: Suppression de l'environnement .venv existant..."
    rm -rf .venv
    echo ""
fi

# Installer uv si pas déjà fait
echo "[1/3] Installation de uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
echo ""

# Créer l'environnement et installer les dépendances (sans PyTorch)
echo "[2/3] Création environnement + installation dépendances..."
echo "(utilise Python 3.12 pour compatibilité PyTorch CUDA)"
echo "Installation: dépendances de base + notebooks + advanced (darts, nixtla) + dashboard"
uv sync --extra notebooks --extra advanced --extra dashboard --python 3.12
echo ""

# Installer PyTorch avec CUDA (GPU NVIDIA)
echo "[3/3] Installation PyTorch avec support GPU..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

echo "================================================"
echo "Installation terminée!"
echo "================================================"
echo ""
echo "Vérification GPU:"
.venv/bin/python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
echo ""
echo "Dans VSCode:"
echo "1. Ouvrir notebooks/1_train_baselines.ipynb"
echo "2. Cliquer sur 'Select Kernel' en haut à droite"
echo "3. Choisir Python: .venv/bin/python"
echo "4. Run All"
echo ""
echo "Pour lancer le dashboard:"
echo "  streamlit run dashboard/app.py"
echo ""
echo "================================================"
