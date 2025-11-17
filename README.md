# Junon Time Series

Benchmark de prévision de séries temporelles pour données piézométriques (niveaux d'eau souterraine).

## 🎯 Description

Projet de Deep Learning pour prédire les niveaux d'eau souterraine en utilisant 3 modèles:
- **DLinear** : Décomposition linéaire (rapide, baseline)
- **PatchTST** : Transformer sur patchs (performant)
- **PatchMixer** : Convolutions + MLP-Mixer (compromis)

**Données** : 18 stations piézométriques françaises (~30 ans de données)
**Features** : level, PRELIQ_Q (précipitations), T_Q (température), ETP_Q (évapotranspiration)

## 🚀 Installation

### Setup automatique (recommandé)

**Windows :**
```bash
.\SETUP.bat
```

**Linux/Mac :**
```bash
chmod +x setup.sh
./setup.sh
```

Ces scripts installent uv, créent le `.venv`, et installent toutes les dépendances (y compris PyTorch avec support GPU NVIDIA).

### Installation manuelle

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# ou
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Installer les dépendances (sans PyTorch)
uv sync --extra notebooks

# Installer PyTorch avec CUDA 12.1 (GPU NVIDIA)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Activer l'environnement
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 📊 Utilisation

### Entraînement rapide

```bash
python main.py
```

Résultats dans `results/` et `figs/`

### Configuration

Choisir un fichier de configuration dans `configs/` :

```bash
# DLinear
python main.py --config configs/linearModel.json

# PatchTST
python main.py --config configs/patchTST.json
```

Ou éditer directement `configs/linearModel.json` ou `configs/patchTST.json` :

```json
{
  "model": "DLinear",          // ou PatchTST, PatchMixer
  "data": "piezos/piezo1.csv", // choisir un piézomètre
  "input_len": 15,             // historique
  "pred_len": 30,              // horizon
  "epochs": 100,
  "batch_size": 16
}
```

### Notebooks (VSCode recommandé)

1. Ouvrir `notebooks/1_train_baselines.ipynb` dans VSCode
2. Sélectionner le kernel : `.venv/bin/python` (ou `.venv\Scripts\python.exe` sur Windows)
3. Run All

**Notebooks disponibles :**
- `1_train_baselines.ipynb` : Entraîne les 3 modèles et compare
- `2_explain_model.ipynb` : Charge un modèle et analyse avec Grad-CAM + SHAP

**Alternative avec Jupyter Lab :**
```bash
jupyter lab
```

## 📂 Structure

```
junon-time-series/
├── configs/                 # Fichiers de configuration
│   ├── linearModel.json     # Config DLinear
│   └── patchTST.json        # Config PatchTST
├── main.py                  # Point d'entrée
├── pyproject.toml           # Dépendances (uv/pip)
│
├── data/piezos/             # 18 stations
├── models/                  # DLinear, PatchTST, PatchMixer
├── exp/                     # Train/test logic
├── tools/                   # Métriques, plots, gradcam
│
└── notebooks/               # Jupyter
    ├── 1_train_baselines.ipynb
    └── 2_explain_model.ipynb
```

## 📈 Métriques

- **MAE** : Mean Absolute Error
- **RMSE** : Root Mean Squared Error
- **KGE** : Kling-Gupta Efficiency (hydrologie)

## 🔬 Explicabilité

Le projet intègre plusieurs méthodes d'interprétabilité :

1. **Grad-CAM** : Importance temporelle des timesteps
2. **Gradient × Input** : Importance par feature
3. **SHAP** : Explications model-agnostic (optionnel)

Voir `notebooks/2_explain_model.ipynb`

## 🌍 CodeCarbon

Suivi automatique de l'empreinte carbone pendant l'entraînement → `emissions.csv`

## 📚 Références

- **PatchTST**: Nie et al. (2022) - [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- **DLinear**: Zeng et al. (2022) - [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
- **RevIN**: Kim et al. (2021) - [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p)

## 📄 Licence

MIT
