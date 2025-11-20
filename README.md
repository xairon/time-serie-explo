# 🌊 Junon Time Series - Prévision Piézométrique avec Darts

Projet de Deep Learning pour prédire les niveaux d'eau souterraine (nappes phréatiques) en utilisant la librairie **Darts**.

## 🎯 Description

Prévision des niveaux piézométriques avec des modèles deep learning state-of-the-art :
- **TFT** (Temporal Fusion Transformer) - Attention multi-horizon + interprétabilité
- **N-BEATS** - Décomposition tendance/saisonnalité automatique
- **TCN** (Temporal Convolutional Network) - Rapide et performant
- **LSTM** - Baseline récurrente classique

### Pourquoi Darts ?

Les nappes phréatiques ont des caractéristiques spécifiques :
- **Forte autocorrélation** - Le niveau d'aujourd'hui dépend d'hier
- **Influences exogènes** - Précipitations, température, évapotranspiration
- **Délais temporels** - La pluie influence la nappe avec 2-3 semaines de retard
- **Saisonnalité** - Cycles annuels de recharge/décharge

Darts offre une gestion native des covariates, un backtesting automatique, et une API unifiée pour plus de 40 modèles.

## 📊 Données

**18 stations piézométriques françaises** (~30 ans de données historiques)

**Features :**
- `level` - Niveau de la nappe (cible)
- `PRELIQ_Q` - Précipitations (mm)
- `T_Q` - Température (°C)
- `ETP_Q` - Évapotranspiration potentielle (mm)

Les données sont dans `data/piezos/piezoX.csv` (X = 1 à 18)

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

Ces scripts :
1. Installent `uv` (gestionnaire de packages ultra-rapide)
2. Créent l'environnement virtuel `.venv`
3. Installent toutes les dépendances (PyTorch + Darts + Optuna)
4. Configurent PyTorch avec support GPU NVIDIA (CUDA 12.1)

### Installation manuelle

```bash
# 1. Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# ou
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Créer l'environnement et installer les dépendances
uv sync --extra all

# 3. Installer PyTorch avec CUDA 12.1 (GPU NVIDIA)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Activer l'environnement
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 📓 Utilisation

Le projet est organisé en **deux notebooks complémentaires** :

### 1. Analyse des Données (`1_data_analysis_darts.ipynb`)

**Objectif** : Analyse statistique complète des séries temporelles avec tous les outils Darts.

**Contenu** :
- Tests de stationnarité (ADF, KPSS)
- Détection de saisonnalité (hebdomadaire, mensuelle, annuelle)
- Décomposition STL (tendance + saisonnalité + résidus)
- ACF, PACF (autocorrélation)
- CCF (cross-corrélation avec covariates)
- Tests de causalité de Granger
- Analyse des délais optimaux
- Distribution et outliers
- Analyse des résidus

### 2. Modèles de Prévision (`2_forecasting_models.ipynb`)

**Objectif** : Entraîner et comparer 4 modèles deep learning state-of-the-art.

**Contenu** :
1. **Setup et imports** - Configuration PyTorch Lightning
2. **Chargement des données** - 18 stations piézométriques
3. **Préparation** - Format Darts, split train/val/test, normalisation
4. **Configuration** - Hyperparamètres par défaut recommandés
5. **Entraînement** - N-BEATS, TFT, TCN, LSTM
6. **Métriques** - Standards (MAE, RMSE, R²) + bornées (sMAPE, NRMSE, Dir_Acc)
7. **Évaluation** - Comparaison sur le test set
8. **Visualisations** - Graphiques de performances
9. **Résumé** - Meilleurs modèles et fichiers générés

**Note** : Ce notebook utilise des hyperparamètres par défaut. Pour l'optimisation Optuna, voir une version future.

### Lancer les notebooks

**Avec VSCode** (recommandé) :
1. Ouvrir le notebook
2. Sélectionner le kernel : `.venv/bin/python` (ou `.venv\Scripts\python.exe` sur Windows)
3. Run All

**Avec Jupyter Lab** :
```bash
jupyter lab
```

### Résultats

Tous les résultats sont sauvegardés automatiquement :

- **Métriques** : `results/darts_comparison_complete.csv`
- **Visualisations** : `figs/darts_*.png`
- **Checkpoints** : `checkpoints/darts/{model}/`
- **Logs TensorBoard** : `logs/darts/`

Pour visualiser les logs d'entraînement :
```bash
tensorboard --logdir logs/darts
```

## 📂 Structure du projet

```
junon-time-series/
├── data/
│   └── piezos/              # 18 stations piézométriques (piezo1.csv à piezo18.csv)
│
├── notebooks/
│   ├── 1_data_analysis_darts.ipynb      # Analyse statistique complète
│   └── 2_forecasting_models.ipynb       # Entraînement des modèles
│
├── pyproject.toml           # Dépendances (uv/pip)
├── uv.lock                  # Lock file pour reproductibilité
├── setup.sh                 # Setup automatique Linux/Mac
├── SETUP.bat                # Setup automatique Windows
├── .python-version          # Python 3.11
│
├── results/                 # Métriques et résultats (généré)
├── figs/                    # Graphiques (généré)
├── checkpoints/             # Modèles sauvegardés (généré)
└── logs/                    # TensorBoard logs (généré)
```

## 🔧 Dépendances principales

- **Darts** (`>=0.32.0`) - Librairie de forecasting
- **PyTorch** (`>=2.0.0`) - Deep learning backend
- **PyTorch Lightning** (`>=2.0.0`) - Framework d'entraînement
- **Optuna** (`>=3.0.0`) - Optimisation bayésienne des hyperparamètres
- **TensorBoard** (`>=2.13.0`) - Visualisation des métriques
- **Pandas, NumPy, Matplotlib, Seaborn** - Data science stack

Voir `pyproject.toml` pour la liste complète.

## 📈 Métriques utilisées

### Métriques standards
- **MAE** (Mean Absolute Error) - Erreur moyenne en mètres
- **RMSE** (Root Mean Square Error) - Pénalise les grosses erreurs
- **MAPE** (Mean Absolute Percentage Error) - Erreur en %
- **R²** (Coefficient de détermination) - Qualité de l'ajustement (1.0 = parfait)

### Métriques bornées (plus interprétables)
- **sMAPE** (0-100%) - Erreur symétrique
- **NRMSE** (0-1) - Erreur normalisée par la plage
- **Directional Accuracy** (0-100%) - % de prédictions avec la bonne tendance

## 🧠 Modèles

### TFT (Temporal Fusion Transformer)
- Architecture encoder-decoder avec attention multi-head
- Variable selection networks (apprend l'importance de chaque feature)
- Gère automatiquement les délais temporels (pluie → nappe)
- **Idéal pour** : Interprétabilité + covariates multiples

### N-BEATS (Neural Basis Expansion Analysis)
- Décomposition automatique tendance + saisonnalité
- Architecture doubly residual
- Pas de preprocessing nécessaire
- **Idéal pour** : Séries avec forte saisonnalité

### TCN (Temporal Convolutional Network)
- Convolutions causales 1D avec dilations exponentielles
- Champ réceptif large (regarde loin dans le passé)
- Très rapide à entraîner
- **Idéal pour** : Compromis performance/vitesse

### LSTM (Long Short-Term Memory)
- Réseau récurrent classique
- Mémoire à long et court terme
- **Idéal pour** : Baseline de référence

## 🎯 Workflow

### 1. Analyse des données (`1_data_analysis_darts.ipynb`)

Comprendre les caractéristiques statistiques des séries temporelles :
- Stationnarité et transformations nécessaires
- Périodes de saisonnalité dominantes
- Relations entre covariates et cible
- Délais temporels optimaux

### 2. Modélisation (`2_forecasting_models.ipynb`)

Entraînement et comparaison des modèles :
- Split temporel : 50% train / 10% val / 40% test
- Normalisation (StandardScaler)
- Entraînement avec early stopping
- Évaluation sur 7 métriques
- Visualisations des prédictions

**Note** : Le notebook actuel utilise des hyperparamètres par défaut recommandés. Une future version intégrera **Optuna** pour l'optimisation automatique.

## 🌍 Carbon Footprint

Le projet utilise **CodeCarbon** pour tracker automatiquement l'empreinte carbone de l'entraînement.
Résultats sauvegardés dans `emissions.csv`.

Pour désactiver : modifier `use_codecarbon = False` dans le notebook.

## 📚 Références

- **Darts** - Unit8 - [GitHub](https://github.com/unit8co/darts) | [Docs](https://unit8co.github.io/darts/)
- **TFT** - Lim et al. (2021) - [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- **N-BEATS** - Oreshkin et al. (2020) - [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
- **TCN** - Bai et al. (2018) - [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
- **Optuna** - [Documentation](https://optuna.org/)

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)
