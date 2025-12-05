# 🌊 Junon Dashboard - Applications Séparées

Le dashboard Junon est divisé en deux applications Streamlit totalement indépendantes.

## 📊 Application 1 : Data Explorer

Interface dédiée **uniquement** à l'exploration et l'analyse statistique des données piézométriques.

### Fonctionnalités (3 pages)
- **📊 Data Explorer** : Visualisation interactive des séries temporelles
- **📈 Statistical Analysis** : Tests de stationnarité (ADF, KPSS), ACF/PACF avec différenciation, détection d'outliers (Z-Score, IQR)
- **🔗 Correlations** : Matrices de corrélation, causalité de Granger, analyse des lags

### Lancement
```bash
python -m streamlit run dashboard/explorer/app.py --server.port 8520
```

**URL** : http://localhost:8520

## 🎯 Application 2 : Model Training

Interface dédiée **uniquement** à l'entraînement et l'évaluation de modèles de prévision.

### Fonctionnalités (3 pages)
- **🎯 Train Models** : Entraînement manuel ou optimisation Optuna automatique
- **🔮 Forecasting** : Prédictions interactives avec modèles entraînés
- **📉 Model Comparison** : Comparaison détaillée des performances

### Lancement
```bash
python -m streamlit run dashboard/training/app.py --server.port 8521
```

**URL** : http://localhost:8521

## 📁 Nouvelle Structure (Séparation complète)

```
dashboard/
├── explorer/                  # Application Data Explorer
│   ├── app.py                # Point d'entrée Explorer
│   └── pages/                # Pages Explorer UNIQUEMENT
│       ├── 1_📊_Data_Explorer.py
│       ├── 2_📈_Statistical_Analysis.py
│       └── 3_🔗_Correlations.py
│
├── training/                  # Application Model Training
│   ├── app.py                # Point d'entrée Training
│   └── pages/                # Pages Training UNIQUEMENT
│       ├── 1_🎯_Train_Models.py
│       ├── 2_🔮_Forecasting.py
│       └── 3_📉_Model_Comparison.py
│
├── utils/                     # Modules partagés par les deux apps
│   ├── data_loader.py
│   ├── statistics.py
│   ├── plots.py
│   ├── forecasting.py
│   └── optuna_utils.py
│
├── config.py                  # Configuration commune
└── .streamlit/               # Config Streamlit globale
    └── config.toml
```

## ✅ Séparation Totale

Les deux applications sont maintenant **complètement séparées** :
- **Data Explorer** : Ne contient QUE les pages d'exploration (plus de Train Models, Forecasting, etc.)
- **Model Training** : Ne contient QUE les pages de modélisation (plus de Data Explorer, etc.)

## 🚀 Workflow Recommandé

1. **Étape 1 : Exploration** (App Explorer - Port 8520)
   - Analyser et comprendre les données
   - Vérifier la qualité, les patterns
   - Identifier les corrélations

2. **Étape 2 : Modélisation** (App Training - Port 8521)
   - Entraîner les modèles
   - Optimiser avec Optuna
   - Comparer les performances

## ⚙️ Installation des Dépendances

```bash
# Depuis le répertoire racine junon-time-series
uv sync --extra dashboard --extra advanced
```

## 🐛 Troubleshooting

### Si vous voyez encore toutes les pages mélangées
Vérifiez que vous lancez bien depuis les nouveaux chemins :
- `dashboard/explorer/app.py` (et non dashboard/data_explorer_app.py)
- `dashboard/training/app.py` (et non dashboard/training_app.py)

### Port déjà utilisé
Fermez les anciennes instances ou utilisez d'autres ports :
```bash
python -m streamlit run dashboard/explorer/app.py --server.port 8530
python -m streamlit run dashboard/training/app.py --server.port 8531
```

## 📊 Avantages de cette Séparation

- **Clarté** : Chaque app a un rôle unique et clair
- **Performance** : Moins de pages = chargement plus rapide
- **Maintenance** : Plus facile de maintenir et déboguer
- **UX** : L'utilisateur n'est pas perdu avec trop d'options