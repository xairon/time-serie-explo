# 🌊 Time Series Exploration Dashboard

Dashboard interactif Streamlit pour l'analyse et la prévision de séries temporelles hydrologiques avec des modèles deep learning.

## 🎯 Description

Dashboard complet pour l'exploration, la modélisation et l'évaluation de modèles de prévision sur des données piézométriques (nappes phréatiques). Intègre des outils d'analyse statistique avancée, d'optimisation d'hyperparamètres, et d'explicabilité via SHAP.

### Modèles Supportés

- **N-BEATS** - Décomposition automatique tendance/saisonnalité
- **TFT** (Temporal Fusion Transformer) - Attention multi-horizon + interprétabilité
- **TCN** (Temporal Convolutional Network) - Rapide et performant
- **LSTM** - Baseline récurrente classique

### Fonctionnalités

✅ **Exploration interactive** des données temporelles  
✅ **Analyses statistiques** complètes (ADF, KPSS, STL, ACF/PACF)  
✅ **Entraînement de modèles** avec hyperparamètres personnalisés  
✅ **Optimisation Optuna** pour recherche automatique d'hyperparamètres  
✅ **Backtesting** robuste via fenêtre glissante  
✅ **Explicabilité SHAP** pour comprendre les prédictions  
✅ **Métriques hydrologiques** (NSE, KGE) en plus des métriques classiques  

## 📊 Données

**18 stations piézométriques françaises** (~30 ans de données historiques)

**Variables :**
- `level` - Niveau de la nappe (cible à prédire)
- `PRELIQ_Q` - Précipitations quotidiennes (mm)
- `T_Q` - Température moyenne (°C)
- `ETP_Q` - Évapotranspiration potentielle (mm)

Les données sont dans `data/piezos/piezo1.csv` à `piezo18.csv`.

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://scm.univ-tours.fr/ringuet/time-serie-explo.git
cd time-serie-explo
```

### 2. Setup automatique (recommandé)

**Windows :**
```bash
.\SETUP.bat
```

**Linux/Mac :**
```bash
chmod +x setup.sh
./setup.sh
```

Ces scripts installent automatiquement :
- `uv` (gestionnaire de packages ultra-rapide)
- Python 3.11 dans un environnement virtuel `.venv`
- Toutes les dépendances (PyTorch + Darts + Optuna + Streamlit + SHAP)
- Support GPU NVIDIA (CUDA 12.1) si disponible

### 3. Installation manuelle (alternative)

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# ou
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Créer l'environnement
uv sync --extra all

# Installer PyTorch avec CUDA 12.1 (GPU NVIDIA)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Activer l'environnement
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 🖥️ Lancer le Dashboard

```bash
# Activer l'environnement (si pas déjà fait)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Lancer Streamlit
streamlit run dashboard/app.py
```

Le dashboard s'ouvre automatiquement dans votre navigateur à `http://localhost:8501`.

## 📂 Structure du Projet

```
time-serie-explo/
├── dashboard/
│   ├── app.py                          # Page d'accueil du dashboard
│   ├── config.py                       # Configuration globale
│   ├── pages/
│   │   ├── 1_📊_Data_Explorer.py       # Visualisation des séries temporelles
│   │   ├── 2_📈_Statistical_Analysis.py # Tests statistiques (ADF, KPSS, STL)
│   │   ├── 3_🔗_Correlations.py        # Corrélations et causalité de Granger
│   │   ├── 4_🎯_Train_Models.py        # Entraînement + Optuna
│   │   ├── 5_🔮_Forecasting.py         # Prédictions interactives
│   │   ├── 6_📉_Model_Comparison.py    # Comparaison des modèles
│   │   ├── 7_🔄_Backtesting.py         # Validation historique
│   │   └── 8_💡_Explainability.py      # Analyse SHAP
│   └── utils/
│       ├── data_loader.py              # Chargement et préparation des données
│       ├── forecasting.py              # Pipeline d'entraînement et prédiction
│       ├── optuna_utils.py             # Optimisation Optuna
│       ├── plots.py                    # Visualisations Plotly
│       ├── statistics.py               # Tests statistiques
│       └── state.py                    # Gestion de l'état de session
│
├── data/
│   └── piezos/                         # 18 stations piézométriques
│
├── notebooks/                          # Notebooks d'analyse exploratoire
│   ├── 1_data_analysis_darts.ipynb
│   └── 3_darts_advanced_forecasting.ipynb
│
├── checkpoints/                        # Modèles entraînés (généré)
├── results/                            # Métriques CSV (généré)
├── logs/                               # TensorBoard logs (généré)
├── figs/                               # Graphiques (généré)
│
├── pyproject.toml                      # Dépendances du projet
├── uv.lock                             # Lock file pour reproductibilité
└── README.md                           # Ce fichier
```

## 🧭 Guide d'Utilisation

### 1. **📊 Data Explorer**
- Visualisez les séries temporelles de plusieurs stations simultanément
- Graphiques interactifs Plotly avec zoom, pan, export
- Sélection multiple de variables et stations

### 2. **📈 Statistical Analysis**
- **Tests de stationnarité** : ADF, KPSS
- **Décomposition STL** : Tendance + Saisonnalité + Résidus
- **ACF/PACF** : Autocorrélation et autocorrélation partielle
- **Tests de normalité** : Shapiro-Wilk, Q-Q plots

### 3. **🔗 Correlations**
- **Matrices de corrélation** entre stations
- **Cross-corrélation** (CCF) entre cible et covariables
- **Tests de causalité de Granger** : est-ce que la pluie cause le niveau ?
- **Analyse des lags optimaux**

### 4. **🎯 Train Models**
Deux modes d'entraînement :

**Quick Train** : Entraînement rapide avec hyperparamètres manuels
- Sélectionnez un modèle et une station
- Ajustez les hyperparamètres via sliders
- Visualisez les résultats immédiatement

**Optuna Optimization** : Recherche automatique des meilleurs hyperparamètres
- Définissez le nombre d'essais et la métrique à optimiser
- Optuna teste intelligemment différentes combinaisons
- Visualisez l'historique d'optimisation et l'importance des paramètres
- Entraînez le modèle final avec les meilleurs paramètres trouvés

### 5. **🔮 Forecasting**
- Chargez un modèle entraîné depuis les checkpoints
- Générez des prédictions sur le test set
- Visualisez les prédictions vs réalité
- Calculez automatiquement 9 métriques de performance

### 6. **📉 Model Comparison**
- Comparez les performances de plusieurs modèles
- Tableaux de métriques, graphiques comparatifs
- Radar charts pour visualisation multidimensionnelle
- Identifiez le meilleur modèle par station

### 7. **🔄 Backtesting**
- Validation robuste via fenêtre glissante
- Testez la stabilité de vos modèles sur différentes périodes
- Visualisez l'évolution des performances dans le temps
- Détectez les périodes où le modèle performe moins bien

### 8. **💡 Explainability**
- Comprenez **pourquoi** votre modèle prédit ce qu'il prédit
- **Importance globale** : quelles features sont les plus influentes ?
- **Explications locales** : qu'est-ce qui a causé cette prédiction précise ?
- Visualisations interactives : beeswarm, waterfall, dependence plots
- Fonctionne avec TFT, TCN, LSTM (N-BEATS peut échouer car très profond)

## 📊 Métriques de Performance

### Métriques Classiques
- **MAE** (Mean Absolute Error) - Erreur moyenne en mètres
- **RMSE** (Root Mean Square Error) - Pénalise les grosses erreurs
- **MAPE** (Mean Absolute Percentage Error) - Erreur en %
- **R²** (Coefficient de détermination) - 1.0 = ajustement parfait

### Métriques Bornées
- **sMAPE** (0-100%) - Erreur symétrique
- **NRMSE** (0-1) - Erreur normalisée par la plage
- **Dir_Acc** (0-100%) - % de tendances correctement prédites

### Métriques Hydrologiques
- **NSE** (Nash-Sutcliffe Efficiency) - Standard en hydrologie (1.0 = parfait)
- **KGE** (Kling-Gupta Efficiency) - Amélioration de NSE

## 🔧 Technologies Utilisées

- **Streamlit** (`>=1.28.0`) - Framework du dashboard
- **Darts** (`>=0.32.0`) - Librairie de forecasting
- **PyTorch** (`>=2.0.0`) - Deep learning backend
- **PyTorch Lightning** (`>=2.0.0`) - Framework d'entraînement
- **Optuna** (`>=3.0.0`) - Optimisation bayésienne
- **SHAP** (`>=0.42.0`) - Explicabilité des modèles
- **Plotly** (`>=5.15.0`) - Visualisations interactives
- **Pandas, NumPy, Scikit-learn** - Data science stack

Voir `pyproject.toml` pour la liste complète.

## 💡 Conseils d'Utilisation

### Pour l'Entraînement
1. Commencez par **Statistical Analysis** pour comprendre vos données
2. Utilisez **Optuna Optimization** pour trouver les meilleurs hyperparamètres (10-30 trials suffisent)
3. Entraînez le modèle final avec ces hyperparamètres
4. Si vous avez un GPU, préférez TFT ou TCN (plus rapides que LSTM)

### Pour l'Évaluation
1. Vérifiez plusieurs métriques (ne vous fiez pas qu'au MAE)
2. Pour l'hydrologie, privilégiez NSE et KGE
3. Utilisez le **Backtesting** pour valider la robustesse
4. L'**Explainability** SHAP aide à détecter les artefacts (ex: le modèle se base trop sur un lag spécifique)

### Performances
- **N-BEATS** : Excellent pour la saisonnalité, mais lent
- **TFT** : Meilleur pour l'interprétabilité, supporte les covariates
- **TCN** : Excellent compromis vitesse/performance
- **LSTM** : Baseline de référence, souvent surpassé

## 📚 Références

- **Darts** - [GitHub](https://github.com/unit8co/darts) | [Documentation](https://unit8co.github.io/darts/)
- **TFT** - Lim et al. (2021) - [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- **N-BEATS** - Oreshkin et al. (2020) - [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
- **TCN** - Bai et al. (2018) - [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
- **SHAP** - Lundberg & Lee (2017) - [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- **Optuna** - [Documentation](https://optuna.org/)
- **Streamlit** - [Documentation](https://docs.streamlit.io/)

## 👤 Auteur

Nicolas Ringuet - [nicolas.ringuet@univ-tours.fr](mailto:nicolas.ringuet@univ-tours.fr)

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)
