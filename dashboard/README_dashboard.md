# 🌊 Junon Dashboard - Guide d'Utilisation

Dashboard interactif pour l'analyse et la prédiction des niveaux piézométriques.

## 🚀 Installation

### 1. Installer les dépendances

```bash
cd junon-time-series
uv sync --extra dashboard --extra advanced
```

Ou avec pip :

```bash
pip install streamlit plotly statsmodels darts pytorch-lightning optuna
```

### 2. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

### 3. Ouvrir dans le navigateur

Le dashboard s'ouvrira automatiquement à l'adresse : `http://localhost:8501`

---

## 📄 Pages Disponibles

### 🏠 Page d'Accueil

Vue d'ensemble du projet avec :
- Résumé des 18 stations piézométriques
- Accès rapide aux fonctionnalités
- Statistiques globales

### 📊 1. Data Explorer

**Fonctionnalités :**
- Visualisation des séries temporelles (multi-stations, multi-variables)
- Graphiques interactifs Plotly (zoom, pan, hover)
- Statistiques descriptives
- Distributions et boxplots mensuels
- Export CSV

**Comment utiliser :**
1. Sélectionnez une ou plusieurs stations
2. Choisissez les variables à afficher (level, PRELIQ_Q, T_Q, ETP_Q)
3. Explorez les graphiques interactifs
4. Consultez les statistiques dans les onglets

### 📈 2. Statistical Analysis

**Fonctionnalités :**
- Tests de stationnarité (ADF, KPSS)
- Détection de saisonnalité (7, 30, 365 jours)
- Décomposition STL (tendance, saisonnalité, résidus)
- ACF/PACF
- Tests de normalité (Shapiro-Wilk)

**Comment utiliser :**
1. Sélectionnez une station
2. Parcourez les onglets pour voir les différents tests
3. Ajustez les paramètres (lags, périodes) selon vos besoins

### 🔗 3. Correlations

**Fonctionnalités :**
- Matrice de corrélation (heatmap)
- Cross-correlation (précipitation → niveau)
- Tests de causalité de Granger
- Analyse des lags optimaux

**Comment utiliser :**
1. Sélectionnez une station
2. Explorez la matrice de corrélation
3. Analysez le lag optimal dans l'onglet Cross-Correlation
4. Vérifiez la causalité de Granger pour chaque covariable

**Interprétation :**
- Lag optimal : délai entre la pluie et son effet sur la nappe
- p-value < 0.05 dans Granger : la covariable aide à prédire le niveau

### 🎯 4. Train Models

**La page la plus importante du dashboard !**

#### Tab 1 : Quick Train

**Fonctionnalités :**
- Entraînement rapide avec hyperparamètres manuels
- Support des 4 modèles : N-BEATS, TFT, TCN, LSTM
- Visualisation des métriques et prédictions
- Sauvegarde automatique des checkpoints

**Comment utiliser :**
1. Sélectionnez une station et un modèle
2. Ajustez les hyperparamètres dans l'expander
3. Cliquez sur "Lancer l'entraînement"
4. Attendez la fin (5-15 min selon le modèle)
5. Consultez les métriques et graphiques
6. Le modèle est automatiquement sauvegardé

**Hyperparamètres importants :**
- `input_chunk` : nombre de jours passés utilisés (défaut: 30)
- `output_chunk` : horizon de prédiction (défaut: 7)
- `n_epochs` : nombre d'époques d'entraînement
- `learning_rate` : taux d'apprentissage

#### Tab 2 : Optuna Optimization

**Fonctionnalités :**
- Optimisation automatique des hyperparamètres avec Optuna
- Recherche bayésienne intelligente
- Visualisation de l'historique d'optimisation
- Importance des hyperparamètres

**Comment utiliser :**
1. Sélectionnez une station et un modèle
2. Configurez le nombre d'essais (10-50 pour un test, 50-200 pour production)
3. Choisissez la métrique à optimiser (MAE recommandé)
4. Cliquez sur "Lancer l'optimisation"
5. Attendez la fin (peut prendre plusieurs heures)
6. Consultez les meilleurs hyperparamètres
7. Analysez les graphiques d'optimisation

**Recommandations :**
- **Test rapide** : 10-20 trials, timeout 1h
- **Production** : 50-100 trials, timeout 4-6h
- La première trial est souvent lente (compilation PyTorch)

### 🔮 5. Forecasting

**Fonctionnalités :**
- Chargement de modèles pré-entraînés
- Génération de prédictions interactives
- Calcul automatique des métriques
- Visualisation avec zoom
- Export CSV

**Comment utiliser :**
1. Sélectionnez un checkpoint (modèle entraîné)
2. Choisissez une station
3. Ajustez l'horizon de prédiction
4. Cliquez sur "Générer les prédictions"
5. Consultez les métriques et graphiques
6. Exportez les résultats si besoin

**Note :** Les checkpoints sont automatiquement détectés depuis `checkpoints/darts/`

### 📉 6. Model Comparison

**Fonctionnalités :**
- Comparaison de plusieurs modèles
- Tableaux de métriques avec highlights
- Bar charts par métrique
- Radar charts multi-métriques
- Boxplots de variance

**Comment utiliser :**
1. Les résultats sont chargés depuis `results/darts_comparison_complete.csv`
2. Filtrez les stations et modèles dans la sidebar
3. Parcourez les onglets pour différentes visualisations
4. Identifiez le meilleur modèle par métrique

**Prérequis :** Avoir exécuté le notebook `2_forecasting_models.ipynb` ou entraîné plusieurs modèles

---

## 📊 Métriques Expliquées

### Métriques Standards

- **MAE** (Mean Absolute Error) : Erreur moyenne en mètres
  - Plus bas = mieux
  - Facile à interpréter (même unité que la cible)

- **RMSE** (Root Mean Square Error) : Pénalise les grosses erreurs
  - Plus bas = mieux
  - Sensible aux outliers

- **MAPE** (Mean Absolute Percentage Error) : Erreur en pourcentage
  - Plus bas = mieux
  - Peut exploser si valeurs proches de 0

- **R²** (Coefficient de Détermination) : Qualité de l'ajustement
  - Plus haut = mieux (max = 1.0)
  - 1.0 = prédiction parfaite

### Métriques Bornées (Plus Interprétables)

- **sMAPE** (Symmetric MAPE) : Erreur symétrique (0-100%)
  - Plus bas = mieux
  - Évite les problèmes de MAPE avec valeurs proches de 0

- **NRMSE** (Normalized RMSE) : RMSE normalisé par la plage (0-1)
  - Plus bas = mieux
  - Permet de comparer entre stations

- **Dir_Acc** (Directional Accuracy) : % de prédictions avec bonne tendance
  - Plus haut = mieux (max = 100%)
  - Utile pour savoir si le modèle capte les variations

---

## 🎯 Workflow Recommandé

### Pour un Projet Complet

1. **Exploration** (Data Explorer)
   - Visualisez les données de toutes les stations
   - Identifiez les patterns et anomalies

2. **Analyse Statistique** (Statistical Analysis)
   - Vérifiez la stationnarité
   - Détectez la saisonnalité
   - Analysez la décomposition STL

3. **Corrélations** (Correlations)
   - Identifiez les lags optimaux
   - Vérifiez la causalité de Granger
   - Comprenez les relations entre variables

4. **Entraînement** (Train Models)
   - Commencez par Quick Train pour tester rapidement
   - Utilisez Optuna pour optimiser les hyperparamètres
   - Entraînez sur plusieurs stations

5. **Prédictions** (Forecasting)
   - Testez vos modèles entraînés
   - Comparez les performances

6. **Comparaison** (Model Comparison)
   - Analysez les résultats globaux
   - Identifiez le meilleur modèle

### Pour un Test Rapide

1. Data Explorer → Visualiser piezo1
2. Train Models (Quick Train) → Entraîner TFT sur piezo1 (30 epochs)
3. Forecasting → Tester le modèle
4. Model Comparison → (si résultats existants)

---

## 💡 Astuces et Bonnes Pratiques

### Performance

- Le **cache Streamlit** accélère le chargement des données (automatique)
- Les modèles TFT et TCN sont plus lents que N-BEATS et LSTM
- Pour tester rapidement, utilisez 10-20 epochs

### GPU

- Le dashboard détecte automatiquement le GPU
- Si GPU disponible : entraînement 3-5x plus rapide
- CPU fonctionne aussi, mais plus lent

### Hyperparamètres

- **input_chunk** :
  - Plus grand = modèle voit plus loin dans le passé
  - Recommandé : 30-60 jours pour les nappes phréatiques

- **output_chunk** :
  - Horizon de prédiction
  - Recommandé : 7-14 jours

- **learning_rate** :
  - Trop élevé : instabilité
  - Trop bas : convergence lente
  - Recommandé : 1e-3 à 1e-4

### Optuna

- Utilise la recherche bayésienne (plus intelligent que random search)
- Les premières trials explorent, les suivantes affinent
- Regardez l'historique d'optimisation pour voir la convergence

---

## 🐛 Troubleshooting

### "No module named dashboard"
- Vérifiez que vous êtes dans `junon-time-series/`
- Relancez avec : `streamlit run dashboard/app.py`

### Le dashboard est lent
- Vérifiez que le cache Streamlit fonctionne
- Réduisez le nombre d'epochs pour les tests

### GPU non détecté
- Vérifiez l'installation PyTorch : `python -c "import torch; print(torch.cuda.is_available())"`
- Si False, réinstallez PyTorch avec CUDA

### Erreur "FileNotFoundError"
- Vérifiez que les données sont dans `data/piezos/`
- Vérifiez les noms de fichiers : `piezo1.csv`, ..., `piezo18.csv`

### Optuna très lent
- Normal pour les premières trials (compilation JIT)
- Réduisez `n_epochs` à 20-30 pour accélérer
- Utilisez moins de trials pour tester

---

## 📚 Ressources

- **Darts Documentation** : https://unit8co.github.io/darts/
- **Streamlit Docs** : https://docs.streamlit.io/
- **Optuna Documentation** : https://optuna.readthedocs.io/
- **Notebooks du projet** : `notebooks/`

---

## 🆘 Support

Pour toute question ou problème :
1. Consultez ce README
2. Vérifiez les logs d'erreur dans le terminal
3. Ouvrez une issue sur GitHub

---

**Bonne exploration ! 🌊**
