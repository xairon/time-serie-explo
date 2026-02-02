# Segmentation de performance et explicabilité par segment

Document de référence : bibliographie et cahier des charges pour un outil d’extraction de segments (bon / moyen / mauvais) sur le jeu de test, en vue d’explicabilité ciblée.

---

## 1. Bibliographie

### 1.1 Synthèse

La littérature ne formalise pas un outil unique « extraction de segments bons/moyens/mauvais pour l’explicabilité », mais plusieurs courants s’en rapprochent :

- **Évaluation par aspects / conditions** : ne pas se contenter d’une métrique agrégée ; évaluer selon des conditions (horizon, stationnarité, anomalies, etc.) pour voir *quand* le modèle performe ou échoue.
- **Explicabilité multi-granularité / par régime** : explications qui varient selon le contexte temporel ou le régime de la série, et pas seulement une explication globale ou une seule prédiction.
- **Monitoring / robustesse** : identifier les périodes ou les cas où la performance se dégrade (proche de la notion de « segments mauvais »).

L’idée d’extraire des **segments temporels** où le modèle est bon, moyen ou mauvais, puis d’y appliquer l’explicabilité (ex. TimeSHAP), s’inscrit dans la combinaison de ces trois axes.

### 1.2 Références principales

#### Évaluation par aspects et « quand le modèle échoue »

- **Cerqueira, V., Roque, L., & Soares, C. (2025). ModelRadar: aspect-based forecast evaluation.** *Machine Learning*, 114, 229.  
  https://doi.org/10.1007/s10994-025-06877-z  
  - Cadre d’évaluation des modèles de prévision **par aspects** : stationnarité, anomalies, horizon, difficulté du problème, etc.  
  - Montre que les métriques agrégées masquent les cas où les modèles échouent (ex. PatchTST meilleur en global mais moins robuste aux anomalies ; classiques meilleurs en one-step).  
  - **Lien direct** : segmenter la performance (bon/moyen/mauvais) selon des conditions ou des fenêtres est une forme d’évaluation « aspect-based » au niveau temporel.  
  - Package Python : https://github.com/vcerqueira/modelradar  

- **Cerqueira, V., Torgo, L., & Soares, C. (2024). Forecasting with deep learning: Beyond average of average of average performance.** In *International Conference on Discovery Science*, pp. 135–149. Springer.  
  - Insiste sur le fait qu’il faut aller au-delà des moyennes de moyennes pour comprendre les forces/faiblesses des modèles selon les conditions.

#### Explicabilité séries temporelles et liens avec la performance

- **PAX-TS (multi-granular explanations).**  
  - Méthode d’explicabilité *post-hoc* par perturbations localisées ; explications à plusieurs niveaux.  
  - Les explications des algorithmes performants vs peu performants diffèrent sur les mêmes données ; des patterns d’explication sont corrélés aux erreurs de prévision.  
  - Référence : *PAX-TS: Model-agnostic multi-granular explanations for time series forecasting via localized perturbations* (arXiv 2508.18982 / MLR 2025).  
  - **Lien** : une fois des segments « mauvais » extraits, des méthodes du type PAX-TS (ou TimeSHAP) peuvent être appliquées spécifiquement sur ces segments pour comprendre le comportement du modèle.

- **TimeSHAP: Explaining Recurrent Models through Sequence Perturbations.**  
  - Explicabilité SHAP pour modèles récurrents/séquentiels ; attributions par feature, par pas de temps, et (selon implémentation) par cellule.  
  - Local (une prédiction) et global (agrégation sur plusieurs instances).  
  - **Lien** : l’outil envisagé peut fournir les **segments** (périodes) sur lesquels lancer TimeSHAP (ex. seulement les fenêtres « mauvaises » ou « bonnes ») pour une analyse ciblée.  
  - Implémentation : https://github.com/feedzai/timeshap  

#### Régimes et incertitude

- **Adaptive regime-switching forecasts (DS3M + conformal prediction).**  
  - Travaux sur les changements de régime et les intervalles de prévision adaptatifs.  
  - **Lien** : la notion de « régime » (ou segment) où le modèle est fiable ou non rejoint l’idée de segments bon/moyen/mauvais.  

- **“On Identifying Why and When Foundation Models Perform Well on Time-Series Forecasting” (arXiv 2508.20437).**  
  - Analyse « quand » et « pourquoi » les modèles performent bien selon le domaine (ex. volatilité, type de série).  
  - **Lien** : même philosophie que la segmentation par performance (identifier les conditions/segments où le modèle est bon ou mauvais).

#### Bonnes pratiques d’évaluation

- **Hewamalage, H., Ackermann, K., & Bergmeir, C. (2022). Forecast evaluation for data scientists: Common pitfalls and best practices.** *Data Mining and Knowledge Discovery*, 37(2), 788–832.  
  - Rappel des pièges des métriques agrégées et de l’importance d’évaluer selon le contexte (horizon, type de série, etc.).

### 1.3 Positionnement de l’outil envisagé

- **Pas un doublon de ModelRadar** : ModelRadar segmente par *aspects* (stationnarité, anomalies, horizon, etc.) au niveau série ou observation. Notre outil segmente par **fenêtres temporelles** sur le test, en classant chaque fenêtre en bon/moyen/mauvais selon des seuils de métriques, puis en regroupant en segments consécutifs.
- **Complément à TimeSHAP/PAX-TS** : ces méthodes expliquent une prédiction ou un ensemble d’instances. L’outil fournit **quelles** fenêtres/segments prioriser pour l’explicabilité (ex. uniquement les segments mauvais ou bons).
- **Alignement** : évaluation « aspect-based » au niveau temporel + ciblage de l’explicabilité sur des segments de performance définis.

---

## 2. Cahier des charges (v1)

### 2.1 Objectif

- Parcourir **toutes les fenêtres de prédiction** possibles sur le **jeu de test** (sliding window).
- Pour chaque fenêtre : calculer les métriques (MAPE, NRMSE, NSE, KGE, etc.) et **classer** la fenêtre en **bon** / **moyen** / **mauvais** selon des seuils configurables.
- **Regrouper** les fenêtres consécutives de même classe pour obtenir des **segments** (périodes) « bon », « moyen », « mauvais ».
- Exposer ces segments (et les métriques par fenêtre) pour :
  - visualisation (timeline, cartes de performance) ;
  - **explicabilité ciblée** : lancer TimeSHAP (ou autre) sur des segments choisis (ex. seulement « mauvais » ou « bon »).

Aucun entraînement ni modification du modèle ; uniquement évaluation et post-traitement sur le test.

### 2.2 Contraintes et stack actuelle

- **Framework** : Darts (séries `TimeSeries`), modèles `ForecastingModel`.
- **Évaluation** : même protocole que le pipeline actuel :
  - `historical_forecasts` sur `full_series = train+val+test`, `start = début du test`, `forecast_horizon = output_chunk_length`, `stride` configurable (ex. `stride = horizon` pour fenêtres non chevauchantes, ou `< horizon` pour fenêtres chevauchantes si besoin).
- **Métriques** : réutiliser `calculate_metrics` de `dashboard/utils/training.py` (MAE, RMSE, MAPE, sMAPE, WAPE, NRMSE, NSE, KGE, Dir_Acc) sur chaque fenêtre (actual vs predicted), après `slice_intersect` et éventuel `inverse_transform` si un `target_scaler` est fourni.
- **Seuils** : s’appuyer sur `dashboard/config.py` (`METRICS`, `METRICS_INFO`) et sur les seuils indicatifs déjà affichés en Forecasting (ex. MAPE/NRMSE ≤ 10 % bon, 10–20 % moyen, > 20 % faible ; NSE/KGE > 0.75 bon, etc.). Les seuils doivent être **configurables** (fichier de config ou paramètres de l’outil).
- **Explicabilité** : intégration avec le flux actuel TimeSHAP (`dashboard/utils/timeshap_wrapper.py`, `explainability.py`) : l’outil produit des **segments** (dates ou indices de début/fin) et optionnellement les **fenêtres** associées (séries, covariables) pour alimenter un appel TimeSHAP sur une sélection (ex. « seulement les segments mauvais »).
- **Mode global / indépendant** : gérer comme le reste du dashboard les cas **global** (une série concaténée / une liste de séries) et **indépendant** (une série par station) ; pour l’indépendant, les segments sont par série (station).

### 2.3 Entrées

- **Modèle** : instance Darts entraînée (`ForecastingModel`).
- **Données** :
  - `full_train` : série(s) train+val (ou train seul si pas de val).
  - `test` : série(s) de test.
  - `past_covariates` (optionnel) : covariables passées, même format que pour `historical_forecasts`.
  - `target_scaler` (optionnel) : pour remettre les prédictions et la cible en échelle originale avant calcul des métriques.
- **Paramètres** :
  - `horizon` : égal à `output_chunk_length` du modèle.
  - `stride` : pas entre deux fenêtres (défaut : `horizon` pour fenêtres non chevauchantes).
  - `metrics_list` : sous-ensemble des métriques utilisées pour le classement (ex. MAPE, NRMSE, NSE, KGE).
  - **Règle de classement** : quelle métrique principale (ou combinaison) et quels seuils pour bon / moyen / mauvais (ex. MAPE seul ; ou NSE+KGE avec seuils cohérents avec la page Forecasting).
- **Option** : fenêtres chevauchantes (`stride < horizon`) pour une résolution plus fine des segments (plus de points, segments plus courts).

### 2.4 Sorties

- **Par fenêtre** :
  - Position temporelle (début/fin de la fenêtre de prédiction, ou index).
  - Métriques calculées (toutes celles demandées).
  - Label : `bon` | `moyen` | `mauvais`.
- **Segments** :
  - Liste de segments : `{ "label": "bon"|"moyen"|"mauvais", "start": datetime ou index, "end": datetime ou index, "window_indices": [i1, i2, ...] }`.
  - Optionnel : statistiques par segment (métriques moyennes, min, max sur les fenêtres du segment).
- **Format** : structuré (dict, DataFrame, ou petit objet métier) pour :
  - affichage dans le dashboard (timeline, filtres par label) ;
  - export (JSON/CSV) pour analyse externe ;
  - passage à un module d’explicabilité (ex. « segments mauvais » → choix de fenêtres pour TimeSHAP).

### 2.5 Règles de classement (bon / moyen / mauvais)

- **Par défaut** : aligné sur les seuils indicatifs de la page Forecasting, par exemple :
  - **MAPE / WAPE / NRMSE** (en %) : bon ≤ 10 %, moyen 10–20 %, mauvais > 20 %.
  - **NSE** : bon > 0.75, moyen 0.5–0.75, faible 0–0.5, mauvais < 0 → on peut fusionner « faible » et « mauvais » en « mauvais » selon le besoin.
  - **KGE** : bon > 0.75, moyen 0.5–0.75, mauvais ≤ 0.5.
- **Règle composite** : possibilité de définir une règle unique à partir de plusieurs métriques (ex. « mauvais si MAPE > 20 % OU NSE < 0 » ; « bon si NSE > 0.75 ET MAPE ≤ 10 % »). La v1 peut se limiter à **une métrique principale** configurable + seuils à trois niveaux.
- Les seuils doivent être **paramétrables** (config ou arguments) pour s’adapter au domaine (hydrologie piézométrique ici).

### 2.6 Intégration pipeline et UI

- **Pipeline** :
  - Nouvelle fonction (ou petit module) dans `dashboard/utils/` (ex. `performance_segmentation.py`) qui :
    - appelle `historical_forecasts` avec les mêmes conventions que `evaluate_model_sliding` ;
    - pour chaque fenêtre, appelle `calculate_metrics` puis applique la règle de classement ;
    - fusionne les fenêtres consécutives de même label en segments ;
    - retourne fenêtres + segments (+ optionnellement séries découpées pour explicabilité).
  - Réutilisation de `evaluate_model_sliding` possible pour la logique `historical_forecasts` + découpage, en étendant le retour pour garder les métriques **par fenêtre** au lieu d’agréger.
- **UI (Forecasting)** :
  - Section dédiée « Segmentation de performance » (ou onglet) :
    - choix de la métrique de classement et des seuils (ou profil prédéfini) ;
    - lancement du calcul (même modèle/données que la page) ;
    - affichage : timeline des segments (bon/moyen/mauvais en couleurs), tableau des segments, résumé (nombre/longueur par label) ;
    - action « Lancer l’explicabilité sur les segments sélectionnés » (ex. cocher « segments mauvais » → préremplir ou lancer TimeSHAP sur les fenêtres correspondantes).
- **MLflow** : optionnel en v1 ; plus tard, stocker les seuils utilisés et un résumé des segments (nombre, longueur médiane par label) comme paramètres/artefacts de run.

### 2.7 Explicabilité ciblée

- **Entrée** : liste de segments (ou de fenêtres) sélectionnés par l’utilisateur (ex. tous les « mauvais », ou un sous-ensemble).
- **Sortie** : mêmes sorties que l’explicabilité actuelle (TimeSHAP), mais calculées uniquement sur ces fenêtres (ou sur un échantillon si trop nombreuses).
- **Implémentation** : le module d’explicabilité existant (`timeshap_wrapper`, `explainability.py`) reçoit les dates ou les séries découpées correspondant aux segments choisis ; pas de changement majeur de l’API TimeSHAP, seulement le ciblage des instances à expliquer.

### 2.8 Non-fonctionnel

- **Performance** : pour un test long, beaucoup de fenêtres possibles ; le calcul par fenêtre peut être coûteux. Options : stride plus grand par défaut, calcul asynchrone ou en arrière-plan, et/ou échantillonnage pour l’explicabilité si trop de fenêtres.
- **Reproductibilité** : même seed et mêmes données que le reste du pipeline ; pas de tirage aléatoire dans la segmentation.
- **Tests** : au moins un test unitaire sur une petite série synthétique (quelques fenêtres) avec seuils fixes → vérification des labels et du regroupement en segments.

### 2.9 Évolutions possibles (hors v1)

- Seuils adaptatifs (percentiles sur la distribution des métriques du test).
- Règles multi-métriques (vote ou score composite).
- Export des segments vers ModelRadar ou autres outils d’évaluation par aspects.
- Corrélation avec des covariables (ex. segments mauvais pendant des périodes de forte précipitation) pour analyse qualitative.

---

## 3. Références rapides (stack)

| Élément | Fichier / détail |
|--------|-------------------|
| Métriques et seuils indicatifs | `dashboard/config.py` (METRICS, METRICS_INFO), `3_Forecasting.py` (bloc « Comprendre les métriques ») |
| Calcul métriques par fenêtre | `dashboard/utils/training.py` : `calculate_metrics`, `evaluate_model_sliding` |
| Sliding window sur test | `evaluate_model_sliding` : `historical_forecasts(..., start=test.start_time(), forecast_horizon=horizon, stride=stride)` |
| Explicabilité | `dashboard/utils/timeshap_wrapper.py`, `dashboard/utils/explainability.py` |
| Horizon / stride | `model.output_chunk_length`, `stride` passé à `historical_forecasts` |

---

*Document rédigé pour le projet Junon (time-serie-explo). À mettre à jour si la stack ou les seuils évoluent.*
