# Rapport : Pipeline d'entraînement — De la donnée brute aux métriques de qualité

**Projet** : Time-serie-explo  
**Objectif** : Documenter de façon complète et pédagogique le flux d'entraînement des modèles de prévision (séries temporelles), dans tous les cas d'usage.

---

## Sommaire

1. [Introduction et objectifs](#1-introduction-et-objectifs)
2. [Glossaire](#2-glossaire)
3. [Vue d'ensemble du pipeline](#3-vue-densemble-du-pipeline)
4. [Étape 1 — Donnée brute et sélection](#4-étape-1--donnée-brute-et-sélection)
5. [Étape 2 — Préparation par station (DataFrame → TimeSeries)](#5-étape-2--préparation-par-station-dataframe--timeseries)
6. [Étape 3 — Split temporel train / val / test](#6-étape-3--split-temporel-train--val--test)
7. [Étape 4 — Normalisation (fit sur train, transform sur val/test)](#7-étape-4--normalisation-fit-sur-train-transform-sur-valtest)
8. [Étape 5 — Modes Independent vs Global](#8-étape-5--modes-independent-vs-global)
9. [Étape 6 — Optuna (optionnel)](#9-étape-6--optuna-optionnel)
10. [Étape 7 — Entraînement final du modèle](#10-étape-7--entraînement-final-du-modèle)
11. [Étape 8 — Évaluation sur le test](#11-étape-8--évaluation-sur-le-test)
12. [Étape 9 — Métriques de qualité](#12-étape-9--métriques-de-qualité)
13. [Récapitulatif et bonnes pratiques](#13-récapitulatif-et-bonnes-pratiques)
14. [Annexes](#14-annexes)

---

## 1. Introduction et objectifs

### 1.1 Contexte

On entraîne des **modèles de prévision** (forecasting) sur des **séries temporelles** : par exemple niveau de piézomètre, consommation électrique, trafic, etc. La donnée est **ordonnée dans le temps** ; on prédit des valeurs **futures** à partir du **passé**.

### 1.2 Pourquoi un pipeline strict ?

- **Reproductibilité** : mêmes données, mêmes étapes → résultats reproductibles.
- **Pas de fuite** : le **test** n’est jamais utilisé pour entraîner ni pour choisir des hyperparamètres.
- **Standard recherche** : split temporel, normalisation fit sur train uniquement, évaluation sur test uniquement.

### 1.3 Ce que couvre ce rapport

- **Tous les cas** : une seule série (single) ou plusieurs (global), avec ou sans covariables, entraînement manuel ou Optuna.
- **Chaque étape** : ce qu’on fait, pourquoi, et ce qui change selon les cas.
- **De la donnée brute** (fichier CSV, base, etc.) **jusqu’aux métriques** (MAE, RMSE, MAPE, sMAPE, Dir_Acc).

---

## 2. Glossaire

| Terme | Signification |
|-------|---------------|
| **Série temporelle** | Suite de valeurs indexées par le temps (dates ou entiers). |
| **Cible (target)** | Variable à prévoir (ex. niveau piézomètre). |
| **Covariable** | Variable d’accompagnement (ex. température, lags, features temporelles). On n’en fait pas la prédiction ; elles aident le modèle. |
| **Train** | Partie des données utilisée **uniquement** pour apprendre (et pour fit des scalers). |
| **Val (validation)** | Partie utilisée pour **early stopping** et, si Optuna, pour choisir les hyperparamètres. **Jamais** pour les métriques finales. |
| **Test** | Partie utilisée **uniquement** pour calculer les métriques finales. Jamais vue pendant l’entraînement. |
| **Horizon** | Nombre de pas dans le futur que l’on prédit à chaque fois (ex. 7 jours). |
| **Past covariates** | Covariables connues **au moment de la prédiction** (pas de fuite temporelle). |
| **Future covariates** | Covariables futures : non utilisées ici (risque de fuite en production). |
| **Normalisation** | Transformation (ex. MinMax, z-score) pour mettre les données à l’échelle. **Fit sur train**, **transform** sur val/test. |
| **Inverse transform** | Retour à l’échelle originale après prédiction, pour calculer les métriques sur les vraies unités. |

---

## 3. Vue d'ensemble du pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DONNÉE BRUTE (CSV, DB, …)                                                       │
│  Colonnes : date, cible, [station], [covariables]                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SÉLECTION                                                                       │
│  Stations, variable cible, covariables                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PRÉPARATION PAR STATION (DataFrame)                                             │
│  Missing values → Doublons → Datetime features → Lags → Conversion Darts         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SPLIT TEMPOREL                                                                  │
│  train | val | test   (aucun shuffle)                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  NORMALISATION                                                                   │
│  Fit sur train → Transform sur val & test (cible + covariables)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │  INDEPENDENT          │               │  GLOBAL               │
        │  1 modèle / station   │               │  1 modèle / stations  │
        └───────────────────────┘               └───────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OPTUNA (optionnel)                                                              │
│  Recherche d’hyperparamètres sur train/val uniquement                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ENTRAÎNEMENT                                                                    │
│  model.fit(train, val ; past_covariates seulement)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ÉVALUATION SUR TEST                                                             │
│  full_train = train+val → predict(horizon) → inverse_transform → métriques       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MÉTRIQUES                                                                       │
│  MAE, RMSE, MAPE, sMAPE, Dir_Acc (sur échelle originale)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Principe clé** : **train** sert à apprendre, **val** à régulariser/choisir (early stop, Optuna), **test** uniquement à **mesurer** la qualité une seule fois à la fin.

---

## 4. Étape 1 — Donnée brute et sélection

### 4.1 Entrée

- **DataFrame** chargé en page *Dataset Preparation* (`df_raw`).
- Colonnes minimales :
  - **Date** : index ou colonne dédiée.
  - **Cible** : variable à prévoir (`target_var`).
- Optionnel :
  - **Station** : identifiant de série (ex. capteur, site).
  - **Covariables** : variables explicatives (ex. température, pluie).

### 4.2 Sélection (UI)

- **Stations** : une ou plusieurs. En single-station, une seule série.
- **Variable cible** : quelle colonne prévoir.
- **Covariables** : quelles colonnes utiliser en accompagnement.

### 4.3 Multi‑station

En multi-station, le même schéma (date, cible, covariables) est appliqué **par station**. Chaque station est préparée séparément (prétraitement, split, normalisation), puis soit on entraîne **un modèle par station** (Independent), soit **un seul modèle** sur toutes (Global).

---

## 5. Étape 2 — Préparation par station (DataFrame → TimeSeries)

Pour **chaque** station (ou la série unique), on enchaîne les opérations **sur le DataFrame** avant toute modélisation.

### 5.1 Filtrage des colonnes

On ne garde que : `[cible] + covariables` (et la colonne station si multi-station). Le reste est ignoré.

### 5.2 Gestion des valeurs manquantes (`fill_method`)

| Méthode | Action |
|--------|--------|
| **Drop rows** | Supprimer les lignes avec NaN. |
| **Linear Interpolation** | Interpolation linéaire, puis ffill/bfill sur les bords. |
| **Forward fill** | Propagation de la dernière valeur connue. |
| **Backward fill** | Propagation de la valeur suivante connue. |

Toute valeur manquante restante est ensuite supprimée (sécurité).

### 5.3 Doublons de dates

Si une même date apparaît plusieurs fois (ex. plusieurs capteurs à la même date), on **agrège par moyenne** (`groupby(index).mean()`). Une seule ligne par date.

### 5.4 Features temporelles (optionnel)

Si **datetime features** activées, on ajoute des colonnes dérivées de la date :

- `day_of_week`, `month`
- `day_sin`, `day_cos`, `month_sin`, `month_cos` (encodage cyclique)

Elles sont ajoutées comme **covariables**, pas comme cible.

### 5.5 Lags (optionnel)

Ex. `lags = [1, 7, 30]` :

- `cible_lag_1` = valeur de la cible à *t−1*
- `cible_lag_7` = valeur à *t−7*
- etc.

On fait un `shift` puis `dropna()`. Les lags sont **covariables** : information du passé, pas de fuite.

### 5.6 Conversion Darts

- **Cible** : `TimeSeries.from_dataframe(..., value_cols=cible, fill_missing_dates=True)`.
- **Covariables** : même DataFrame, `value_cols = covariables` (y compris datetime + lags si activés).
- **Fréquence** : inférée (`pd.infer_freq`) ou `'D'` par défaut.

**Résultat** : par station, une `target_series` et éventuellement une `covariates_series`.

---

## 6. Étape 3 — Split temporel train / val / test

### 6.1 Principe

On découpe **dans l’ordre du temps** : pas de mélange, pas de shuffle.

```
 temps ─────────────────────────────────────────────────────────────────────────►

 [====== train ======][== val ==][===== test ======]
 |                    |         |
 0              train_end   val_end                 fin
```

- **train** : `[0, train_end)`
- **val** : `[train_end, val_end)`
- **test** : `[val_end, fin]`

Les limites sont déduites des **ratios** (ex. 70 % / 15 % / 15 %).

### 6.2 Application

- **Cible** : `split_train_val_test(target_series, train_ratio, val_ratio, test_ratio)`  
  → `train_raw`, `val_raw`, `test_raw`.
- **Covariables** : même split avec les **mêmes** ratios → `train_cov_raw`, `val_cov_raw`, `test_cov_raw`.

Les tranches sont **contiguës** et **alignées** dans le temps entre cible et covariables.

### 6.3 Contrôles

- Les ratios doivent sommer à 1.
- Aucune tranche ne doit être **vide** (sinon erreur explicite). Il faut suffisamment de données et des ratios raisonnables.

---

## 7. Étape 4 — Normalisation (fit sur train, transform sur val/test)

### 7.1 Pourquoi normaliser ?

- Beaucoup de modèles (ex. réseaux de neurones) sont sensibles à l’échelle des entrées.
- Réduire ou standardiser améliore la stabilité et souvent les performances.

### 7.2 Règle anti‑fuite

- **Fit** : uniquement sur **train** (cible et covariables).
- **Transform** : appliqué à **val** et **test** avec les paramètres **appris sur train**.

Ainsi, val et test n’influencent jamais les paramètres des scalers.

### 7.3 Chaîne de préprocessing (cible et covariables)

La config peut inclure :

1. **Transformation** (avant scaling) : Log, Box‑Cox, différenciation.
2. **Normalisation** : MinMax, StandardScaler (z‑score), RobustScaler (médiane + IQR).

On utilise un `TimeSeriesPreprocessor` (Darts) par type de série (cible vs covariables), avec la même config.

### 7.4 Opérations effectuées

- **Cible** :
  - `fit_transform(train_raw)` → `train`
  - `transform(val_raw)` → `val`, `transform(test_raw)` → `test`
- **Covariables** (si présentes) :
  - `fit_transform(train_cov_raw)` → `train_cov`
  - `transform(val_cov_raw)` → `val_cov`, `transform(test_cov_raw)` → `test_cov`
  - `transform(covariates_series)` sur la **série complète** (train+val+test) → `covariates_scaled` = **full_cov**

`full_cov` servira plus tard pour la **prédiction** (past covariates sur toute la période).

### 7.5 Vérification

Si des NaN apparaissent dans `train` ou `val` après préprocessing, le pipeline s’arrête avec une erreur. Il faut corriger la stratégie de gestion des manquants.

---

## 8. Étape 5 — Modes Independent vs Global

### 8.1 Mode Independent (un modèle par station)

- On traite **une station à la fois** (ou une seule série).
- Pour chaque station : `train`, `val`, `test`, et éventuellement `train_cov`, `val_cov`, `test_cov`, `full_cov`.
- **Un modèle** est entraîné **par** station (ou un seul modèle si une seule station).

**Cas typique** : séries très différentes (sites différents), pas de partage d’information entre stations.

### 8.2 Mode Global (un modèle pour toutes les stations)

- **Plusieurs** stations sont préparées (mêmes étapes 2–4).
- On accumule tout dans des **listes** :
  - `global_data['train']`, `['val']`, `['test']` : une `TimeSeries` par station.
  - Idem pour les covariables : `train_cov`, `val_cov`, `test_cov`, `full_cov`.
- **Scalers** : un `target_preprocessor` et un `cov_preprocessor` **par station**, conservés (ex. dans `global_metadata`).
- **Un seul modèle** est entraîné sur **toutes** les séries (listes).

**Cas typique** : séries similaires (même type de capteur, même processus), bénéfice à partager les paramètres.

### 8.3 Synthèse

| | Independent | Global |
|--|-------------|--------|
| **Données** | 1 station à la fois | Liste de séries (une par station) |
| **Modèles** | 1 par station | 1 commun |
| **Scalers** | 1 set par station | 1 set par station (inverse transform par série) |

---

## 9. Étape 6 — Optuna (optionnel)

### 9.1 Rôle

**Optuna** cherche des **hyperparamètres** (learning rate, dropout, etc.) en minimisant une **métrique sur la validation**.  
`input_chunk_length` et `output_chunk_length` restent **fixés** par l’interface (non optimisés par Optuna).

### 9.2 Données utilisées

- **Uniquement** **train** et **val**.
- Le **test** n’est **jamais** utilisé dans Optuna.

### 9.3 Déroulé type

1. Pour chaque **trial** :
   - Créer un modèle avec des hyperparamètres proposés par Optuna.
   - `model.fit(train, val, train_cov, val_cov)` (past covariates seulement).
   - Prédire sur **val** (à partir de la fin de `train`).
   - Calculer une métrique (ex. MAE) sur ces prédictions.
2. Optuna minimise cette métrique (ex. MAE sur val).
3. Les **meilleurs** hyperparamètres sont retenus pour l’**entraînement final**.

### 9.4 Entraînement final après Optuna

On ré‑entraîne **un** modèle avec ces meilleurs hyperparamètres, toujours sur **train** (et **val** pour early stopping). L’évaluation finale se fait uniquement sur **test**.

---

## 10. Étape 7 — Entraînement final du modèle

### 10.1 Données d’entraînement

- **Cible** : `train` (+ `val` pour early stopping uniquement).
- **Covariables** : **past covariates** seulement (`train_cov`, `val_cov`), si le modèle les supporte.

**Aucune** future covariable, **aucune** donnée de test.

### 10.2 Appel modèle

```text
model.fit(
  series=train,
  val_series=val,
  past_covariates=train_cov,      # si utilisé
  val_past_covariates=val_cov     # si utilisé
)
```

### 10.3 Callbacks

- **Métriques** (loss, etc.) écrites dans un fichier JSON pour le suivi Streamlit.
- **Early stopping** sur `val_loss` si activé : arrêt si plus d’amélioration sur la validation.

### 10.4 Reproductibilité

- `pl.seed_everything(RANDOM_SEED)` est appelé au début du pipeline d’entraînement.
- Même graine → mêmes initialisations, même ordre de batch → résultats reproductibles (à environnement constant).

---

## 11. Étape 8 — Évaluation sur le test

### 11.1 Objectif

Estimer la **qualité** du modèle sur une période **jamais vue** pendant l’entraînement. Une seule évaluation sur **test**.

### 11.2 Historique pour la prédiction

À la fin de l’entraînement, on **simule** qu’on est à la fin de la **validation** et qu’on a observé **train + val**.

- **Independent** :  
  `full_train = concatenate([train, val], axis=0)`  
  (nouvelle série, `train` et `val` ne sont pas modifiés).
- **Global** :  
  `full_train = [concatenate([t, v], axis=0) for t, v in zip(train, val)]`  
  (une série concaténée par station).

Le modèle prédit ensuite la **suite** : le **test**.

### 11.3 Horizon de prédiction

- **output_chunk_length** : nombre de pas que le modèle produit en une fois (ex. 7 jours).
- **Longueur du test** :
  - Single : `len(test)`.
  - Global : `min(len(ts) for ts in test)`.
- **Horizon utilisé** :  
  `horizon = min(output_chunk_length, longueur_du_test)`  

On ne prédit ni au‑delà du test, ni plus que ce que le modèle peut sortir.

### 11.4 Prédiction

- **Entrée** :
  - `series = full_train` (historique),
  - `n = horizon`,
  - `past_covariates = full_cov` (série complète des covariables) si covariables utilisées et modèle compatible.
- **Sortie** : **horizon** pas de prédiction, alignés sur le **début** de la période test.

### 11.5 Retour à l’échelle originale (inverse transform)

- Les **prédictions** sont repassées à l’échelle brute via  
  `target_preprocessor.inverse_transform(predictions)`  
  (par station en global, avec les scalers sauvegardés).
- Le **test** est aussi inverse‑transformé pour le calcul des métriques.  
On compare donc **même échelle** que la donnée brute.

### 11.6 Alignement temporel

Avant de calculer les métriques :

- `actual_aligned = actual.slice_intersect(predicted)`
- `predicted_aligned = predicted.slice_intersect(actual)`

On mesure **sur la même plage de temps** (souvent la période de prévision). Les métriques sont calculées sur ces séries alignées.

### 11.7 Standard d’évaluation : une fenêtre vs fenêtrage (sliding window)

**Pratique standard en séries temporelles (y compris multivariées) :**

| Split | Rôle | Fenêtrage ? | Commentaire |
|-------|------|-------------|-------------|
| **Train** | Apprentissage | Non | On utilise la **loss** d’entraînement. Pas d’évaluation métriques type MAE/RMSE sur train ; fenêtrer n’apporte rien de standard. |
| **Val** | Early stopping, Optuna | **En général non** | Une **seule** prédiction (un bloc à la fin de train) par epoch pour la loss de validation. Fenêtrer sur val à chaque epoch serait très coûteux ; le standard est **single-block** pour la val. |
| **Test** | Métriques finales | **Oui, standard** | **Backtesting / sliding window** : plusieurs fenêtres sur le test, métriques agrégées (ex. MAE, RMSE moyennes sur les fenêtres). Plus robuste qu’une seule fenêtre, reflète mieux l’usage en production (prévisions répétées). |

**Pourquoi fenêtrer sur le test ?**

- Une **seule** fenêtre (un bloc au début du test) donne une métrique très dépendante de cette période (variance élevée).
- Le **fenêtrage** (ex. `historical_forecasts` avec `stride = horizon`) produit plusieurs fenêtres **non chevauchantes** sur le test ; on agrège les métriques (moyenne MAE, RMSE, etc.). C’est la pratique recommandée dans la littérature (benchmarks M4, backtesting, etc.).

**Ce qu’on fait dans le pipeline :**

- **Val** : single-block pour early stopping (comme ci‑dessus).
- **Test** : on calcule à la fois :
  - **Test (1 fenêtre)** : une prédiction au début du test (comportement historique).
  - **Test (fenêtré)** : sliding window sur tout le test, métriques agrégées → **référence principale** pour la qualité.

Les métriques sauvegardées et affichées incluent donc les deux quand le fenêtrage est disponible (série unique ; modèle global peut rester en single-block selon implémentation).

---

## 12. Étape 9 — Métriques de qualité

Toutes les métriques sont calculées **après inverse transform** (échelle originale) et **uniquement sur le test** (soit 1 fenêtre, soit fenêtré selon le mode).

### 12.1 MAE (Mean Absolute Error)

- **Définition** : moyenne des |prédit − réel|.
- **Unité** : même que la cible (ex. m, °C).
- **Interprétation** : erreur absolue moyenne ; pénalise toutes les erreurs de façon égale.

### 12.2 RMSE (Root Mean Squared Error)

- **Définition** : racine de la moyenne des (prédit − réel)².
- **Unité** : même que la cible.
- **Interprétation** : pénalise davantage les **grosses** erreurs que les petites.

### 12.3 MAPE (Mean Absolute Percentage Error)

- **Définition** : moyenne des |prédit − réel| / |réel|, souvent en %.
- **Attention** : instable si **réel proche de 0** (division par de très petits nombres).

### 12.4 sMAPE (Symmetric MAPE)

- **Définition** : dénominateur = |prédit| + |réel| (évite les divisions par 0).
- Souvent **plus robuste** que le MAPE classique.

### 12.5 Dir_Acc (Direction Accuracy)

- **Définition** : pour des paires de pas consécutifs (t, t+1), proportion où la **direction** (hausse / baisse) est correcte.
- **Unité** : %.
- **Intérêt** : importe surtout si la **tendance** (monter/descendre) compte plus que l’erreur absolue.

### 12.6 Récapitulatif

| Métrique | Formule (idée) | Usage |
|----------|----------------|--------|
| MAE | mean(\|y − ŷ\|) | Erreur moyenne simple |
| RMSE | sqrt(mean((y − ŷ)²)) | Pénaliser les grandes erreurs |
| MAPE | mean(\|y − ŷ\| / \|y\|) | Erreur relative (attention si y ≈ 0) |
| sMAPE | variant symétrique | Erreur relative plus stable |
| Dir_Acc | % de bonnes directions | Qualité de la tendance |

En **global**, les métriques sont typiquement agrégées par série (ex. moyenne sur les stations), selon l’implémentation.

---

## 13. Récapitulatif et bonnes pratiques

### 13.1 Flux linéaire résumé

```text
df_raw
  → sélection (stations, cible, covariables)
  → préparation DataFrame (missing, doublons, datetime, lags)
  → conversion Darts (cible + cov)
  → split temporel → train_raw, val_raw, test_raw (idem cov)
  → normalisation : fit sur train → train ; transform → val, test (idem cov)
  → [Independent : 1 modèle/station | Global : listes, 1 modèle]
  → [Optuna sur train/val si activé]
  → model.fit(train, val ; past_cov only)
  → full_train = concat(train, val)
  → horizon = min(output_chunk_length, len_test)
  → predict(full_train, n=horizon, past_cov=full_cov)
  → inverse_transform(prédictions + test)
  → calculate_metrics(actual, predicted) → MAE, RMSE, MAPE, sMAPE, Dir_Acc
```

### 13.2 Rôles de train / val / test

| Ensemble | Rôle | Utilisé pour |
|----------|------|----------------|
| **Train** | Apprentissage | Fit des scalers, fit du modèle |
| **Val** | Régularisation / choix | Early stopping, Optuna |
| **Test** | Évaluation finale | Métriques **une seule fois** |

**Test** n’est **jamais** utilisé pour : fit, early stopping, ou recherche d’hyperparamètres.

### 13.3 Bonnes pratiques respectées

- **Split temporel** : pas de shuffle, ordre chronologique conservé.
- **Normalisation** : fit **uniquement** sur train.
- **Covariables** : **past** seulement ; pas de future.
- **Évaluation** : **uniquement** sur test, **après** inverse transform.
- **Repro** : graine fixée ; `concatenate` pour éviter de modifier train/val.
- **Guards** : erreurs claires si tranches vides ou horizon invalide.

### 13.4 Tableau des cas d’usage

| Cas | Données | Entraînement | Évaluation |
|-----|---------|--------------|------------|
| Single, sans cov | train/val/test | 1 modèle, `fit(train, val)` | full_train, predict(horizon), inverse transform, métriques |
| Single, avec cov | + train_cov, val_cov, test_cov, full_cov | + past_covariates | idem + past_covariates=full_cov |
| Global, sans cov | Listes train/val/test | 1 modèle, `fit(listes)` | full_train list, prédictions par série, scalers par station |
| Global, avec cov | + listes de cov | + past_covariates (listes) | idem + full_cov (listes) |

---

## 14. Annexes

### A. Fichiers principaux du pipeline

- **Préparation / UI** : `dashboard/training/pages/1_Dataset_Preparation.py`, `2_Train_Models.py`
- **Prétraitement** : `dashboard/utils/preprocessing.py` (split, `TimeSeriesPreprocessor`)
- **Entraînement / évaluation** : `dashboard/utils/training.py` (`run_training_pipeline`, `train_model`, `evaluate_model`, `calculate_metrics`)
- **Optuna** : `dashboard/utils/optuna_training.py`
- **Modèles** : `dashboard/utils/model_factory.py`, `core/callbacks.py`

### B. Préconditions et garde-fous

- Ratios train/val/test > 0 et somme = 1.
- Assez de données pour que **aucune** tranche ne soit vide.
- `output_chunk_length` ≥ 1.
- Test non vide pour avoir un horizon ≥ 1.

Si ces conditions ne sont pas remplies, des **ValueError** explicites sont levées.

### C. Dépendances techniques

- **Darts** : séries temporelles, modèles, métriques.
- **PyTorch Lightning** : entraînement des modèles deep learning.
- **Optuna** : recherche d’hyperparamètres (si mode Optuna).
- **scikit-learn** : scalers (StandardScaler, MinMaxScaler, RobustScaler) via Darts.

---

*Document généré pour le projet Time-serie-explo. Pipeline aligné avec le code des modules `dashboard.utils.training`, `dashboard.utils.preprocessing` et `dashboard.training.pages.2_Train_Models`.*
