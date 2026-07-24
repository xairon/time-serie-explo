# Climat — SPEI (Standardized Precipitation-Evapotranspiration Index)

**Date** : 2026-07-23
**Statut** : design validé, prêt pour plan d'implémentation
**Repos** : `hubeau_data_integration` (warehouse, gros du travail) + `time-serie-explo` (API + frontend, expo)
**Suite de** : `2026-06-30-era5-sti-standardized-temperature.md` (le SPEI y était listé « out of scope, separate »).

## 1. Contexte & décision

Le module `/climat` expose déjà deux indices de grille standardisés, précalculés dans le
warehouse : **SPI** (déficit pluviométrique, gamma) et **STI** (anomalie thermique, z-score).
Le SPEI est le troisième membre naturel de la triade sécheresse :

- **SPI** — déficit de pluie.
- **SPEI** — déficit de pluie **corrigé de la demande évaporative** (`D = P − ETP`).
- **STI** — chaleur relative.

Le SPEI colle exactement à la **doctrine projet** énoncée dans `2026-07-16-climat-etp-echelle-temperature-design.md` :

> Soit un vrai indicateur (IPS, SPLI, SPI, STI…), soit une vraie valeur. Jamais un intermédiaire inventé.

Le SPEI est un standard WMO publié (Vicente-Serrano et al., 2010), avec seuils de classes
publiés. Et ses ingrédients existent déjà **par cellule** : `fct_era5_monthly_grid.bilan_hydrique`
(`= P − ETP`, mm/mois).

### 1.1 Pas de blocage température — correction d'une fausse piste

Une première analyse a supposé que le SPEI était bloqué par le biais froid de la
température (lecture instantanée 00:00 UTC). **C'est faux, et la spec le documente pour
éviter que quelqu'un re-bute dessus :**

- Le biais 00:00 UTC ne touchait **que la température** (grandeur diurne échantillonnée à un
  instant). Il a par ailleurs été **corrigé** (cutover grille 2026-07-13, station 2026-07-15 —
  `fct_era5_monthly_grid.temperature_*` vient désormais de `stg_era5_daily_temp_stats`).
- L'**ETP** dérive de `potential_evaporation` d'ERA5, un **flux d'accumulation** produit par
  le modèle ECMWF. La valeur à 00:00 UTC **est** le cumul journalier correct — pas un
  échantillon biaisé. Confirmé dans `hubeau_data_integration/docs/ERA5.md` (« précip/ETP =
  flux d'accumulation journaliers, chaque variable a sa représentation journalière correcte »).

Conclusion : `bilan_hydrique` mensuel est **déjà correct**. Le SPEI est **buildable
immédiatement, mono-lot**, sans prérequis warehouse.

### 1.2 Caveat de données (à afficher, pas un blocage)

`potential_evaporation` d'ERA5-Land est une estimation d'ET potentielle **grossière** — ce
n'est **pas** un Penman-Monteith FAO-56. Le SPEI hérite de cette approximation. C'est
acceptable et cohérent : c'est déjà la donnée que la plateforme affiche comme « ETP » et
« bilan hydrique ». À documenter dans le hover ⓘ du SPEI ; améliorer l'ETP (Penman-Monteith)
est un chantier data séparé, **hors périmètre**.

## 2. Méthode

Pour une cellule 0.1°, une fenêtre `N ∈ {1,3,6,12}` mois finissant au mois `M` :

- **Observé** : `D_N = Σ bilan_hydrique` sur la fenêtre de N mois finissant à `M`
  (cumul du bilan hydrique, mm — miroir exact de `precip_cumul` du SPI).
- **Référence** : pour chaque année `y ∈ 1991-2020`, le `D_N` de la fenêtre finissant au mois
  calendaire de `M` (fenêtres à cheval sur l'année : convention d'**année de fin**, comme le
  SPI/STI). → un échantillon de ~30 valeurs `D_N` par (cellule, mois calendaire, fenêtre).
- **Ajustement** : **log-logistique à 3 paramètres** (loi de Fisk translatée), fittée par
  **L-moments** (méthode canonique de Vicente-Serrano 2010 — `D` est signé et asymétrique,
  ni la gamma du SPI ni le z-score du STI ne conviennent).
- **SPEI** = `Φ⁻¹(F(D_N))`, borné.
- **Classification** : les **7 classes McKee/WMO** et seuils z du projet (`_THRESHOLDS_7` de
  `dashboard/utils/drought.py` / `classify_index` de `api/era5_anomaly.py`) : ±0.84, ±1.28,
  ±1.75. Sémantique **sécheresse** (comme le SPI) : négatif = sec, positif = humide.

### 2.0 ⚠️ AMENDEMENT DU 2026-07-24 — la loi retenue est finalement la **logistique généralisée (GLO)**

La §2.1 ci-dessous décrit le choix **initial** (log-logistique 3 paramètres). Il a été
**remplacé après mise en production**, sur constat mesuré. On garde le texte d'origine pour
la traçabilité de la décision.

**Ce qu'on a observé** : la log-logistique n'ajustait que **74,6 %** des couples
cellule × mois × fenêtre. Instrumentation des motifs de rejet (ajoutée pour l'occasion) :
**100 % des rejets** venaient de la garde `β ≤ 1`, et **zéro** d'un manque de données
(`n_annees_insuffisant = 0` partout).

**La cause, démontrée** : avec la convention PWM utilisée, `β = 1/τ₃` (τ₃ = L-asymétrie). Or
|τ₃| < 1 pour toute distribution réelle, donc `β ≤ 1` signifie nécessairement **β < 0, donc
τ₃ < 0**. Vérifié empiriquement, séparation parfaite :

| | n | τ₃ min | médiane | max | % τ₃ < 0 |
|---|---|---|---|---|---|
| acceptés | 100 578 | 0,000 | 0,086 | 0,379 | **0,0 %** |
| rejetés | 37 374 | −0,308 | −0,050 | −0,000 | **100,0 %** |

La log-logistique est une loi à **asymétrie positive** : elle ne peut structurellement pas
représenter les mailles dont le bilan hydrique est à asymétrie négative. Ce n'était donc ni
un bug ni un défaut de données, mais une **loi trop étroite**.

**Le correctif** : la **logistique généralisée (GLO)**, dont le paramètre de forme
`k = −τ₃` accepte les deux signes. Ce n'est pas un changement de famille opportuniste :
c'est la loi qu'utilise l'implémentation de référence (paquet R `SPEI` → `parglo`).
Estimateurs de Hosking :

```
k = −τ₃
α = λ₂ / (Γ(1+k)·Γ(1−k))
ξ = λ₁ − α·(1/k − π/sin(kπ))
F(x) = 1 / (1 + (1 − k(x−ξ)/α)^(1/k))        (cas limite k≈0 : logistique)
```

**Résultats mesurés après bascule** :
- couverture de l'ajustement : **74,6 % → 100,0 %** sur les 4 fenêtres, **zéro rejet** de
  quelque motif que ce soit ;
- couverture du `spei` en carte (juin 2026) : **75 % → 99,2 %** (le reliquat = valeurs du
  mois hors du support ajusté, comportement légitime) ;
- **non-régression prouvée** : sur les 35 614 mailles que l'ancienne loi ajustait déjà,
  écart médian **0,000**, p95 **0,000**, max **0,000**, corrélation **1,0000**. La GLO redonne
  exactement les mêmes valeurs là où la log-logistique fonctionnait — les deux coïncident sur
  le domaine τ₃ > 0. Le changement est donc **purement additif**.

Colonnes de la table de référence : `ll_alpha/ll_beta/ll_gamma` → `glo_alpha/glo_k/glo_xi`
(les anciennes colonnes sont conservées, non supprimées, pour ne rien détruire en prod).

### 2.1 Fit log-logistique par L-moments (implémentation maison) — *choix initial, remplacé (cf. §2.0)*

**Décision** : implémentation **maison** (numpy/scipy), **aucune dépendance externe** —
cohérent avec `drought.py` qui refuse explicitement la dépendance `spei`. `climate-indices`
a été écarté car il ne propose que gamma/Pearson III, pas la log-logistique ; les autres
packages (xclim, standard-precip) ont été écartés pour ne pas alourdir le warehouse.

PWM non biaisés sur l'échantillon `x = D_N` (n valeurs, triées croissant, indices `i = 1..n`) :

```
w0 = mean(x)
w1 = (1/n) Σ [(n−i)/(n−1)]           · x_i
w2 = (1/n) Σ [(n−i)(n−i−1)/((n−1)(n−2))] · x_i
```

Paramètres log-logistiques (β = forme, α = échelle, γ = origine) :

```
β = (2·w1 − w0) / (6·w1 − w0 − 6·w2)
α = (w0 − 2·w1)·β / (Γ(1+1/β)·Γ(1−1/β))
γ = w0 − α·Γ(1+1/β)·Γ(1−1/β)
```

CDF : `F(x) = [1 + (α/(x − γ))^β]^(−1)`, puis `SPEI = Φ⁻¹(clip(F, 0.001, 0.999))`.

`Γ` via `scipy.special.gamma`.

**Cas dégénérés** → SPEI `NaN` (jamais une valeur fausse) :
- `β ≤ 0` ou non fini (échantillon quasi constant, L-moments incohérents).
- `1/β ≥ 1` (⇒ `Γ(1−1/β)` diverge) → fit rejeté.
- `x ≤ γ` pour l'observé (hors support de la loi).
- `nb_annees_ref < 25` (seuil WMO `MIN_YEARS_REF`, déjà en place).

Tests unitaires : valeurs de référence calculées à la main / croisées avec le package R
`SPEI` sur un échantillon fixe (fixture), + chaque cas dégénéré → NaN.

## 3. Warehouse (`hubeau_data_integration`)

### 3.1 `src/hubeau_pipeline/ml/era5_indices.py`

Ajouter deux fonctions pures (miroir de `compute_spi`/`compute_sti`), vectorisées où
possible :

- `fit_loglogistic_lmoments(samples: np.ndarray) -> tuple[α, β, γ] | (nan, nan, nan)` —
  fit sur un échantillon 1D (les ~30 `D_N` de référence d'une cellule×mois×fenêtre).
- `compute_spei(d_cumul, ll_alpha, ll_beta, ll_gamma) -> np.ndarray` — applique la CDF
  log-logistique puis `norm.ppf`, NaN sur params invalides ou `x ≤ γ`.

### 3.2 Table de référence — `gold.fct_era5_spei_climatology_grid` (Python-managée)

**Pourquoi une table à part et non le mart dbt `fct_era5_climatology_grid`** : le fit
L-moments a besoin (a) des ~30 échantillons annuels `D_N` **et** (b) de la fonction Γ. Aucun
des deux n'est faisable proprement en SQL dbt (PostgreSQL n'a pas `gamma()`). La table est
donc créée et peuplée par du Python, à côté des marts dbt.

Schéma (miroir de `fct_era5_indices_grid`, params au lieu de valeurs) :

```sql
CREATE TABLE IF NOT EXISTS gold.fct_era5_spei_climatology_grid (
    era5_latitude   numeric(6,3) NOT NULL,
    era5_longitude  numeric(6,3) NOT NULL,
    mois_calendaire smallint     NOT NULL,   -- 1..12
    fenetre         smallint     NOT NULL,   -- 1/3/6/12
    ll_alpha        double precision,
    ll_beta         double precision,
    ll_gamma        double precision,
    nb_annees       smallint,
    computed_at     timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (era5_latitude, era5_longitude, mois_calendaire, fenetre)
);
```

Peuplée par un **asset Python de référence** (rebuild rare, pas nightly) qui, pour chaque
fenêtre :

1. Interroge `fct_era5_monthly_grid` : cumuls glissants `SUM(bilan_hydrique) OVER (…​ ROWS
   BETWEEN N-1 PRECEDING AND CURRENT ROW)` sur **1990-01 → 2020-12** (warmup 11 mois pour la
   fenêtre 12), filtrés `mois_complet` (complétude **précipitation** — l'ETP suit la précip,
   pas la température ; on n'utilise **pas** `temp_complet` ici, contrairement au STI), et
   `n_mois = N`.
2. Assigne chaque fenêtre à son mois calendaire de fin, groupe par cellule×mois×fenêtre → un
   `np.ndarray` d'échantillons annuels.
3. `fit_loglogistic_lmoments` par groupe, upsert des `(α, β, γ, nb_annees)`.

### 3.3 `src/hubeau_pipeline/assets/era5_indices_assets.py`

L'asset `fct_era5_indices_grid` calcule aussi le SPEI (nightly 3 mois + bootstrap) :

- `_QUERY` : ajouter `SUM(bilan_hydrique) OVER w AS bilan_cumul` (à côté de `precip_cumul`) ;
  jointure supplémentaire sur `gold.fct_era5_spei_climatology_grid` (mêmes clés que la
  jointure climato) pour ramener `ll_alpha/beta/gamma`.
- `_compute_range` : `spei = compute_spei(df["bilan_cumul"], df["ll_alpha"], …)` ; appliquer
  le même masque WMO `nb_annees < MIN_YEARS_REF → NaN` ; ajouter `spei` aux tuples upsertés.
- `deps` : ajouter la table de référence SPEI.

### 3.4 `src/hubeau_pipeline/ml/era5_indices_persistence.py`

- `_CREATE` : `+ spei double precision`.
- Migration idempotente : `ALTER TABLE gold.fct_era5_indices_grid ADD COLUMN IF NOT EXISTS
  spei double precision;` (la table existe déjà en prod → pas de recreate).
- `_UPSERT` / `_TEMPLATE` : ajouter la colonne `spei`.
- `upsert_era5_indices` : docstring + arité des tuples `(lat, lon, month, fenetre, spi, sti, spei)`.

### 3.5 Rebuild

1. Créer + peupler `gold.fct_era5_spei_climatology_grid` (asset référence, full).
2. `ALTER TABLE … ADD COLUMN spei` (idempotent).
3. Ré-exécuter `fct_era5_indices_grid` en bootstrap (ou backfill ciblé) pour peupler `spei`
   sur l'historique.
4. Ré-étiquetage / purge caches junon (cf. §4.4).

## 4. Exposition — API (`time-serie-explo/api/routers/observatory_climat.py`)

### 4.1 Séries de point — `_merge_point_series`

Ajouter `spei_{fen}` à côté de `spi_{fen}`/`sti_{fen}` :

```python
indices_by_month[key][f"spei_{fen}"] = _num(r["spei"])
...
entry[f"spei_{fen}"] = idx.get(f"spei_{fen}")
```

Le `SELECT` de la requête indices doit inclure `spei`.

### 4.2 Couche carte — endpoint grille-indices

L'endpoint renvoie déjà `index_class` générique calculé via `classify_index`. Ajouter `spei`
comme variable d'indice sélectionnable (`?variable=spei`), classifiée par le **même**
`classify_index` (seuils partagés). Pas de nouvelle route.

### 4.3 Épisodes SPEI — `_build_drought_episodes`

**Décision : on ajoute les épisodes SPEI** (périmètre validé « + épisodes »). La logique
gaps-and-islands existante (mois calendaires consécutifs `SPI < -1`) est **paramétrée par
l'indice** plutôt que dupliquée :

- Généraliser `_build_drought_episodes(index_rows, monthly_rows, clim_rows, index_key)` pour
  accepter `index_key ∈ {"spi", "spei"}` (défaut `"spi"` — rétrocompat).
- L'endpoint `point-episodes` accepte `?index=spei` (défaut `spi`). Deux séries d'épisodes
  distinctes, jamais fusionnées.
- Sémantique documentée : un épisode SPEI capture une sécheresse **incluant la demande
  évaporative** — typiquement plus long / plus sévère qu'un épisode SPI en été.

### 4.4 Caches

Les clés Redis `junon:obs_climat_*` (TTL 24h) couvrent grid-indices / point-series /
point-episodes. Purge au déploiement (pattern `junon:obs_climat_*`), comme pour STI.

## 5. Exposition — Frontend (`time-serie-explo/frontend/src`)

### 5.1 `lib/climat-colors.ts`

- `ClimatVariable` : `+ 'spei'`, `kind: 'index'`, `unit: 'σ'`, `labelKey:
  'climat.variables.spei'`, `stops: []`.
- **Palette : réutilise `SPI_CLASS_COLORS`/`SPI_CLASS_ORDER`** — le SPEI est un indice de
  **sécheresse** comme le SPI (négatif = déficit → rouge/brun, positif = surplus → bleu),
  **pas** la palette thermique du STI.
- `climatIndexColorExpression(variable: 'spi' | 'sti' | 'spei')` : accepter `'spei'`, mapper
  vers `SPI_CLASS_ORDER`/`SPI_CLASS_COLORS`.
- `climatFormatValue` : brancher `'spei'` sur la branche signée `±x.x σ` (comme `spi`/`sti`).

### 5.2 Picker / légende / popup / PointPanel

- `VariablePicker` : nouvel item SPEI dans le groupe **Anomalie** (à côté de SPI/STI).
- Légende, popup carte, `PointPanel` (série multi-fenêtres 1/3/6/12) : patron STI à
  l'identique. Hover ⓘ décrivant le SPEI + le caveat ETP (§1.2).

### 5.3 i18n (`fr.json` / `en.json`)

- `climat.variables.spei` = « SPEI (précip. − ETP) » / « SPEI (precip. − PET) ».
- Texte du hover ⓘ (définition + caveat PEV ERA5). Réutiliser les libellés de classes McKee
  existants (`CLASSIFICATION_LABELS`), sémantique sécheresse (très sec … très humide).

### 5.4 Tests (vitest)

- `climat-colors.test.ts` : `spei` dans le modèle couleur (7 classes SPI), `spei` → palette
  SPI (pas STI), format `±x.x σ`.
- `VariablePicker.test.tsx` : SPEI rendu dans le groupe Anomalie.
- `PointPanel` : série SPEI multi-fenêtres, `null` géré.
- **`npm run build` obligatoire** (tsc -b attrape les trous du type union exhaustif — leçon
  2026-07-16).

## 6. Tests (warehouse)

- pytest `test_era5_indices.py` : `fit_loglogistic_lmoments` contre une fixture de référence
  (échantillon fixe, params attendus au 1e-3) ; `compute_spei` (CDF → z, cas dégénérés → NaN,
  masque WMO). Purs, sans DB.
- Contrôle post-bootstrap (curl / SQL) : `fct_era5_indices_grid.spei` non-NULL sur ~11 500
  cellules pour un mois récent, distribution de z plausible (pas tout à 0), cohérence de
  signe avec `bilan_hydrique` (bilan très négatif ⇒ SPEI négatif).

## 7. Hors périmètre

- **SPEI niveau station** : les stations piézo/hydro n'ont pas d'ETP → grille/climat only.
- **Amélioration de l'ETP** (Penman-Monteith FAO-56 vs PEV ERA5-Land) : chantier data séparé.
- **SPEI empirique (KDE)** : la log-logistique canonique est retenue ; un repli KDE façon
  SPLI n'est pas implémenté (cas dégénérés → NaN, pas de repli distributionnel).
- **precip STI / autres indices** : non concernés.

## 8. Risques

- **Fit log-logistique instable sur mailles sèches** (bilan hydrique quasi toujours négatif,
  faible variance) → cas dégénérés fréquents en climat méditerranéen. Mitigation : masque NaN
  strict (§2.1) + contrôle post-bootstrap de la couverture NaN par région (loguer le taux de
  NaN, comme les autres jobs indices n'imposent pas de cap silencieux).
- **Coût du bootstrap SPEI** : le fit L-moments par cellule×mois×fenêtre (~11 500 × 12 × 4 ≈
  550 k fits) s'exécute une fois (référence) puis l'application nightly est cheap (CDF
  vectorisée). Vérifier le temps de l'asset référence ; vectoriser le fit par groupe si besoin.
- **Rétrocompat épisodes** : `_build_drought_episodes` gagne un paramètre `index_key` avec
  défaut `"spi"` → aucun appelant existant cassé.
- **Périmètre confiné côté front** : le type union exhaustif `ClimatVariable` fait échouer le
  build sur tout branchement oublié — filet de sécurité identique à l'ajout STI.
