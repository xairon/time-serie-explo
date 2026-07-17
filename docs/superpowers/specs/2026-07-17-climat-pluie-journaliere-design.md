# Climat — la pluie journalière en couche carte

**Date** : 2026-07-17
**Statut** : design validé, prêt pour plan d'implémentation
**Repo** : `time-serie-explo` (frontend + API ; **aucune modification entrepôt**)
**Prolonge** : `2026-07-16-climat-etp-echelle-temperature-design.md` (doctrine carte = indicateur / nombre = valeur)

## 1. Contexte & problème

Retour utilisateur : « pourquoi on n'a pas la température et les précipitations journalières et hebdos dans le choix des fenêtres ? on récupère la donnée de manière journalière pourtant il me semble ».

Deux choses distinctes s'y cachent, et une seule est un défaut.

### 1.1 Les fenêtres ne sont pas un pas de temps (pas de changement)

`showWindow = isClimatIndexVariable(variable)` (`VariablePicker.tsx`) : le sélecteur 1/3/6/12 n'apparaît **que** pour SPI/STI. Ce sont les fenêtres d'**accumulation de l'indice** — SPI-3 = « pluie cumulée sur 3 mois, standardisée contre la climatologie 1991-2020 du même cumul » — pas un pas d'affichage. Les valeurs brutes n'en ont jamais eu, par construction.

**Un SPI journalier ou hebdomadaire n'existera pas** : le WMO définit le SPI sur des pas **mensuels** (1/3/6/9/12/24). Un « SPI-1 semaine » serait un indicateur inventé — interdit par la doctrine — et statistiquement bancal (l'ajustement gamma s'effondre sur des cumuls courts saturés de zéros).

### 1.2 Le vrai défaut : la pluie journalière existe et n'est pas exposée

Vérifié sur l'entrepôt : `silver.stg_era5_timeseries` porte `temperature_2m`, **`total_precipitation`** et `potential_evaporation` au **pas journalier**, du **1950-01-02 au 2026-07-12**, sur la grille France — **321 324 696 lignes, 11 496 mailles**, et un seul pas de temps distinct observé (`1 day`), donc granularité journalière établie et non inférée.

**⚠️ Source obligatoire : `silver`, jamais `bronze`.** `bronze.era5_france_timeseries` porte exactement les mêmes colonnes, la même plage et le **même nombre de lignes** (321 324 696) — mais **22 985 mailles distinctes au lieu de 11 496**, soit le double. C'est le bug connu de précision des coordonnées ERA5 (doublons flottants entre backfill et incrémental) que la couche silver corrige en arrondissant. Taper `bronze` donnerait une grille désalignée de la carte et des mailles fantômes, sans qu'aucun signal évident ne l'indique.

Unités et sémantique **vérifiées** (le doute méritait d'être levé, ERA5 exprimant nativement `tp` en mètres) :

| Contrôle | Mesure | Verdict |
|---|---|---|
| Précip mensuelle (gold), jan-2025 | médiane 115 mm | réaliste |
| Cumul annuel France 2024 | médiane **1101 mm/an** (max 3952 = Alpes) | réaliste (réf. ≈ 900-1100) |
| Brut journalier, 1→5 jan 2024 | 1,5 / 7,4 / 17,8 / 6,1 / 3,4 mm | réaliste |

`total_precipitation` est donc **déjà en mm** et c'est un **cumul journalier** — le `time` à 00h n'est qu'une étiquette de jour, pas un instant. Aucun bug d'unité.

**Asymétrie constatée** : la température journalière est exposée (Tx/Tn/T moy), la pluie non. Sans raison.

## 2. Décision

Exposer **une couche « Pluie (jour) »** dans la section journalière, à côté de Tx/Tn/T moy. **Périmètre volontairement limité** : pas de cumul 7 jours (défendable — c'est une vraie valeur — mais il exige une agrégation glissante sur ~320 M de lignes, à border séparément), pas d'ETP journalière (une carte d'ETP seule ne dit rien d'exploitable, c'est la raison pour laquelle la mensuelle a été retirée).

Conformité doctrine : la pluie journalière est une **vraie valeur** à domaine réellement absolu et fixe — 20 mm, c'est 20 mm, en janvier comme en juillet. Elle a donc droit à une carte, au même titre que Tx/Tn (cf. §2.1 du spec du 2026-07-16 : « domaine absolu fixe → carte absolue légitime »).

## 3. L'échelle

### 3.1 Le problème, mesuré

Distribution réelle de la pluie journalière (France entière, 11 496 mailles) :

| Jour | part < 1 mm | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| 2025-01-03 (hiver pluvieux) | 28 % | 8,74 | 18,57 | 21,10 | 22,56 |
| 2025-06-15 (été) | 51 % | 0,93 | 6,52 | 16,04 | 26,49 |
| 2024-10-17 (épisode) | 20 % | 4,93 | 21,80 | 49,75 | **98,84** |

Une rampe **linéaire 0-50** placerait **71,3 % du territoire dans ses 5 premiers %** un jour ordinaire (mesuré sur 2025-06-15) : carte quasi uniforme. C'est le défaut de la température mensuelle, en pire — mais pour une autre cause : ici l'**asymétrie** de la distribution, pas un décalage saisonnier.

### 3.2 Ce qu'on ne fait PAS

**Aucun ré-ancrage** (par jour, par saison, par percentile). Ce serait convertir une vraie valeur en encodage relatif, c'est-à-dire fabriquer un indice maison — exactement l'erreur commise puis rejetée sur la température mensuelle. Le domaine reste **absolu et fixe** : deux cartes de deux jours différents restent comparables.

### 3.3 Ce qu'on fait : des classes non linéaires fixes

Le domaine reste absolu ; c'est la **forme** de la rampe qui change. Bornes de la convention météo (Météo-France / ECMWF), en mm :

```
< 0,1  │  0,1–1  │  1–2  │  2–5  │  5–10  │  10–20  │  20–50  │  ≥ 50
```

- **8 classes**, rampe séquentielle **Blues** (ColorBrewer, monotone en luminance, sûre en déficience de vision des couleurs).
- Rendu par une expression MapLibre **`step`** — le mécanisme déjà employé par `climatBilanColorExpression`.
- Couvre le 0 → 98,84 mm réellement observé ; au-delà de 50 mm, saturation dans la classe haute, ce qui **est** l'information.

**Légende = les bornes en mm, sans noms de classes.** Contrairement au SPI/STI/bilan, où le nom (« Très sec ») *porte* l'information parce que la valeur brute est un z-score illisible, ici la valeur **est** le sens : « 10–20 mm » se lit seul. Inventer « faible / modéré / fort » serait éditorialiser sans rien ajouter.

## 4. Backend — deux endpoints calqués sur l'existant

**`GET /api/v1/observatory/climat/daily-precip?date=YYYY-MM-DD`**
→ `[{latitude, longitude, value}]`, copie de `/daily-temp` (`observatory_climat.py:472`), source `silver.stg_era5_timeseries`, colonne `total_precipitation`. Aucune ligne pour la date → **liste vide**, pas de 404 (même convention que `/daily-temp` et `/grid-monthly`).

⚠️ **Contrainte dure** : prédicat de plage sur `time` **sans fonction ni cast sur la colonne de partition** — `WHERE time >= :day AND time < CAST(:day AS date) + INTERVAL '1 day'`. Un cast casse l'exclusion de chunks TimescaleDB et fait scanner les ~320 M de lignes. L'avertissement est déjà inscrit dans le code de `/daily-temp` ; il vaut a fortiori ici, la table étant bien plus grosse que `stg_era5_daily_temp_stats`.

**`GET /api/v1/observatory/climat/daily-precip-range`** → `{min_date, max_date}`.
Justification mesurée — les couvertures **divergent** :

| Table | Couverture |
|---|---|
| `stg_era5_daily_temp_stats` (Tx/Tn/Tmoy) | 1950-01-01 → 2026-07-10 |
| `stg_era5_timeseries` (pluie) | 1950-01-02 → **2026-07-12** |

Réutiliser `/daily-temp-range` pour la pluie **masquerait ses deux jours les plus récents**. Couverture pluie vérifiée continue (401 jours distincts sur une fenêtre de 401).

Cache : `GRID_TTL` pour la couche, 1 h pour la plage — mêmes valeurs que leurs jumeaux température.

## 5. Frontend

- **`lib/climat-colors.ts`** : `ClimatVariable` += `'precip_daily'` (`kind: 'daily'`, `dailyParam` n'a plus de sens ici puisque la source diffère → le résolveur d'endpoint distingue pluie et température) ; constantes `PRECIP_DAILY_BOUNDS` + `PRECIP_DAILY_COLORS` ; `climatPrecipDailyColorExpression()` (expression `step`).
- **Renommage** : `DAILY_TEMP_VARIABLE_ORDER` → **`DAILY_VARIABLE_ORDER`**, et le libellé de section `climat.picker.dailyTempLabel` « Températures journalières » → « **Données journalières** » (fr + en). Le nom devient faux sinon.
- **`ClimatPage.tsx`** : la plage du DayStepper vient de `/daily-precip-range` quand la variable est la pluie, de `/daily-temp-range` sinon.
- **`ClimatLegend.tsx`** : branche « classes en mm » pour `precip_daily` (bornes affichées, pas de noms).
- **`DailyTempBanner`** : l'actuel est thermique (« 43,2 °C · 12 cellules > 35 °C »). Équivalent pluie : « **Pluie max France : 98,8 mm · 3 cellules ≥ 50 mm** ». Le seuil **50 mm est la borne haute de la légende** — donc vérifiable à l'œil sur celle-ci, même principe que le recalage du % sécheresse (2026-07-16). **Aucun seuil inventé.**

## 6. Tests

- **Backend** (`tests/test_observatory_climat.py`) : `_build_daily_precip_points` (formatage, entrée vide → liste vide) ; variable/date invalides → 422 ; **le SQL de `/daily-precip` ne pose ni fonction ni cast sur `time`** (régression perf directement testable sur la chaîne de requête).
- **Frontend** : les 8 bornes de classes et leur ordre ; l'expression `step` produit les bonnes couleurs de part et d'autre de chaque borne ; la légende affiche les bornes en mm ; le picker rend 4 journalières ; le bandeau pluie affiche max + compte ≥ 50 mm ; la plage du DayStepper suit la variable.
- **`npm run build` obligatoire** : vitest ne typecheck pas ; seul `tsc -b` attrape les erreurs de type, et c'est lui qui casse le build de l'image et la CI.
- **Backend** : `DEBUG=true DB_PASSWORD=test uv run pytest <fichiers> -q` (le `.env` du dépôt fait échouer `Settings` sinon ; ne pas lancer la suite complète, elle pend sur l'entrepôt).

## 7. Risques

- **Perf** : le seul risque sérieux. `stg_era5_timeseries` pèse **321 324 696 lignes** (mesuré). Mitigé par l'exclusion de chunks (§4) + le cache `GRID_TTL`. À **mesurer** à l'implémentation : la requête d'un jour doit rester de l'ordre de `/daily-temp` ; si elle dérape, c'est que le prédicat casse l'exclusion de chunks.
- **Mauvaise source** : réel et sournois. Voir §1.2 — `bronze.era5_france_timeseries` est un leurre parfait (mêmes colonnes, même volume, même plage) qui livre 22 985 mailles désalignées. **Toujours `silver`.**
- **Grille** : nul. Vérifié — 11 496 mailles côté silver, identiques à celles des marts gold et du journalier température (jointure exacte, 11 496 communes).
- **Périmètre** : le renommage `DAILY_TEMP_VARIABLE_ORDER` → `DAILY_VARIABLE_ORDER` touche le picker et ses tests ; `tsc` attrape toute occurrence oubliée.
