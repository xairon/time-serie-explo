# Module Climat (Lot 2) — Plan d'implémentation junon

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development ou executing-plans, tâche par tâche. Cases `- [ ]`.

**Goal:** Refondre l'expérience météo/climat de junon en un module pratique pour experts hydro/climat : page « Climat » dédiée (Situation / Point-Zone / Comparaison), backend en `SELECT` simples sur les nouveaux marts grille de l'entrepôt, suppression des précalculs lourds, intégrations carte Observatoire + page Station.

**Contexte amont (entrepôt BRGM, déjà livré — Lot 1)** : `gold.fct_era5_monthly_grid`
(mensuel par cellule 0.1°, 11 496 cellules × 919 mois : temp moy/min/max, `precipitation_totale`,
`etp_totale` et `bilan_hydrique` en mm POSITIFS, `mois_complet`), `gold.fct_era5_climatology_grid`
(normales 1991-2020 par cellule × mois calendaire × fenêtre 1/3/6/12 : moyenne/σ précip+temp,
gamma α/β, `prob_zero`, `nb_annees`), `gold.fct_era5_indices_grid` (SPI/STI par cellule × mois ×
fenêtre, 42 M lignes, calibration vérifiée). Accès via `get_brgm_sync_engine` (existant).
⚠️ Étiquetage : la température est un instantané 00:00 UTC (biais froid ~2-4 °C) jusqu'au cutover
« daily statistics » en cours côté entrepôt — libellés à prévoir en conséquence (« T° à 00h UTC »),
les indices STI/SPI sont exacts.

**Spec de référence** : hubeau_data_integration/docs/superpowers/specs/2026-07-06-era5-climate-module-redesign-design.md (§ Lot 2)

**Stack** : FastAPI (`api/routers/`), React+Vite+TS, MapLibre GL, TanStack Query, react-i18next (fr/en), cache Redis via `dashboard.utils.cache.get_cached`.

## Global Constraints

- Public expert : SPI/STI au premier plan (σ, 7 classes WMO ±0.84/±1.28/±1.75 — mêmes seuils que l'IPS), densité d'information, exports CSV.
- AUCUN recalcul statistique côté backend : tout indice/normale vient des marts. Les endpoints sont des SELECT (+ agrégats simples) avec cache Redis 24 h max.
- Réutiliser les briques existantes : `era5-grid.ts` (carrés 0.05°-half), `era5-colors.ts` (échelles), pattern hooks `useObservatory.ts`, client `observatory-api.ts`.
- i18n fr + en pour toute nouvelle chaîne ; libellés français d'abord.
- Le module MétéEAU (`MeteoNappesPage`, `components/meteo/*`) reste DÉSACTIVÉ — ne pas toucher.
- Tests : endpoints (fixtures pytest existantes du repo), composants (vitest/testing-library si présents — suivre les conventions du repo), e2e léger si harnais présent (`e2e/`).
- Environnement dev : junon-backend-dev :49516 / junon-frontend-dev :49518 ; vérifier les changements en dev, pas en prod.
- Fichiers d'audit de référence (états/lignes au 2026-07-06) : `api/routers/observatory_era5.py` (endpoints actuels + warmers `api/main.py:84-159` + math `api/era5_anomaly.py`), `frontend/src/pages/ObservatoryPage.tsx:84-268` (état overlay), `frontend/src/components/observatory/{RightDrawer,Era5Banner,ObservatoryMap,TimeseriesChart,StationKPICards}.tsx`, `frontend/src/lib/{era5-colors,era5-grid,era5-zones}.ts`, `frontend/src/hooks/useObservatory.ts:334-396`, `frontend/src/routes.tsx:14-47`, `frontend/src/components/layout/TopNav.tsx:28-29`, SPI station `api/routers/observatory_piezo.py:397`.

---

## Phase A — Backend sur les nouveaux marts

### Task A1 : Nouveaux endpoints climat (lecture marts)

**Files:** Create `api/routers/observatory_climat.py` (prefix `/api/v1/observatory/climat`) ; wire dans `api/main.py` (sans auth, comme le router era5). Tests `api/tests/` (suivre conventions).

**Endpoints (tous cache 24 h, moteur BRGM) :**
- `GET /grid-monthly?month=YYYY-MM&variable=` → par cellule : valeur de `fct_era5_monthly_grid` (temp/precip/etp/bilan) pour le mois.
- `GET /grid-indices?month=&window=&index=spi|sti` → par cellule : SPI ou STI depuis `fct_era5_indices_grid`.
- `GET /situation-summary?month=&window=` → agrégats territoire : % cellules par classe WMO (7 classes sur SPI), % en sécheresse (spi < −1), rang du mois vs historique (« plus sec depuis AAAA » : comparer la médiane spatiale du SPI du mois aux mêmes mois calendaires 1950→présent), top-5 zones les plus sèches (moyenne SPI par département via jointure spatiale simple arrondie — si trop coûteux, retourner les 5 cellules min avec lat/lon et laisser le front géocoder grossièrement).
- `GET /point-series?lat=&lon=&from=&to=` → série mensuelle complète de la cellule la plus proche (arrondi 0.1) : mensuel + normale du mois calendaire (jointure climatologie fenêtre 1) + SPI/STI 4 fenêtres. Retourner aussi les métadonnées cellule (lat/lon effectifs).
- `GET /point-episodes?lat=&lon=&window=3` → épisodes de sécheresse de la cellule : séquences de mois consécutifs spi < −1 (SQL fenêtré : îlots par cumul de ruptures), colonnes début, fin, durée_mois, spi_min, déficit cumulé (Σ (precip − precip_moyenne_normale) sur l'épisode). Trié par durée desc.
- `GET /compare-years?lat=&lon=&years=1976,2003,2026` → par année demandée : cumul pluviométrique mensuel glissant (jan→déc) + la normale ; et par année : SPI de chaque mois (fenêtre 3) pour petits multiples.
- `GET /export-point.csv?lat=&lon=` → CSV du point (mensuel + indices), streaming.

- [ ] Écrire le router (requêtes SQL paramétrées, mêmes patterns que observatory_era5.py : `text()` + engine sync + get_cached)
- [ ] Tests endpoints (au moins : nominal + cellule inconnue → 404/vide propre + validation params)
- [ ] Vérif manuelle en dev sur 2-3 cellules connues (Tours 47.4/0.7)
- [ ] Commit

### Task A2 : Bascule des endpoints ERA5 existants sur les marts + suppression des warmers

**Files:** Modify `api/routers/observatory_era5.py`, `api/main.py`, `api/era5_anomaly.py` (réduction).

- [ ] `/spi` et `/sti` lisent `fct_era5_indices_grid` (plus de gamma à la volée) ; `/monthly` lit `fct_era5_monthly_grid` ; `/snapshot` inchangé (daily) mais retirer la fusion pondérée des doublons de coordonnées (l'amont est propre + arrondi défensif)
- [ ] Supprimer `_warm_era5_climatology`, locks single-flight, re-warm 6 jours (`api/main.py:84-159`) ; supprimer de `era5_anomaly.py` le code de fit gamma devenu mort (garder `classify_index` si consommé)
- [ ] Supprimer les endpoints `/anomaly` et `/temp-anomaly` APRÈS vérification qu'aucun composant frontend ne les appelle encore (grep `era5.anomaly|temp-anomaly` dans frontend/src — l'audit indique la variable `anomaly` orpheline côté UI ; la Task C1 retire le code front en même temps)
- [ ] SPI station (`observatory_piezo.py:397` et équivalent hydro) : lecture directe `fct_era5_indices_grid` via la cellule mappée de la station (jointure `int_station_era5_mapping`), fenêtre paramétrable
- [ ] Tests de non-régression sur les endpoints conservés (mêmes shapes de réponse) ; mesure avant/après du temps de premier hit (attendu : sub-seconde vs 71 s)
- [ ] Commit

## Phase B — Page « Climat »

### Task B1 : Route + vue Situation

**Files:** Create `frontend/src/pages/ClimatPage.tsx` + `frontend/src/components/climat/*` ; Modify `routes.tsx`, `TopNav.tsx` (entrée « Climat »), `observatory-api.ts` (client `climat.*`), hooks (`useClimat.ts`).

- [ ] Route `/climat`, entrée nav « Climat » (l'emplacement laissé par MétéEAU)
- [ ] Vue Situation (défaut) : carte plein écran MapLibre réutilisant `era5PointsToSquares`/`era5-colors` ; variables : SPI (défaut, fenêtre 3), STI, bilan hydrique, précip, temp (libellé « T° 00h UTC »), ETP ; sélecteurs mois (‹/›) + fenêtre (1/3/6/12) ; légende 7 classes
- [ ] Bandeau de synthèse (données `/situation-summary`) : « X % du territoire en sécheresse (SPI < −1) · mois le plus sec depuis AAAA · zones les plus touchées : … »
- [ ] i18n fr/en ; tests composants du bandeau + du sélecteur
- [ ] Commit

### Task B2 : Vue Point/Zone

- [ ] Clic cellule sur la carte Situation → panneau latéral large (ou vue dédiée) : graphique précip mensuelle vs normale (barres + ligne), courbes SPI/STI multi-fenêtres, zoom/brush 1950→présent
- [ ] Tableau des épisodes de sécheresse (`/point-episodes`) : début, fin, durée, SPI min, déficit mm — triable ; l'épisode en cours (si spi<−1 au dernier mois) mis en évidence
- [ ] Recherche commune/département → centre la carte + sélectionne la cellule (réutiliser le géocodage existant du repo s'il y en a un — sinon simple recherche par coordonnées/nom de la barre existante)
- [ ] Export CSV du point (bouton → `/export-point.csv`)
- [ ] Tests composants (tableau épisodes avec fixture) ; commit

### Task B3 : Vue Comparaison

- [ ] Sélection zone (cellule courante) + multi-sélection d'années → courbes de cumul pluviométrique superposées (une par année, normale en référence)
- [ ] Petits multiples : cartes SPI du même mois sur N années (réutiliser la carte en mini, données `/grid-indices` par année)
- [ ] Presets d'années de sécheresse célèbres (1976, 1989, 2003, 2022) ; tests ; commit

## Phase C — Intégrations & nettoyage

### Task C1 : Overlay Observatoire simplifié

- [ ] Retirer la variable fantôme `anomaly` (code, couleurs, i18n, hooks — et l'endpoint côté A2)
- [ ] Exposer `evaporation` (ETP) dans les radios du RightDrawer (config déjà présente dans era5-colors)
- [ ] Popup cellule : ajouter lien « Analyser dans Climat → » (navigue vers /climat, vue Point pré-remplie lat/lon/mois)
- [ ] Libellé température : « T° à 00h UTC » (tooltip explicatif) ; commit

### Task C2 : Section climat page Station

- [ ] `StationPage` : section « Contexte climatique » — SPI local (cellule mappée, via endpoint station A2), cumuls glissants 3/6/12 vs normale, à côté des barres de précip existantes du TimeseriesChart
- [ ] KPI cards : ajouter classe SPI courante du point ; commit

### Task C3 : Finitions

- [ ] Passe i18n complète (fr/en), vérif accessibilité de base (contrastes légende)
- [ ] Docs repo (README/docs) : le module Climat, ses endpoints, la provenance entrepôt
- [ ] Nettoyage : imports/exports morts suite aux suppressions A2/C1 ; commit final + revue de branche

## Ordre & dépendances

A1 → B1 → B2 → B3 (le front consomme A1) ; A2 indépendant après A1 (bascule + suppressions) ;
C1 dépend de A2 (retrait endpoint anomaly) ; C2 dépend de A2 (endpoint SPI station) ; C3 en dernier.
Branche : à créer depuis la base la plus fraîche du repo (au 2026-07-07 : `audit/comprehensive-fixes`,
11 commits devant main — confirmer avec l'utilisateur si l'audit doit être mergé d'abord).
