# Spec — IPS à référence fixe + listes de stations groupées

Date : 2026-06-03
Statut : validé (design), à planifier
Dépôts impactés : `time-serie-explo` (app) **et** `hubeau_data_integration` (warehouse)

## Contexte et problème

Deux constats issus d'un audit de l'Observatoire et de la page Prévision.

### 1. Le mapping IPS ↔ niveau de nappe n'est pas stationnaire et incohérent

L'IPS (Indicateur Piézométrique Standardisé, méthodo BRGM RP-64147-FR) classe un
niveau de nappe en comparant la valeur du mois à la distribution **du même mois
calendaire** sur une période de référence. Les bandes « bas / normal / haut » sont
donc `borne = quantile_mois(cutoff)`.

Aujourd'hui il existe **quatre** calculs de cette référence, tous différents :

| Usage | Code | Référence | Évolue ? |
|---|---|---|---|
| Index courant (Observatoire) | `gold.station_current_index` (warehouse) | KDE par mois sur **tout l'historique** ; `baseline_start = 1ʳᵉ mesure`, `baseline_end = dernière` | continu |
| Série SPLI (`/spli`) | `dashboard/utils/drought.py` | KDE par mois sur **tout l'historique** | continu |
| Barre de seuils (fiche station) | requête `LATERAL` percentile dans `observatory_piezo.py` | percentiles par mois sur **tout l'historique** | continu |
| Bandes de prévision | `dashboard/utils/counterfactual/ips.py` | µ/σ gaussien sur les **données d'entraînement**, figé dans l'artefact MLflow | figé au train |

Conséquences :
- **Shifting baseline** : une baisse tendancielle longue est absorbée dans le « nouveau
  normal » → l'anomalie des niveaux bas récents est sous-estimée.
- **Non reproductible** : un mois passé peut changer de classe quand on ajoute des années.
- **Incohérence** : pour une **même station**, l'Observatoire et la Prévision peuvent
  afficher des bandes différentes.
- **Duplication** : `drought.py` (app) ≈ `ml/indices.py` (warehouse).

`dashboard/utils/counterfactual/ips.py` documente d'ailleurs « Default reference period:
1981–2010 » mais le code l'ignore (plein historique) ou retombe sur la série de train.

### 2. Listes de stations du même groupe absentes / partielles

- **Piézo** : le panneau « autres piézomètres du même bassin BDLISA » a été **retiré**
  (commit `6ccd356`, jugé sans action). À réintroduire.
- **Hydro** : le panneau « stations du même site » existe encore (endpoint + UI) mais
  ne s'affiche que pour les sites multi-stations (720 / 5349). Pas une régression, mais
  à harmoniser avec le piézo et à enrichir d'un niveau de regroupement.

## Décisions validées

- IPS : **période de référence fixe** (défaut **1991–2020**, configurable) avec repli
  intelligent par station ; recalcul seulement sur cadence décennale.
- Méthode unifiée : **empirique / quantiles** (méthodo BRGM KDE), on abandonne le
  z-score gaussien de `ips.py`.
- La page Prévision **consomme la même référence partagée** (plus de gel au train).
- Regroupement piézo : **BDLISA exact (nappe)** + **préfixe BDLISA (système)**.
- Regroupement hydro : **site** + **cours d'eau**.
- Périmètre : référence **calculée dans le warehouse** (couche gold), l'API lit.

---

## Partie A — IPS à référence fixe (warehouse + app)

### A.1 Module de calcul unique (warehouse)

Centraliser dans `hubeau_data_integration/.../ml/indices.py` une fonction pure :

```
compute_reference_grid(monthly_series, ref_period=(1991, 2020),
                       min_years=15, min_per_month=10) -> ReferenceResult
```

`ReferenceResult` par station :
- `grid` : pour chaque mois 1–12, une **grille de quantiles** (percentiles 1..99) de la
  moyenne mensuelle (m NGF), calculée par KDE sur la fenêtre retenue. Représentation
  compacte, suffisante pour (i) les bandes de classe (quantile au cutoff) et (ii) la
  série SPLI (valeur → percentile → `norm.ppf`).
- `baseline_start`, `baseline_end` : bornes de la fenêtre **réellement** utilisée.
- `flag` : `normale` (≥15 ans dans 1991–2020) | `adaptee` (meilleure fenêtre 30 ans
  ≥15 ans) | `provisoire` (<15 ans → historique complet).
- `n_years`, `n_per_month` (par mois).

**Échelle de repli** (par station) :
1. ≥15 ans de moyennes mensuelles dans `[1991-01, 2020-12]` → fenêtre = 1991–2020, `flag=normale`.
2. Sinon : fenêtre glissante de 30 ans (alignée décennie) maximisant les années dispo,
   si ≥15 ans → `flag=adaptee`.
3. Sinon : historique complet → `flag=provisoire`.

Les 7 classes BRGM utilisent les cutoffs CDF déjà en place :
`[0.0401, 0.1003, 0.2005, 0.7995, 0.8997, 0.9599]` (bornes des bandes), plus la médiane.

### A.2 Table gold matérialisée

Nouvel asset dagster `station_reference_stats` → table
`gold.station_reference_stats` :

| colonne | type | note |
|---|---|---|
| `type` | text | `piezo` \| `hydro` |
| `code` | text | `code_bss` \| `code_station` |
| `month` | int | 1–12 |
| `quantile_grid` | jsonb | percentiles 1..99 (m NGF, ou m³/s pour hydro) |
| `baseline_start` | date | fenêtre retenue |
| `baseline_end` | date | |
| `flag` | text | `normale` \| `adaptee` \| `provisoire` |
| `n_years` | int | |
| `computed_at` | timestamptz | |

PK : `(type, code, month)`. Source : `gold.fct_monthly_chroniques` (piézo) et
`gold.fct_monthly_hydro` (hydro).

`station_current_index` est **modifié** pour classer la valeur courante contre cette
référence fixe (z = `norm.ppf(CDF_grille(valeur))`) ; `baseline_start/end` reflètent
désormais la fenêtre fixe, plus le plein historique.

### A.3 Lecture côté app (`time-serie-explo`)

L'API devient un **lecteur** de `gold.station_reference_stats`. Suppression du calcul
de référence côté app.

- `/spli` (`observatory_piezo.py`/`observatory_hydro.py`) : lit la grille, calcule la
  série z par interpolation `valeur → percentile → norm.ppf`. `drought.py` est réduit à
  l'application de la transformation (ou supprimé si plus rien ne l'utilise).
- Fiche station (`get_station`) : la sous-requête `LATERAL` percentile est remplacée par
  une lecture de `quantile_grid` au mois de référence ; expose `flag`, `baseline_start/end`.
- **Prévision** (`api/routers/counterfactual.py`, `dashboard/utils/training.py`) :
  - on **arrête** de figer la référence dans l'artefact au moment du train
    (`ips_meta` / `compute_all_ips_references`).
  - les bandes (`compute_monthly_ips_bounds`) sont calculées depuis la grille de la
    station identifiée par `code_bss` (déjà stocké comme `station_name`).
  - `ips.py` est ramené aux helpers de classification (cutoffs, bandes) qui consomment la
    grille ; le calcul de référence (`compute_ips_reference*`) est retiré.

### A.4 Transparence UI

- Tooltip de l'échelle IPS : « Référence : 1991–2020 (30 ans) » ou « période adaptée
  (AAAA–AAAA) » / « provisoire — série < 15 ans ».
- La page Prévision affiche les mêmes bandes que l'Observatoire pour une station donnée.

### A.5 Cadence

La période est une **constante de config** (`IPS_REF_PERIOD`). Le recalcul = changer la
constante (≈ tous les 10 ans : 1991–2020 → 2001–2030 en 2031) puis rematérialiser l'asset.
Pas de dérive à chaque nouveau mois.

---

## Partie B — Listes de stations groupées

### B.1 Backend

**Piézo** — nouvel endpoint :
`GET /observatory/piezo/stations/{code_bss}/siblings?level=nappe|systeme`
- `nappe` → `WHERE codes_bdlisa = :code_exact`.
- `systeme` → `WHERE codes_bdlisa LIKE :prefixe || '%'` (préfixe = entité parent
  BDLISA, ex. `101AC` pour `101AC01`).
- Réponse : `{ level, code_bdlisa, libelle, nb_stations, non_rattachee: bool, siblings: [...] }`.
- Station sans `codes_bdlisa` → `non_rattachee: true`, `siblings: []`.
- Schéma `PiezoBdlisaSiblings` (réintroduit dans `api/schemas/observatory.py`).

**Hydro** — endpoint `siblings` existant **étendu** :
`GET /observatory/hydro/stations/{code_station}/siblings?level=site|cours_eau`
- `site` = comportement actuel (`WHERE code_site = …`).
- `cours_eau` → `WHERE code_cours_eau = …`.

### B.2 Frontend

Composant unique `SiblingStationsPanel` (`frontend/src/components/observatory/`) :
- Affiché dans `pages/StationPage.tsx` (détail) **et**
  `components/observatory/StationDrawer.tsx`.
- Toggle 2 positions : piézo *Nappe / Système* · hydro *Site / Cours d'eau*.
- Chaque voisine = **lien cliquable** (navigation opt-in) + pastille de classe IPS/SSFI.
- **Aucun filtrage de carte, aucune action implicite** (règle « pas de filtrage implicite »).
- Panneau **masqué** si 0 voisine ; piézo non rattachée → message
  « Station non rattachée à une entité BDLISA ».
- Hook `usePiezoSiblings(code, level)` + `useHydroSiblings(code, level)` ; clients dans
  `lib/observatory-api.ts` ; types dans `lib/observatory-types.ts`.
- i18n : clés FR/EN (libellés toggle, message non-rattachée).

### B.3 Réutilisation / nettoyage

Le rendu hydro inline actuel (`StationPage.tsx:120`, `StationDrawer.tsx:111`) est
remplacé par `SiblingStationsPanel`.

---

## Hors périmètre (YAGNI)

- Pas de filtrage de la carte ni de sélection automatique depuis ces listes.
- Pas de niveau BDLISA au-delà de exact + préfixe (pas d'arbre hiérarchique complet).
- Pas de recalcul automatique annuel de la référence (cadence décennale manuelle assumée).
- Pas de migration des artefacts MLflow existants : les modèles déjà entraînés portent
  `ips_meta` ; il sera ignoré au profit de la grille warehouse (lecture par `code_bss`).

## Plan de tests

- **Warehouse** : tests unitaires `compute_reference_grid` — repli normale/adaptée/
  provisoire ; mois manquants ; <10 obs/mois ; cohérence grille↔cutoffs.
- **App** : test API `/spli` et bandes prévision pour une station avec `flag=normale` et
  une avec `flag=provisoire` ; cohérence Observatoire↔Prévision sur une même station.
- **Siblings** : station seule (liste vide, panneau masqué) ; site/système multi-stations ;
  piézo sans BDLISA (`non_rattachee`).

## Séquence de mise en œuvre

1. Warehouse : `compute_reference_grid` + asset/table `station_reference_stats` + maj
   `station_current_index`. Matérialiser.
2. App : endpoints siblings (piézo + hydro `level`) + `SiblingStationsPanel` (indépendant
   du warehouse, livrable en premier).
3. App : `/spli`, fiche station, prévision → lecture de la grille ; retrait du calcul de
   référence côté app + nettoyage `drought.py`/`ips.py`.
4. UI transparence (tooltips `flag`/baseline).
