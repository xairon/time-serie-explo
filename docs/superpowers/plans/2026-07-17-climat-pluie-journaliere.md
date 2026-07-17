# Pluie journalière en couche carte — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exposer une couche carte « Pluie (jour) » à côté de Tx/Tn/T moy, en classes météo non linéaires fixes.

**Architecture:** Deux endpoints backend calqués sur `/daily-temp` mais lisant `silver.stg_era5_timeseries` (321 M lignes, grille identique aux marts). Côté front, une 6ᵉ `ClimatVariable` de `kind: 'daily'` rendue par une expression MapLibre `step` — le mécanisme déjà utilisé par la carte du bilan hydrique. Trois helpers existants (backend et front) sont **génériques mais mal nommés** (`*_daily_temp_*`) : la pluie les réutilise, donc on les renomme.

**Tech Stack:** FastAPI + SQLAlchemy Core (`text()`) sur PostgreSQL/TimescaleDB ; React 18 + TypeScript, MapLibre, vitest, react-i18next.

**Spec:** `docs/superpowers/specs/2026-07-17-climat-pluie-journaliere-design.md`

## Global Constraints

- **Doctrine produit** : soit un **vrai indicateur** (IPS, SPLI, SPI, STI — standard reconnu, seuils publiés), soit une **vraie valeur** (°C/mm). Jamais un intermédiaire inventé. **Toute échelle ré-ancrée sur la climatologie EST un indice maison → interdit.** La pluie journalière garde un domaine **absolu et fixe** : 20 mm, c'est 20 mm, en janvier comme en juillet.
- **Source obligatoire : `silver.stg_era5_timeseries`. JAMAIS `bronze.era5_france_timeseries`** — le bronze a les mêmes colonnes, la même plage et le même nombre de lignes, mais **22 985 mailles au lieu de 11 496** (bug connu de précision des coordonnées ERA5). Il livrerait une grille désalignée sans aucun signal.
- **JAMAIS de fonction ni de cast sur `time`** (colonne de partition TimescaleDB) dans un prédicat. **Mesuré** : `WHERE time >= :d AND time < CAST(:d AS date) + INTERVAL '1 day'` → **413 ms** ; `WHERE time::date = :d` → **69 032 ms**. Facteur **167**.
- **Pour min/max : caster le RÉSULTAT, pas la colonne.** **Mesuré** : `min(time)::date` → **224 ms** ; `min(time::date)` → **3 036 ms**. (Le code existant de `/daily-temp-range` utilise la forme lente ; c'est tolérable sur son petit mart, pas ici. **Ne pas le modifier** — hors périmètre.)
- **Aucune modification entrepôt.** Lecture seule.
- **Ne pas toucher** : SPI/STI/bilan, le `PointPanel`, `PrecipNormalChart`, `era5-colors.ts` / `era5-zones.ts` (module Observatory, type `Era5Variable` — sans rapport).
- **UI en français** (public BRGM). Toute chaîne visible via i18n, dans `fr.json` **et** `en.json`.
- **`npm run build` obligatoire avant chaque commit front** : vitest ne typecheck PAS ; seul `tsc -b` attrape les erreurs de type, et c'est lui qui casse le build de l'image et la CI.
- **Tests backend** : `DEBUG=true DB_PASSWORD=test uv run pytest <fichiers> -q` depuis la racine (le `.env` du dépôt fait échouer `Settings` sinon). **Ne jamais lancer la suite complète** — elle pend sur l'entrepôt.
- Commandes npm depuis `frontend/`.

**Bornes de classes (identiques partout, backend comme front) :** `0.1 · 1 · 2 · 5 · 10 · 20 · 50` mm → 8 classes.

---

### Task 1: Backend — les deux endpoints pluie

Les helpers `_build_daily_temp_points` et `_build_daily_temp_range` sont déjà **génériques** (ils formatent `{latitude, longitude, value}` et `{min_date, max_date}`, rien de thermique). La pluie les réutilise telles quelles ; on les renomme pour que le nom cesse de mentir. Renommage + réutilisation + nouveaux endpoints forment un seul déliverable : `tsc`… pardon, pytest ne compile pas, mais un renommage à moitié fait casse les tests immédiatement.

**Files:**
- Modify: `api/routers/observatory_climat.py` (renommages ; nouveaux endpoints après `/daily-temp-range`, ~ligne 523)
- Test: `tests/test_observatory_climat.py`

**Interfaces:**
- Consumes: `_parse_date(value: str) -> date` (lève `HTTPException(422)`), `_num(v) -> float | None`, `get_brgm_sync_engine()`, `get_cached(key, params, ttl, fetch)`, `GRID_TTL`, `DAILY_TEMP_RANGE_TTL` — tous existants dans ce fichier.
- Produces: `GET /api/v1/observatory/climat/daily-precip?date=YYYY-MM-DD` → `[{latitude, longitude, value}]` ; `GET /api/v1/observatory/climat/daily-precip-range` → `{min_date, max_date}` (ISO `YYYY-MM-DD` ou `null`). Helpers renommés : `_build_daily_points(rows)`, `_build_daily_range(min_date, max_date)`.

- [ ] **Step 1: Écrire les tests qui échouent**

Dans `tests/test_observatory_climat.py`, ajouter l'import en tête du bloc `from api.routers.observatory_climat import (` :

```python
    _build_daily_points,
    _build_daily_range,
    _DAILY_PRECIP_SQL,
```

Puis ajouter cette classe de tests à la fin du fichier :

```python
class TestDailyPrecip:
    def test_build_daily_points_formats_cells(self):
        rows = [{"latitude": 47.4, "longitude": 0.7, "value": 12.5}]
        assert _build_daily_points(rows) == [{"latitude": 47.4, "longitude": 0.7, "value": 12.5}]

    def test_build_daily_points_empty_input_yields_empty_list(self):
        # Pas de 404 : "aucune donnée pour ce jour" est une réponse attendue
        # (couverture partielle), pas une erreur. Même convention que /grid-monthly.
        assert _build_daily_points([]) == []

    def test_build_daily_range_is_none_safe(self):
        assert _build_daily_range(None, None) == {"min_date": None, "max_date": None}

    def test_daily_precip_reads_silver_never_bronze(self):
        # bronze.era5_france_timeseries a les mêmes colonnes, la même plage et le
        # même nombre de lignes que silver, mais 22 985 mailles au lieu de 11 496
        # (doublons flottants ERA5). Y taper donnerait une grille désalignée sans
        # aucun signal visible.
        assert "silver.stg_era5_timeseries" in _DAILY_PRECIP_SQL
        assert "bronze" not in _DAILY_PRECIP_SQL

    def test_daily_precip_never_casts_the_partition_column(self):
        # Régression PERF, mesurée : un cast/une fonction sur `time` casse
        # l'exclusion de chunks TimescaleDB — 413 ms -> 69 032 ms (×167) sur les
        # 321 M de lignes de la table.
        assert "time::date" not in _DAILY_PRECIP_SQL
        assert "date(time)" not in _DAILY_PRECIP_SQL
        assert "CAST(time" not in _DAILY_PRECIP_SQL.replace(" ", "")
        # la forme correcte : borne basse >= et borne haute < jour+1
        assert "time >= :day" in _DAILY_PRECIP_SQL
        assert "INTERVAL '1 day'" in _DAILY_PRECIP_SQL
```

- [ ] **Step 2: Lancer les tests pour les voir échouer**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -q`
Expected: FAIL au collect — `ImportError: cannot import name '_build_daily_points'`.

- [ ] **Step 3: Renommer les deux helpers génériques**

Dans `api/routers/observatory_climat.py` :

- `def _build_daily_temp_points(rows)` → `def _build_daily_points(rows)`. Adapter sa docstring : remplacer la mention de `stg_era5_daily_temp_stats` par « pour une date donnée (température ou pluie — le formatage est identique) ».
- `def _build_daily_temp_range(min_date, max_date)` → `def _build_daily_range(min_date, max_date)`. Même traitement de docstring.
- Mettre à jour leurs appels dans `/daily-temp` et `/daily-temp-range`.
- Mettre à jour toute référence dans `tests/test_observatory_climat.py` (chercher `_build_daily_temp_points` / `_build_daily_temp_range`).

- [ ] **Step 4: Ajouter les deux endpoints**

Toujours dans `api/routers/observatory_climat.py`, juste après `/daily-temp-range` (~ligne 523), ajouter :

```python
# SQL sorti en constante pour être testable : deux régressions coûteuses s'y
# cachent (la source silver vs bronze, et l'absence de cast sur `time`), et un
# test qui lit la chaîne les attrape sans toucher l'entrepôt.
_DAILY_PRECIP_SQL = """
    SELECT latitude, longitude, total_precipitation AS value
    FROM silver.stg_era5_timeseries
    -- jamais de fonction/cast sur la colonne de partition (time) : casse
    -- l'exclusion de chunks TimescaleDB. Mesuré sur cette table (321 M lignes) :
    -- ce prédicat = 413 ms, `WHERE time::date = :day` = 69 032 ms (×167).
    -- silver et NON bronze : le bronze porte 22 985 mailles au lieu de 11 496
    -- (doublons flottants ERA5) et serait désaligné de la carte.
    WHERE time >= :day AND time < CAST(:day AS date) + INTERVAL '1 day'
"""


@router.get("/daily-precip")
def get_daily_precip(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
):
    """Cumul de précipitation journalier par maille (ERA5, mm) depuis
    ``silver.stg_era5_timeseries`` pour une date exacte. Aucune ligne pour la date
    -> liste vide, même convention que ``/daily-temp`` et ``/grid-monthly`` (pas de
    404 : la couverture avance avec l'ingestion, "pas encore de données" est une
    réponse attendue)."""
    day = _parse_date(date)

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(_DAILY_PRECIP_SQL), {"day": day}).mappings().all()
        return _build_daily_points(rows)

    return get_cached("obs_climat_daily_precip", {"date": str(day)}, GRID_TTL, fetch)


@router.get("/daily-precip-range")
def get_daily_precip_range():
    """Bornes de dates de la couche pluie journalière. Endpoint distinct de
    ``/daily-temp-range`` parce que les couvertures DIVERGENT (mesuré : température
    -> 2026-07-10, pluie -> 2026-07-12) : réutiliser celle de la température
    masquerait les jours de pluie les plus récents.

    ``min(time)::date`` et non ``min(time::date)`` : caster la colonne empêche
    d'utiliser l'index/les métadonnées de chunk — mesuré 224 ms contre 3 036 ms."""

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            min_date = conn.execute(
                text("SELECT min(time)::date FROM silver.stg_era5_timeseries")
            ).scalar()
            max_date = conn.execute(
                text("SELECT max(time)::date FROM silver.stg_era5_timeseries")
            ).scalar()
        return _build_daily_range(min_date, max_date)

    return get_cached("obs_climat_daily_precip_range", {}, DAILY_TEMP_RANGE_TTL, fetch)
```

- [ ] **Step 5: Lancer les tests**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -q`
Expected: PASS (tous, y compris les tests préexistants après renommage).

- [ ] **Step 6: Mesurer la vraie latence contre l'entrepôt**

Le backend de prod tourne déjà (conteneur `junon-backend`, port 49514) mais sert l'ancien code. Redémarre-le pour prendre le nouveau code, puis mesure :

```bash
docker compose up -d --build backend
# attendre ~90 s : le backend préchauffe les références SPI/STI au démarrage
until [ "$(docker inspect --format '{{.State.Health.Status}}' junon-backend)" = healthy ]; do sleep 10; done
curl -s -o /dev/null -w "daily-precip       -> HTTP %{http_code} en %{time_total}s\n" \
  "http://localhost:49514/api/v1/observatory/climat/daily-precip?date=2025-06-15"
curl -s -w "\ndaily-precip-range -> %{time_total}s\n" \
  "http://localhost:49514/api/v1/observatory/climat/daily-precip-range"
```

Expected: `daily-precip` HTTP 200 en **< 2 s** (mesuré à 413 ms côté SQL ; le reste est sérialisation de 11 496 points). `daily-precip-range` renvoie `{"min_date":"1950-01-02","max_date":"2026-07-12"}` en **< 1 s**.
**Si `daily-precip` dépasse 5 s, STOP** : le prédicat a perdu l'exclusion de chunks. Rapporte-le, ne contourne pas.

- [ ] **Step 7: Commit**

```bash
git add api/routers/observatory_climat.py tests/test_observatory_climat.py
git commit -m "feat(climat): endpoints /daily-precip et /daily-precip-range

Lit silver.stg_era5_timeseries (jamais bronze : 22 985 mailles au lieu de
11 496, doublons flottants ERA5). Prédicat de plage sans cast sur time —
mesuré 413 ms contre 69 s avec un time::date, l'exclusion de chunks
TimescaleDB fait tout le travail sur 321 M de lignes.

Range dédié parce que les couvertures divergent (temp 07-10, pluie 07-12),
et min(time)::date plutôt que min(time::date) : 224 ms contre 3 036 ms.

Les helpers _build_daily_temp_points/_range étaient déjà génériques : ils
perdent leur 'temp' puisque la pluie les réutilise."
```

---

### Task 2: Frontend — la variable, les classes, l'expression `step`

**Files:**
- Modify: `frontend/src/lib/climat-colors.ts`
- Test: `frontend/src/lib/climat-colors.test.ts`

**Interfaces:**
- Consumes: rien de la Task 1 (côté client).
- Produces: `ClimatVariable` inclut `'precip_daily'` ; `PRECIP_DAILY_BOUNDS: number[]` = `[0.1, 1, 2, 5, 10, 20, 50]` ; `PRECIP_DAILY_COLORS: string[]` (8 entrées) ; `climatPrecipDailyColorExpression(): unknown[]` ; `DAILY_VARIABLE_ORDER: ClimatVariable[]` = `['tmax', 'tmin', 'tmean', 'precip_daily']` (ex-`DAILY_TEMP_VARIABLE_ORDER`).

- [ ] **Step 1: Écrire les tests qui échouent**

Dans `frontend/src/lib/climat-colors.test.ts`, ajouter aux imports `PRECIP_DAILY_BOUNDS`, `PRECIP_DAILY_COLORS`, `climatPrecipDailyColorExpression`, `DAILY_VARIABLE_ORDER`, puis :

```ts
describe('pluie journalière — classes météo fixes', () => {
  it('a 8 classes pour 7 bornes', () => {
    expect(PRECIP_DAILY_BOUNDS).toEqual([0.1, 1, 2, 5, 10, 20, 50])
    expect(PRECIP_DAILY_COLORS).toHaveLength(PRECIP_DAILY_BOUNDS.length + 1)
  })

  it('produit une expression step alignée EXACTEMENT sur les bornes', () => {
    // Le domaine est absolu et fixe : aucune borne ne dépend du jour affiché.
    // Ré-ancrer par jour/saison ferait de la couleur un encodage relatif, donc
    // un indice maison — interdit par la doctrine.
    const expr = climatPrecipDailyColorExpression() as unknown[]
    expect(expr[0]).toBe('step')
    expect(expr[1]).toEqual(['get', 'value'])
    expect(expr[2]).toBe(PRECIP_DAILY_COLORS[0])       // valeur < 0.1 -> classe sèche
    // puis (borne, couleur) alternés
    PRECIP_DAILY_BOUNDS.forEach((b, i) => {
      expect(expr[3 + i * 2]).toBe(b)
      expect(expr[4 + i * 2]).toBe(PRECIP_DAILY_COLORS[i + 1])
    })
    expect(expr).toHaveLength(3 + PRECIP_DAILY_BOUNDS.length * 2)
  })

  it('range la pluie parmi les variables journalières, pas les indices', () => {
    expect(DAILY_VARIABLE_ORDER).toEqual(['tmax', 'tmin', 'tmean', 'precip_daily'])
    expect(isClimatDailyVariable('precip_daily')).toBe(true)
    expect(isClimatIndexVariable('precip_daily')).toBe(false)
  })
})
```

- [ ] **Step 2: Lancer les tests pour les voir échouer**

Run: `npx vitest run src/lib/climat-colors.test.ts`
Expected: FAIL — `PRECIP_DAILY_BOUNDS` n'est pas exporté.

- [ ] **Step 3: Implémenter**

Dans `frontend/src/lib/climat-colors.ts` :

Élargir le type :

```ts
export type ClimatVariable =
  | 'spi' | 'sti' | 'bilan_hydrique'
  | 'tmax' | 'tmin' | 'tmean' | 'precip_daily'
```

Ajouter après `DAILY_TEMP_STOPS` :

```ts
/** Bornes de classes de la pluie journalière (mm), convention des cartes météo
 *  (Météo-France/ECMWF). FIXES et ABSOLUES : 20 mm c'est 20 mm, en janvier comme
 *  en juillet — deux jours restent comparables. Ne JAMAIS les ré-ancrer sur le
 *  jour affiché : la couleur deviendrait un encodage relatif, c'est-à-dire un
 *  indice maison (cf. spec 2026-07-16, l'erreur commise puis rejetée sur la
 *  température mensuelle).
 *
 *  Pourquoi non linéaires : mesuré sur la grille France, une rampe linéaire 0-50
 *  place 71 % du territoire dans ses 5 premiers % un jour ordinaire (la moitié
 *  des mailles est sous 1 mm). Les classes rendent la carte lisible sans toucher
 *  au domaine. Couvre le 0 -> 98,8 mm réellement observé ; au-delà de 50 mm, la
 *  saturation dans la classe haute EST l'information. */
export const PRECIP_DAILY_BOUNDS: number[] = [0.1, 1, 2, 5, 10, 20, 50]

/** ColorBrewer Blues 8 classes — séquentielle mono-teinte, monotone en luminance,
 *  sûre en déficience de vision des couleurs. La première est quasi blanche : elle
 *  porte le « sec » (< 0,1 mm). */
export const PRECIP_DAILY_COLORS: string[] = [
  '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594',
]

/** MapLibre 'step' : classe discrète depuis `value` (mm), bornes alignées EXACTEMENT
 *  sur PRECIP_DAILY_BOUNDS pour que la carte et la légende ne puissent pas diverger.
 *  Même mécanisme que climatBilanColorExpression. */
export function climatPrecipDailyColorExpression(): unknown[] {
  const expr: unknown[] = ['step', ['get', 'value'], PRECIP_DAILY_COLORS[0]]
  PRECIP_DAILY_BOUNDS.forEach((b, i) => expr.push(b, PRECIP_DAILY_COLORS[i + 1]))
  return expr
}
```

Ajouter l'entrée dans `CLIMAT_VARIABLES` (après `tmean`) :

```ts
  precip_daily: {
    key: 'precip_daily', kind: 'daily',
    unit: 'mm', labelKey: 'climat.variables.precipDaily',
    stops: [],   // classes discrètes, pas de dégradé — cf. climatPrecipDailyColorExpression
  },
```

Renommer l'ordre des journalières et y ajouter la pluie :

```ts
/** Ordered for the picker's "Données journalières" section (Tx/Tn/Tmoy + pluie) —
 *  kept apart from CLIMAT_VARIABLE_ORDER so the monthly picker stays uncluttered. */
export const DAILY_VARIABLE_ORDER: ClimatVariable[] = ['tmax', 'tmin', 'tmean', 'precip_daily']
```

- [ ] **Step 4: Lancer les tests + le build**

Run: `npx vitest run src/lib/climat-colors.test.ts`
Expected: PASS

Run: `npm run build`
Expected: échecs `tsc` attendus ailleurs (`DAILY_TEMP_VARIABLE_ORDER` n'existe plus, `CLIMAT_VARIABLES` n'est plus exhaustif). **Corriger uniquement les renommages mécaniques** (`DAILY_TEMP_VARIABLE_ORDER` → `DAILY_VARIABLE_ORDER` dans `VariablePicker.tsx`). Si `tsc` réclame un traitement de `'precip_daily'` dans `ClimatMap`/`ClimatLegend`, c'est la Task 3 : mets le minimum pour compiler (`climatRawColorExpression` rend déjà un dégradé vide, acceptable transitoirement) et **signale-le dans ton rapport**.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/climat-colors.ts frontend/src/lib/climat-colors.test.ts \
        frontend/src/components/climat/VariablePicker.tsx
git commit -m "feat(climat): classes météo de la pluie journalière

Bornes 0,1/1/2/5/10/20/50 mm, fixes et absolues — pas de ré-ancrage, qui
ferait de la couleur un indice maison. Non linéaires parce qu'une rampe
linéaire 0-50 met 71 % du territoire dans ses 5 premiers % (mesuré).

DAILY_TEMP_VARIABLE_ORDER -> DAILY_VARIABLE_ORDER : la section n'est plus
seulement thermique."
```

---

### Task 3: Frontend — câbler la couche (API, hooks, carte, légende, plage)

Un seul déliverable : la couche s'affiche vraiment. La découper laisserait des états intermédiaires cassés (une carte sans légende, une légende sans données).

**Files:**
- Modify: `frontend/src/lib/observatory-api.ts`, `frontend/src/hooks/useClimat.ts`, `frontend/src/pages/ClimatPage.tsx`, `frontend/src/components/climat/ClimatMap.tsx`, `frontend/src/components/climat/ClimatLegend.tsx`
- Modify: `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`
- Test: `frontend/src/components/climat/ClimatLegend.test.tsx` (à créer s'il n'existe pas), `frontend/src/components/climat/VariablePicker.test.tsx`

**Interfaces:**
- Consumes (Task 1) : `GET /observatory/climat/daily-precip?date=YYYY-MM-DD` → `[{latitude, longitude, value}]` ; `GET /observatory/climat/daily-precip-range` → `{min_date, max_date}`.
- Consumes (Task 2) : `PRECIP_DAILY_BOUNDS`, `PRECIP_DAILY_COLORS`, `climatPrecipDailyColorExpression()`, `DAILY_VARIABLE_ORDER`, `ClimatVariable` incluant `'precip_daily'`.
- Produces : `useClimatDailyPrecip(day, enabled)`, `useClimatDailyPrecipRange(enabled?)` (hooks react-query, mêmes formes que leurs jumeaux température).

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter dans `frontend/src/components/climat/VariablePicker.test.tsx` :

```tsx
  it('propose la pluie parmi les journalières', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'Pluie (jour)' })).toBeInTheDocument()
  })
```

Créer `frontend/src/components/climat/ClimatLegend.test.tsx` (ou ajouter au fichier s'il existe) :

```tsx
import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { ClimatLegend } from './ClimatLegend'

describe('ClimatLegend — pluie journalière', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('affiche les bornes en mm, sans noms de classes inventés', () => {
    render(<ClimatLegend variable="precip_daily" window={1} month="2025-06-15" />)
    // La valeur EST le sens : on montre les bornes, pas un « faible/fort » éditorialisé.
    expect(screen.getByText(/0,1|0\.1/)).toBeInTheDocument()
    expect(screen.getByText(/≥\s*50/)).toBeInTheDocument()
    expect(screen.queryByText(/faible|modéré|fort/i)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Lancer les tests pour les voir échouer**

Run: `npx vitest run src/components/climat/VariablePicker.test.tsx src/components/climat/ClimatLegend.test.tsx`
Expected: FAIL — le bouton « Pluie (jour) » est introuvable ; la légende ne rend pas de bornes.

- [ ] **Step 3: Ajouter les libellés i18n**

`frontend/src/i18n/locales/fr.json` — dans `climat.variables` : `"precipDaily": "Pluie (jour)"`. Dans `climat.picker` : remplacer `"dailyTempLabel": "Températures journalières"` par `"dailyTempLabel": "Données journalières"`. Dans `climat.legend` : `"precipDryClass": "< 0,1"`.

`frontend/src/i18n/locales/en.json` — `"precipDaily": "Rainfall (day)"` ; `"dailyTempLabel": "Daily data"` ; `"precipDryClass": "< 0.1"`.

- [ ] **Step 4: Client API + hooks**

`frontend/src/lib/observatory-api.ts`, à côté de `dailyTemp` (~ligne 144) :

```ts
    dailyPrecip: (date: string) =>
      fetchJson<ClimatDailyTempPoint[]>('/observatory/climat/daily-precip', { date }),
    dailyPrecipRange: () => fetchJson<ClimatDailyTempRange>('/observatory/climat/daily-precip-range'),
```

`frontend/src/hooks/useClimat.ts`, à côté de leurs jumeaux température :

```ts
/** Bornes de dates de la couche pluie — endpoint distinct de la température :
 *  les couvertures divergent (mesuré : temp 2026-07-10, pluie 2026-07-12). */
export function useClimatDailyPrecipRange(enabled = true) {
  return useQuery({
    queryKey: ['climat', 'daily-precip-range'],
    queryFn: () => observatoryApi.climat.dailyPrecipRange(),
    enabled,
    staleTime: DAILY_TEMP_RANGE_STALE_TIME,
  })
}

/** Cumul de pluie journalier par maille (mm) pour une date. */
export function useClimatDailyPrecip(day: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ['climat', 'daily-precip', day],
    queryFn: () => observatoryApi.climat.dailyPrecip(day!),
    enabled: enabled && !!day,
    staleTime: CLIMAT_STALE_TIME,
  })
}
```

- [ ] **Step 5: Carte + légende**

`frontend/src/components/climat/ClimatMap.tsx` — importer `climatPrecipDailyColorExpression` et étendre le choix d'expression (~ligne 93) :

```tsx
      isIndex
        ? climatIndexColorExpression(variable as 'spi' | 'sti') as any
        : variable === 'bilan_hydrique'
          ? climatBilanColorExpression() as any
          : variable === 'precip_daily'
            ? climatPrecipDailyColorExpression() as any
            : climatRawColorExpression(variable) as any,
```

`frontend/src/components/climat/ClimatLegend.tsx` — importer `PRECIP_DAILY_BOUNDS`, `PRECIP_DAILY_COLORS`, et insérer cette branche **avant** le bloc dégradé final (après la branche `bilan_hydrique`) :

```tsx
  if (variable === 'precip_daily') {
    // Bornes en mm, sans noms de classes : contrairement au SPI (où « Très sec »
    // porte l'information, le z-score étant illisible), ici la valeur EST le sens.
    return (
      <div className="absolute bottom-4 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none" style={{ maxWidth: '190px' }}>
        <div className="text-xs font-semibold text-text-primary leading-tight">{t(cfg.labelKey)}</div>
        <div className="text-[10px] text-text-secondary mt-0.5">{periodLabel}</div>
        <div className="mt-1.5 space-y-0.5">
          {[...PRECIP_DAILY_COLORS].map((color, i) => {
            const label = i === 0
              ? t('climat.legend.precipDryClass')
              : i === PRECIP_DAILY_COLORS.length - 1
                ? `≥ ${PRECIP_DAILY_BOUNDS[i - 1]}`
                : `${PRECIP_DAILY_BOUNDS[i - 1]} – ${PRECIP_DAILY_BOUNDS[i]}`
            return (
              <div key={color} className="flex items-center gap-1.5">
                <span className="w-3 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="text-[9px] text-text-secondary">{label} {cfg.unit}</span>
              </div>
            )
          }).reverse()}
        </div>
      </div>
    )
  }
```

- [ ] **Step 6: ClimatPage — la plage suit la variable**

`frontend/src/pages/ClimatPage.tsx` : importer `useClimatDailyPrecip`, `useClimatDailyPrecipRange`. Remplacer le bloc `dailyRange` (~ligne 41) par :

```tsx
  // Les deux couches journalières ont des couvertures DIFFÉRENTES (mesuré :
  // température -> 2026-07-10, pluie -> 2026-07-12) : chacune sa plage, sinon le
  // DayStepper masquerait les jours de pluie les plus récents.
  const isPrecipDaily = s.variable === 'precip_daily'
  const { data: tempRange } = useClimatDailyTempRange()
  const { data: precipRange } = useClimatDailyPrecipRange()
  const dailyRange = isPrecipDaily ? precipRange : tempRange
  useEffect(() => {
    if (dailyRange?.max_date && !s.day) s.setDay(resolveDefaultDay(dailyRange.max_date))
  }, [dailyRange, s.day])
```

Puis brancher la source de points. Remplacer le bloc existant (`ClimatPage.tsx:61-63`) :

```tsx
  const { data: dailyPoints, isLoading: dailyLoading } = useClimatDailyTemp(
    s.day, dailyParam ?? 'tmax', s.isDaily && !!s.day,
  )
```

par :

```tsx
  // Les deux couches journalières lisent des tables différentes (mart température
  // vs stg_era5_timeseries) : deux hooks, dont un seul est activé à la fois.
  const { data: tempPoints, isLoading: tempLoading } = useClimatDailyTemp(
    s.day, dailyParam ?? 'tmax', s.isDaily && !isPrecipDaily && !!s.day,
  )
  const { data: precipPoints, isLoading: precipLoading } = useClimatDailyPrecip(
    s.day, s.isDaily && isPrecipDaily,
  )
  const dailyPoints = isPrecipDaily ? precipPoints : tempPoints
  const dailyLoading = isPrecipDaily ? precipLoading : tempLoading
```

`gridLoading` (ligne 66), `dailyPoints={dailyPoints}` (ligne 81) et le bandeau (ligne 86) consomment déjà ces deux noms : **ils n'ont pas à changer**.

Note : `dailyParam` vaut `undefined` pour `precip_daily` (l'entrée `CLIMAT_VARIABLES.precip_daily` n'en déclare pas — sa source diffère). Le `?? 'tmax'` existant absorbe ce cas, et le hook température est de toute façon désactivé (`!isPrecipDaily`).

- [ ] **Step 7: Lancer les tests + le build**

Run: `npx vitest run src/` — Expected: PASS
Run: `npm run build` — Expected: succès, aucune erreur TS.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/lib/observatory-api.ts frontend/src/hooks/useClimat.ts \
        frontend/src/pages/ClimatPage.tsx frontend/src/components/climat/ClimatMap.tsx \
        frontend/src/components/climat/ClimatLegend.tsx frontend/src/components/climat/ClimatLegend.test.tsx \
        frontend/src/components/climat/VariablePicker.test.tsx \
        frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): couche « Pluie (jour) » sur la carte

Classes discrètes en mm, légende sans noms inventés (la valeur est le sens).
Chaque couche journalière a sa propre plage de dates : les couvertures
divergent, réutiliser celle de la température masquerait les jours de pluie
les plus récents."
```

---

### Task 4: Bandeau pluie

**Files:**
- Rename: `frontend/src/lib/climat-daily-temp-format.ts` → `frontend/src/lib/climat-daily-format.ts` (+ son test)
- Rename: `frontend/src/components/climat/DailyTempBanner.tsx` → `frontend/src/components/climat/DailyBanner.tsx`
- Modify: `frontend/src/pages/ClimatPage.tsx`, `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

**Interfaces:**
- Consumes (Task 2) : `PRECIP_DAILY_BOUNDS` (sa dernière borne, 50, sert de seuil de comptage).
- Produces : `buildDailyBannerData(points, locale, countIf): DailyBannerData | null` avec `DailyBannerData = { maxValueLabel: string; countAboveThreshold: number }` ; `formatOneDecimal(value, locale): string` (ex-`formatTemperature`) ; composant `DailyBanner`.

- [ ] **Step 1: Écrire les tests qui échouent**

Dans le test du formateur (renommé `frontend/src/lib/climat-daily-format.test.ts`) :

```ts
import { buildDailyBannerData, formatOneDecimal } from './climat-daily-format'
import { PRECIP_DAILY_BOUNDS } from './climat-colors'

describe('buildDailyBannerData — pluie', () => {
  const TOP = PRECIP_DAILY_BOUNDS[PRECIP_DAILY_BOUNDS.length - 1]   // 50

  it('compte les mailles au-dessus de la borne HAUTE de la légende', () => {
    // Le seuil n'est pas inventé : c'est la dernière borne de la légende, donc
    // le chiffre se vérifie à l'œil sur celle-ci (même principe que le %
    // sécheresse recalé sur une frontière de classe).
    const points = [{ latitude: 1, longitude: 1, value: 98.8 }, { latitude: 1, longitude: 2, value: 50 },
                    { latitude: 1, longitude: 3, value: 49.9 }, { latitude: 1, longitude: 4, value: null }]
    const d = buildDailyBannerData(points as any, 'fr', (v) => v >= TOP)
    expect(d?.maxValueLabel).toBe('98,8')
    expect(d?.countAboveThreshold).toBe(2)   // 98.8 et 50 (>= inclusif, comme la classe « ≥ 50 »)
  })

  it('rend null quand aucune maille n’a de valeur', () => {
    const d = buildDailyBannerData([{ latitude: 1, longitude: 1, value: null }] as any, 'fr', () => true)
    expect(d).toBeNull()
  })
})
```

- [ ] **Step 2: Lancer le test pour le voir échouer**

Run: `npx vitest run src/lib/climat-daily-format.test.ts`
Expected: FAIL — le module `./climat-daily-format` n'existe pas.

- [ ] **Step 3: Généraliser le formateur**

`git mv frontend/src/lib/climat-daily-temp-format.ts frontend/src/lib/climat-daily-format.ts` (et son test). Puis :

- `formatTemperature` → `formatOneDecimal` : le corps ne fait que du formatage localisé à une décimale, rien de thermique — le nom mentait.
- `DailyTempBannerData` → `DailyBannerData` (champs inchangés).
- Remplacer `buildDailyTempBannerData(points, locale)` par :

```ts
/** Max + comptage au-dessus d'un seuil, pour le bandeau des couches journalières.
 *  Le prédicat est passé par l'appelant : la température compte `> 35 °C` (marqueur
 *  canicule), la pluie `>= 50 mm` (la borne haute de sa légende, donc vérifiable
 *  à l'œil dessus). Rend null quand aucune maille n'a de valeur, pour que
 *  l'appelant affiche « indisponible » au lieu d'un « 0 cellules » trompeur. */
export function buildDailyBannerData(
  points: ClimatDailyTempPoint[],
  locale: FormatLocale = 'fr',
  countIf: (v: number) => boolean = (v) => v > HEAT_CELL_THRESHOLD_C,
): DailyBannerData | null {
  const values = points.map((p) => p.value).filter((v): v is number => v != null)
  if (values.length === 0) return null
  return {
    maxValueLabel: formatOneDecimal(Math.max(...values), locale),
    countAboveThreshold: values.filter(countIf).length,
  }
}
```

Garder `HEAT_CELL_THRESHOLD_C = 35` (le défaut du prédicat).

- [ ] **Step 4: Généraliser le bandeau**

`git mv frontend/src/components/climat/DailyTempBanner.tsx frontend/src/components/climat/DailyBanner.tsx`. Renommer le composant `DailyTempBanner` → `DailyBanner`. Dans son corps, remplacer l'appel `buildDailyTempBannerData(points, locale)` par une branche :

```tsx
  const isPrecip = variable === 'precip_daily'
  const TOP_PRECIP = PRECIP_DAILY_BOUNDS[PRECIP_DAILY_BOUNDS.length - 1]
  const data = buildDailyBannerData(points, locale, isPrecip ? (v) => v >= TOP_PRECIP : undefined)
```

et le rendu par :

```tsx
        {t(isPrecip ? 'climat.banner.dailyPrecipSummary' : 'climat.banner.dailyTempSummary', {
          variable: t(CLIMAT_VARIABLES[variable].labelKey),
          max: data.maxValueLabel,
          count: data.countAboveThreshold,
        })}
```

Importer `PRECIP_DAILY_BOUNDS` depuis `@/lib/climat-colors`. Mettre à jour l'import et l'usage dans `ClimatPage.tsx` (`DailyTempBanner` → `DailyBanner`).

i18n — `fr.json`, dans `climat.banner` :

```json
"dailyPrecipSummary_one": "{{variable}} max France : {{max}} mm · {{count}} cellule ≥ 50 mm",
"dailyPrecipSummary_other": "{{variable}} max France : {{max}} mm · {{count}} cellules ≥ 50 mm"
```

`en.json` :

```json
"dailyPrecipSummary_one": "{{variable}} max France: {{max}} mm · {{count}} cell ≥ 50 mm",
"dailyPrecipSummary_other": "{{variable}} max France: {{max}} mm · {{count}} cells ≥ 50 mm"
```

- [ ] **Step 5: Lancer les tests + le build**

Run: `npx vitest run src/` — Expected: PASS
Run: `npm run build` — Expected: succès (tsc attrape tout `DailyTempBanner`/`buildDailyTempBannerData` oublié).

- [ ] **Step 6: Commit**

```bash
git add -A frontend/src
git commit -m "feat(climat): bandeau pluie (max + cellules ≥ 50 mm)

Le seuil 50 mm est la borne haute de la légende, pas un nombre inventé : le
compte se vérifie à l'œil sur celle-ci.

Le formateur et le bandeau perdent leur 'Temp' : ils servent désormais les
deux familles journalières. formatTemperature -> formatOneDecimal, son corps
n'ayant jamais rien eu de thermique."
```

---

### Task 5: Vérification de bout en bout

Les tests unitaires ne prouvent pas qu'une carte s'affiche.

**Files:** aucun (vérification).

**Interfaces:** Consumes le module Climat complet.

- [ ] **Step 1: Suites + build**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -q` puis, depuis `frontend/`, `npx vitest run src/ && npm run build`.
Expected: tout PASS.

- [ ] **Step 2: Déployer et exercer**

```bash
docker compose up -d --build backend frontend
until [ "$(docker inspect --format '{{.State.Health.Status}}' junon-backend)" = healthy ]; do sleep 10; done
```

Puis ouvrir `/climat` (frontend de prod, port 49513) et vérifier :
1. La section « **Données journalières** » propose Tx, Tn, T moy **et Pluie (jour)**.
2. Sélectionner « Pluie (jour) » → la carte se peint en classes bleues, pas en blanc uniforme.
3. La légende affiche les bornes en mm (`< 0,1` … `≥ 50`), **sans** noms de classes.
4. Le bandeau affiche « Pluie (jour) max France : X mm · N cellules ≥ 50 mm ».
5. Le DayStepper permet d'atteindre le **2026-07-12** en pluie (2 jours de plus qu'en température — c'est tout l'intérêt du range dédié).
6. Console navigateur **sans erreur**.

- [ ] **Step 3: Mesurer la latence perçue**

```bash
curl -s -o /dev/null -w "daily-precip -> HTTP %{http_code} en %{time_total}s\n" \
  "http://localhost:49514/api/v1/observatory/climat/daily-precip?date=2025-06-15"
```
Expected: < 2 s à froid, quasi instantané ensuite (cache `GRID_TTL`). **> 5 s = STOP**, l'exclusion de chunks est cassée.

- [ ] **Step 4: Commit (seulement si des correctifs ont été nécessaires)**

```bash
git add -A && git commit -m "fix(climat): correctifs issus de la vérification end-to-end"
```

Si rien n'a dû être corrigé, ne rien commiter.
