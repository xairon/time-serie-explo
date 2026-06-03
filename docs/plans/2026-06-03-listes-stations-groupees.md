# Listes de stations groupées — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réintroduire la liste des piézomètres de la même nappe/du même système (BDLISA) et harmoniser la liste des stations hydro du même site/cours d'eau, via un composant unique avec sélecteur de niveau.

**Architecture:** Deux endpoints `siblings` (piézo nouveau, hydro étendu) lus par un composant React unique `SiblingStationsPanel` affiché dans la fiche station et le tiroir. Liens opt-in, aucun filtrage de carte (règle « pas de filtrage implicite »).

**Tech Stack:** FastAPI + SQLAlchemy (sync) contre `gold.*` (brgm-postgres) ; React 19 + TanStack Query + i18next.

**Périmètre :** dépôt `time-serie-explo` uniquement. C'est la Partie B de la spec
`docs/specs/2026-06-03-ips-reference-et-groupement-stations.md`. La Partie A (IPS à
référence fixe, cross-repo) fera l'objet d'un plan séparé.

**Note sur les tests :** la suite existante teste les **helpers purs** (cf.
`tests/test_observatory_hydro_units.py`), pas les endpoints DB (brgm-postgres absent en
CI). On suit ce modèle : TDD sur les helpers, vérification **live via curl** contre le
backend prod (`http://localhost:49514`) pour les endpoints. Le front est vérifié par
`tsc --noEmit` + contrôle manuel.

---

## File Structure

**Backend**
- Modify `api/schemas/observatory.py` — ajoute `PiezoBdlisaSibling`, `PiezoBdlisaSiblings`.
- Modify `api/routers/observatory_piezo.py` — helpers BDLISA + endpoint `get_siblings`.
- Modify `api/routers/observatory_hydro.py` — param `level` sur `get_siblings`.
- Create `tests/test_observatory_siblings.py` — tests des helpers BDLISA.

**Frontend**
- Modify `frontend/src/lib/observatory-types.ts` — types piézo siblings.
- Modify `frontend/src/lib/observatory-api.ts` — `piezo.siblings`, `hydro.siblings(level)`.
- Modify `frontend/src/hooks/useObservatory.ts` — `usePiezoSiblings`, `useHydroSiblings(level)`.
- Create `frontend/src/components/observatory/SiblingStationsPanel.tsx` — composant unifié.
- Modify `frontend/src/pages/StationPage.tsx` — remplace le bloc hydro inline.
- Modify `frontend/src/components/observatory/StationDrawer.tsx` — remplace le bloc hydro inline.
- Modify `frontend/src/i18n/locales/fr.json` + `en.json` — clés du composant.

---

### Task 1 : Helpers BDLISA (pure, TDD)

**Files:**
- Modify: `api/routers/observatory_piezo.py` (ajout en tête, après les constantes)
- Test: `tests/test_observatory_siblings.py`

- [ ] **Step 1 : Écrire le test qui échoue**

Create `tests/test_observatory_siblings.py` :

```python
"""Unit tests for BDLISA grouping helpers in observatory_piezo router."""
from api.routers.observatory_piezo import _bdlisa_primary, _bdlisa_system_prefix


def test_bdlisa_primary_single_code():
    assert _bdlisa_primary("101AC01") == "101AC01"


def test_bdlisa_primary_takes_first_of_list():
    assert _bdlisa_primary("101AC01,123AK03") == "101AC01"
    assert _bdlisa_primary("101AC01, 123AK03") == "101AC01"


def test_bdlisa_primary_empty_is_none():
    assert _bdlisa_primary(None) is None
    assert _bdlisa_primary("") is None
    assert _bdlisa_primary("   ") is None


def test_bdlisa_system_prefix_strips_entity_suffix():
    assert _bdlisa_system_prefix("101AC01") == "101AC"
    assert _bdlisa_system_prefix("121BD01") == "121BD"
    assert _bdlisa_system_prefix("139AM15") == "139AM"


def test_bdlisa_system_prefix_keeps_bare_system():
    assert _bdlisa_system_prefix("101AC") == "101AC"


def test_bdlisa_system_prefix_none_and_fallback():
    assert _bdlisa_system_prefix(None) is None
    assert _bdlisa_system_prefix("") is None
    # No regex match → return the primary code unchanged
    assert _bdlisa_system_prefix("WEIRD") == "WEIRD"
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `docker compose exec backend pytest tests/test_observatory_siblings.py -v`
Expected: FAIL — `ImportError: cannot import name '_bdlisa_primary'`.

- [ ] **Step 3 : Implémenter les helpers**

Dans `api/routers/observatory_piezo.py`, ajouter `import re` en tête (avec les autres
imports stdlib, après `from datetime import date`), et après la constante `SPLI_TTL = 86400`
(ligne ~31) insérer :

```python
SIBLINGS_TTL = 3600

_BDLISA_SYSTEM_RE = re.compile(r"^(\d{3}[A-Z]{2})")


def _bdlisa_primary(codes_bdlisa: str | None) -> str | None:
    """Return the primary (first) BDLISA entity code from a possibly comma-joined string."""
    if not codes_bdlisa:
        return None
    first = codes_bdlisa.split(",")[0].strip()
    return first or None


def _bdlisa_system_prefix(codes_bdlisa: str | None) -> str | None:
    """Return the BDLISA system-level prefix (3 digits + 2 letters) of the primary code.

    '101AC01' -> '101AC' ; '101AC' -> '101AC' ; None/'' -> None.
    Falls back to the primary code unchanged if it doesn't match the BDLISA shape.
    """
    primary = _bdlisa_primary(codes_bdlisa)
    if not primary:
        return None
    m = _BDLISA_SYSTEM_RE.match(primary)
    return m.group(1) if m else primary
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `docker compose exec backend pytest tests/test_observatory_siblings.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5 : Commit**

```bash
git add tests/test_observatory_siblings.py api/routers/observatory_piezo.py
git commit -m "feat(observatory): BDLISA grouping helpers (nappe/système)"
```

---

### Task 2 : Schéma + endpoint siblings piézo

**Files:**
- Modify: `api/schemas/observatory.py` (après `HydroSiteSiblings`, ligne ~233)
- Modify: `api/routers/observatory_piezo.py` (AVANT la route `get_station`, ligne ~368)

- [ ] **Step 1 : Ajouter les schémas Pydantic**

Dans `api/schemas/observatory.py`, après la classe `HydroSiteSiblings` (ligne ~232) :

```python
class PiezoBdlisaSibling(BaseModel):
    code_bss: str
    nom_commune: str | None = None
    codes_bdlisa: str | None = None
    classification: str | None = None


class PiezoBdlisaSiblings(BaseModel):
    level: str
    code_bdlisa: str | None = None
    non_rattachee: bool = False
    nb_stations: int
    siblings: list[PiezoBdlisaSibling]
```

- [ ] **Step 2 : Importer le schéma dans le routeur**

Dans `api/routers/observatory_piezo.py`, ajouter `PiezoBdlisaSiblings` à l'import
`from api.schemas.observatory import (...)` (ligne ~11-19) :

```python
from api.schemas.observatory import (
    PiezoBdlisaSiblings,
    PiezoDaily,
    PiezoMonthly,
    PiezoPercentiles,
    PiezoSPI,
    PiezoSPLI,
    PiezoStation,
    PiezoYearly,
)
```

- [ ] **Step 3 : Ajouter l'endpoint AVANT `get_station`**

> ⚠️ La route `@router.get("/stations/{code_bss:path}")` (`get_station`, ligne ~368) est
> un catch-all `:path` : elle capture aussi `.../siblings`. L'endpoint siblings DOIT être
> déclaré **avant** elle (comme le sont déjà `/spli` et `/spi`).

Juste avant la ligne `@router.get("/stations/{code_bss:path}", response_model=PiezoStation)` :

```python
@router.get("/stations/{code_bss:path}/siblings", response_model=PiezoBdlisaSiblings)
def get_siblings(code_bss: str, level: str = Query("nappe", pattern="^(nappe|systeme)$")):
    """Other piezometers in the same BDLISA entity (nappe) or system (préfixe)."""

    def fetch():
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT codes_bdlisa FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                    {"code": code_bss},
                ).mappings().first()
                if row is None:
                    raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")

                codes_bdlisa = row["codes_bdlisa"]
                primary = _bdlisa_primary(codes_bdlisa)
                if not primary:
                    return {
                        "level": level,
                        "code_bdlisa": None,
                        "non_rattachee": True,
                        "nb_stations": 1,
                        "siblings": [],
                    }

                if level == "systeme":
                    match = _bdlisa_system_prefix(codes_bdlisa)
                    where = "s.codes_bdlisa LIKE :pat AND s.code_bss != :code"
                    params = {"pat": f"{match}%", "code": code_bss}
                else:
                    match = primary
                    where = "s.codes_bdlisa = :pat AND s.code_bss != :code"
                    params = {"pat": match, "code": code_bss}

                query = f"""
                    SELECT s.code_bss, s.nom_commune, s.codes_bdlisa,
                           sci.index_class AS classification
                    FROM gold.dim_piezo_stations s
                    LEFT JOIN gold.station_current_index sci
                      ON sci.type = 'piezo' AND sci.code = s.code_bss
                    WHERE {where}
                    ORDER BY s.code_bss
                    LIMIT 50
                """
                result = conn.execute(text(query), params)
                siblings = [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

        return {
            "level": level,
            "code_bdlisa": match,
            "non_rattachee": False,
            "nb_stations": len(siblings) + 1,
            "siblings": [
                {
                    "code_bss": s["code_bss"],
                    "nom_commune": s.get("nom_commune"),
                    "codes_bdlisa": s.get("codes_bdlisa"),
                    "classification": s.get("classification"),
                }
                for s in siblings
            ],
        }

    return get_cached(
        "obs_piezo_siblings",
        {"code_bss": code_bss, "level": level},
        SIBLINGS_TTL,
        fetch,
    )
```

- [ ] **Step 4 : Recharger le backend et vérifier en live**

Run: `docker compose up -d --build backend` puis :
```bash
# nappe : un piézo avec BDLISA partagé (ex. 119AA01)
curl -s "http://localhost:49514/api/v1/observatory/piezo/stations/00073X0043%2FF1/siblings?level=nappe" | python3 -m json.tool | head -20
# système : préfixe élargi
curl -s "http://localhost:49514/api/v1/observatory/piezo/stations/00073X0043%2FF1/siblings?level=systeme" | python3 -c "import sys,json;d=json.load(sys.stdin);print('level',d['level'],'code',d['code_bdlisa'],'nb',d['nb_stations'])"
# non rattachée (BDLISA None)
curl -s "http://localhost:49514/api/v1/observatory/piezo/stations/00027X0049%2FPZ3BIS/siblings?level=nappe" | python3 -c "import sys,json;d=json.load(sys.stdin);print('non_rattachee',d['non_rattachee'])"
```
Expected : `nappe` renvoie une liste ≥1 ; `systeme` un `code_bdlisa` plus court (préfixe) et `nb_stations` ≥ celui de `nappe` ; la station sans BDLISA → `non_rattachee: true`.

- [ ] **Step 5 : Commit**

```bash
git add api/schemas/observatory.py api/routers/observatory_piezo.py
git commit -m "feat(observatory): piezo siblings endpoint (nappe/système BDLISA)"
```

---

### Task 3 : Param `level` sur l'endpoint siblings hydro

**Files:**
- Modify: `api/routers/observatory_hydro.py` (fonction `get_siblings`, lignes ~453-500)

- [ ] **Step 1 : Remplacer la fonction `get_siblings`**

Remplacer intégralement le corps de `@router.get("/stations/{code_station}/siblings", ...)`
(lignes ~453-500) par :

```python
@router.get("/stations/{code_station}/siblings", response_model=HydroSiteSiblings)
def get_siblings(code_station: str, level: str = Query("site", pattern="^(site|cours_eau)$")):
    """Other hydro stations at the same hydrometric site or on the same watercourse."""

    def fetch():
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT code_site, libelle_site, code_cours_eau, nom_cours_eau "
                        "FROM gold.dim_hydro_stations WHERE code_station = :code"
                    ),
                    {"code": code_station},
                ).mappings().first()
                if not row:
                    raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")

                if level == "cours_eau":
                    group_val = row["code_cours_eau"]
                    if not group_val:
                        raise HTTPException(404, f"Aucun cours d'eau pour la station {code_station}")
                    where = "code_cours_eau = :grp AND code_station != :code"
                else:
                    group_val = row["code_site"]
                    if not group_val:
                        raise HTTPException(404, f"Aucun code de site pour la station {code_station}")
                    where = "code_site = :grp AND code_station != :code"

                query = f"""
                    SELECT code_station, libelle_station, grandeur_hydro_principale,
                           classification_resultat_dern_annee, derniere_mesure
                    FROM gold.dim_hydro_stations
                    WHERE {where}
                    ORDER BY code_station
                    LIMIT 50
                """
                result = conn.execute(text(query), {"grp": group_val, "code": code_station})
                siblings = [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

        return {
            "code_site": row["code_site"],
            "libelle_site": row["libelle_site"],
            "nom_cours_eau": row["nom_cours_eau"],
            "nb_stations": len(siblings) + 1,
            "siblings": [
                {
                    "code_station": s["code_station"],
                    "libelle_station": s.get("libelle_station"),
                    "grandeur_hydro_principale": s.get("grandeur_hydro_principale"),
                    "classification": s.get("classification_resultat_dern_annee"),
                    "derniere_mesure": s.get("derniere_mesure"),
                }
                for s in siblings
            ],
        }

    return get_cached(
        "obs_hydro_siblings",
        {"code_station": code_station, "level": level},
        SIBLINGS_TTL,
        fetch,
    )
```

- [ ] **Step 2 : Recharger le backend et vérifier en live**

Run: `docker compose up -d --build backend` puis :
```bash
# site (comportement historique) — site multi-stations connu : 12320001
curl -s "http://localhost:49514/api/v1/observatory/hydro/stations/1232000101/siblings?level=site" | python3 -c "import sys,json;d=json.load(sys.stdin);print('site nb',d['nb_stations'])"
# cours d'eau (élargi)
curl -s "http://localhost:49514/api/v1/observatory/hydro/stations/1232000101/siblings?level=cours_eau" | python3 -c "import sys,json;d=json.load(sys.stdin);print('cours_eau nb',d['nb_stations'])"
```
Expected : `site` → `nb_stations` = 2 (inchangé) ; `cours_eau` → `nb_stations` ≥ 2.

- [ ] **Step 3 : Commit**

```bash
git add api/routers/observatory_hydro.py
git commit -m "feat(observatory): hydro siblings level (site/cours d'eau)"
```

---

### Task 4 : Types + client API + hooks (frontend)

**Files:**
- Modify: `frontend/src/lib/observatory-types.ts` (après `HydroSiteSiblings`, ligne ~114)
- Modify: `frontend/src/lib/observatory-api.ts` (objets `piezo` et `hydro`, + import)
- Modify: `frontend/src/hooks/useObservatory.ts` (remplace `useHydroSiblings`, ligne ~52)

- [ ] **Step 1 : Ajouter les types piézo**

Dans `frontend/src/lib/observatory-types.ts`, après `HydroSiteSiblings` (ligne ~114) :

```typescript
export interface PiezoBdlisaSibling {
  code_bss: string
  nom_commune: string | null
  codes_bdlisa: string | null
  classification: string | null
}

export interface PiezoBdlisaSiblings {
  level: string
  code_bdlisa: string | null
  non_rattachee: boolean
  nb_stations: number
  siblings: PiezoBdlisaSibling[]
}
```

- [ ] **Step 2 : Étendre le client API**

Dans `frontend/src/lib/observatory-api.ts` :

Ajouter `PiezoBdlisaSiblings` à l'import de types (ligne ~12, à côté de `HydroSiteSiblings`) :
```typescript
  HydroSiteSiblings,
  PiezoBdlisaSiblings,
```

Dans l'objet `piezo` (après la ligne `spi:` ~58), ajouter :
```typescript
    siblings: (code: string, level: 'nappe' | 'systeme' = 'nappe') =>
      fetchJson<PiezoBdlisaSiblings>(`/observatory/piezo/stations/${encodeURIComponent(code)}/siblings`, { level }),
```

Remplacer la ligne `hydro.siblings` existante (ligne ~72) par :
```typescript
    siblings: (code: string, level: 'site' | 'cours_eau' = 'site') =>
      fetchJson<HydroSiteSiblings>(`/observatory/hydro/stations/${encodeURIComponent(code)}/siblings`, { level }),
```

- [ ] **Step 3 : Mettre à jour les hooks**

Dans `frontend/src/hooks/useObservatory.ts`, remplacer la fonction `useHydroSiblings`
(lignes ~52-60) par :

```typescript
export function usePiezoSiblings(code: string, level: 'nappe' | 'systeme') {
  return useQuery({
    queryKey: ['obs-siblings', 'piezo', code, level],
    queryFn: () => observatoryApi.piezo.siblings(code, level),
    enabled: !!code,
    staleTime: 3_600_000,
    retry: false,
  })
}

export function useHydroSiblings(code: string, level: 'site' | 'cours_eau' = 'site') {
  return useQuery({
    queryKey: ['obs-siblings', 'hydro', code, level],
    queryFn: () => observatoryApi.hydro.siblings(code, level),
    enabled: !!code,
    staleTime: 3_600_000,
    retry: false,
  })
}
```

- [ ] **Step 4 : Type-check**

Run: `docker compose run --rm frontend npx tsc --noEmit`
Expected : peut afficher des erreurs dans `StationPage.tsx`/`StationDrawer.tsx` car
`useHydroSiblings` a changé de signature — **normal**, corrigé en Task 6. Vérifier qu'il
n'y a **aucune** erreur dans `observatory-types.ts`, `observatory-api.ts`,
`useObservatory.ts`.

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/lib/observatory-types.ts frontend/src/lib/observatory-api.ts frontend/src/hooks/useObservatory.ts
git commit -m "feat(observatory): piezo siblings types/hook + hydro level param"
```

---

### Task 5 : Composant `SiblingStationsPanel`

**Files:**
- Create: `frontend/src/components/observatory/SiblingStationsPanel.tsx`

Le composant gère piézo **et** hydro, le sélecteur de niveau, et deux variantes
d'affichage (`page` = section pleine, `drawer` = compact). Règle d'affichage :
- piézo non rattachée → message, pas de toggle ;
- sinon : toggle + liste, ou « aucune autre station » si vide au niveau courant ;
- hydro sans données → rien.

- [ ] **Step 1 : Créer le composant**

```tsx
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Waves } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePiezoSiblings, useHydroSiblings } from '@/hooks/useObservatory'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'

type Props = {
  code: string
  type: 'piezo' | 'hydro'
  variant?: 'page' | 'drawer'
}

type Row = { to: string; title: string; subtitle?: string; classification: string | null }

export function SiblingStationsPanel({ code, type, variant = 'page' }: Props) {
  const { t } = useTranslation()
  const isPiezo = type === 'piezo'
  const [piezoLevel, setPiezoLevel] = useState<'nappe' | 'systeme'>('nappe')
  const [hydroLevel, setHydroLevel] = useState<'site' | 'cours_eau'>('site')

  const piezo = usePiezoSiblings(isPiezo ? code : '', piezoLevel)
  const hydro = useHydroSiblings(!isPiezo ? code : '', hydroLevel)

  const levels: { value: string; label: string }[] = isPiezo
    ? [
        { value: 'nappe', label: t('observatory.siblings.piezo.nappe') },
        { value: 'systeme', label: t('observatory.siblings.piezo.systeme') },
      ]
    : [
        { value: 'site', label: t('observatory.siblings.hydro.site') },
        { value: 'cours_eau', label: t('observatory.siblings.hydro.coursEau') },
      ]
  const activeLevel = isPiezo ? piezoLevel : hydroLevel
  const setLevel = (v: string) =>
    isPiezo ? setPiezoLevel(v as 'nappe' | 'systeme') : setHydroLevel(v as 'site' | 'cours_eau')

  // Build a uniform shape from the two payloads
  const data = isPiezo ? piezo.data : hydro.data
  if (!data) return null

  const nonRattachee = isPiezo && (piezo.data?.non_rattachee ?? false)
  const subtitle = isPiezo
    ? piezo.data?.code_bdlisa ?? ''
    : `${hydro.data?.libelle_site || hydro.data?.code_site || ''}${hydro.data?.nom_cours_eau ? ` - ${hydro.data.nom_cours_eau}` : ''}`

  const rows: Row[] = isPiezo
    ? (piezo.data?.siblings ?? []).map(s => ({
        to: `/station/piezo/${encodeURIComponent(s.code_bss)}`,
        title: s.nom_commune || s.code_bss,
        subtitle: s.code_bss,
        classification: s.classification,
      }))
    : (hydro.data?.siblings ?? []).map(s => ({
        to: `/station/hydro/${s.code_station}`,
        title: s.libelle_station || s.code_station,
        subtitle: s.code_station,
        classification: s.classification,
      }))

  const isDrawer = variant === 'drawer'
  const Toggle = !nonRattachee && (
    <div className="flex gap-1">
      {levels.map(l => (
        <button
          key={l.value}
          onClick={() => setLevel(l.value)}
          className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
            activeLevel === l.value
              ? 'bg-accent-cyan/20 text-accent-cyan'
              : 'text-text-secondary hover:bg-bg-hover'
          }`}
        >
          {l.label}
        </button>
      ))}
    </div>
  )

  const header = (
    <div className="flex items-center justify-between gap-2 mb-1">
      <span className="flex items-center gap-2 text-sm font-semibold text-gray-300">
        <Waves className="w-4 h-4" />
        {t('observatory.siblings.title')}
      </span>
      {Toggle}
    </div>
  )

  const body = nonRattachee ? (
    <p className="text-xs text-text-secondary">{t('observatory.siblings.notLinked')}</p>
  ) : rows.length === 0 ? (
    <p className="text-xs text-text-secondary">{t('observatory.siblings.empty')}</p>
  ) : (
    <div className={`space-y-1 ${isDrawer ? 'max-h-32 overflow-y-auto' : ''}`}>
      {(isDrawer ? rows.slice(0, 5) : rows).map(r => (
        <Link
          key={r.to}
          to={r.to}
          className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-bg-hover transition-colors"
        >
          <span className="text-xs text-gray-200 truncate">{r.title}</span>
          <span className="flex items-center gap-2 shrink-0 ml-2">
            {!isDrawer && r.subtitle && (
              <span className="text-[10px] text-gray-500">{r.subtitle}</span>
            )}
            {r.classification && (
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: CLASSIFICATION_COLORS[r.classification] ?? '#6b7280' }}
                title={r.classification}
              />
            )}
          </span>
        </Link>
      ))}
    </div>
  )

  if (isDrawer) {
    return (
      <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
        {header}
        {subtitle && !nonRattachee && (
          <p className="text-xs text-text-secondary mb-2">{subtitle}</p>
        )}
        {body}
      </div>
    )
  }

  return (
    <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
      {header}
      {subtitle && !nonRattachee && <p className="text-xs text-gray-500 mb-3">{subtitle}</p>}
      {body}
    </section>
  )
}
```

- [ ] **Step 2 : Type-check (isolé)**

Run: `docker compose run --rm frontend npx tsc --noEmit`
Expected : pas d'erreur dans `SiblingStationsPanel.tsx` (les clés i18n manquantes ne sont
pas des erreurs TS). Erreurs résiduelles tolérées seulement dans `StationPage.tsx` /
`StationDrawer.tsx` (Task 6).

- [ ] **Step 3 : Commit**

```bash
git add frontend/src/components/observatory/SiblingStationsPanel.tsx
git commit -m "feat(observatory): unified SiblingStationsPanel (piezo + hydro, level toggle)"
```

---

### Task 6 : Brancher le composant dans StationPage et StationDrawer

**Files:**
- Modify: `frontend/src/pages/StationPage.tsx` (import ligne ~5-12 ; bloc lignes ~120-126 ; hook ligne ~76)
- Modify: `frontend/src/components/observatory/StationDrawer.tsx` (import ligne ~8 ; bloc lignes ~111-126 ; hook ligne ~46)

- [ ] **Step 1 : StationPage — importer le composant et retirer le hook hydro local**

Dans `frontend/src/pages/StationPage.tsx` :
- Ajouter l'import après la ligne 12 :
```typescript
import { SiblingStationsPanel } from '@/components/observatory/SiblingStationsPanel'
```
- Retirer `useHydroSiblings` de l'import de hooks ligne 5 (devient) :
```typescript
import { usePiezoStationDetail, useHydroStationDetail, usePiezoMonthly, useHydroMonthly, usePiezoDaily, useHydroDaily, usePiezoYearly, useHydroYearly, usePiezoSPLI, useHydroSSFI, useSPI } from '@/hooks/useObservatory'
```
- Supprimer la ligne 76 :
```typescript
  const { data: hydroSiblings } = useHydroSiblings(!isPiezo ? code : '')
```

- [ ] **Step 2 : StationPage — remplacer le bloc hydro inline**

Remplacer le bloc lignes ~120-126 (`{!isPiezo && hydroSiblings && ...}` … `</section>)}`) par :

```tsx
        <SiblingStationsPanel code={code} type={type} variant="page" />
```

- [ ] **Step 3 : StationDrawer — importer et retirer le hook local**

Dans `frontend/src/components/observatory/StationDrawer.tsx` :
- Remplacer l'import ligne 8 pour retirer `useHydroSiblings` :
```typescript
import { usePiezoStationDetail, useHydroStationDetail, useObsPastasSummary } from '@/hooks/useObservatory'
```
- Ajouter l'import du composant (à côté des autres imports de composants) :
```typescript
import { SiblingStationsPanel } from '@/components/observatory/SiblingStationsPanel'
```
- Supprimer la ligne 46 :
```typescript
  const hydroSiblings = useHydroSiblings(!isPiezo ? stationCode : '')
```
> Vérifier le nom exact de la variable code dans ce fichier (`stationCode` / `sCode`) ; la
> remplacer telle quelle dans l'étape suivante.

- [ ] **Step 4 : StationDrawer — remplacer le bloc hydro inline**

Remplacer le bloc lignes ~111-126 (`{!isPiezo && hydroSiblings.data && ...}` … fermeture)
par :

```tsx
        <SiblingStationsPanel code={sCode} type={type} variant="drawer" />
```
> `sCode` et `type` sont les variables déjà présentes dans `StationDrawer` (cf. ligne 128
> `to={`/station/${type}/${sCode}`}`). Si `type` n'existe pas comme variable, utiliser
> `{isPiezo ? 'piezo' : 'hydro'}`.

- [ ] **Step 5 : Vérifier que `CLASSIFICATION_COLORS`/`Waves` ne sont plus orphelins**

Si `StationDrawer.tsx` ou `StationPage.tsx` n'utilisent plus `CLASSIFICATION_COLORS`,
`Waves`, ou `Link` après suppression du bloc, retirer l'import inutilisé pour éviter
l'erreur TS `noUnusedLocals`. (Dans `StationPage`, `Waves`/`Link`/`CLASSIFICATION_COLORS`
restent utilisés ailleurs — vérifier par recherche avant de retirer.)

Run: `grep -n "CLASSIFICATION_COLORS\|<Waves\|<Link" frontend/src/pages/StationPage.tsx frontend/src/components/observatory/StationDrawer.tsx`

- [ ] **Step 6 : Type-check complet**

Run: `docker compose run --rm frontend npx tsc --noEmit`
Expected : **0 erreur**.

- [ ] **Step 7 : Commit**

```bash
git add frontend/src/pages/StationPage.tsx frontend/src/components/observatory/StationDrawer.tsx
git commit -m "feat(observatory): use SiblingStationsPanel in station page + drawer"
```

---

### Task 7 : Clés i18n

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json` (objet `observatory`, vers ligne ~114)
- Modify: `frontend/src/i18n/locales/en.json` (objet `observatory`, vers ligne ~114)

- [ ] **Step 1 : Ajouter le bloc `siblings` (FR)**

Dans `frontend/src/i18n/locales/fr.json`, dans l'objet `observatory` (après la clé
`"hydroSite"` ligne ~114), ajouter :

```json
    "siblings": {
      "title": "Stations du même groupe",
      "empty": "Aucune autre station à ce niveau",
      "notLinked": "Station non rattachée à une entité BDLISA",
      "piezo": { "nappe": "Nappe", "systeme": "Système" },
      "hydro": { "site": "Site", "coursEau": "Cours d'eau" }
    },
```

- [ ] **Step 2 : Ajouter le bloc `siblings` (EN)**

Dans `frontend/src/i18n/locales/en.json`, même emplacement :

```json
    "siblings": {
      "title": "Stations in the same group",
      "empty": "No other station at this level",
      "notLinked": "Station not linked to a BDLISA entity",
      "piezo": { "nappe": "Aquifer", "systeme": "System" },
      "hydro": { "site": "Site", "coursEau": "Watercourse" }
    },
```

- [ ] **Step 3 : Valider le JSON**

Run: `python3 -c "import json; json.load(open('frontend/src/i18n/locales/fr.json')); json.load(open('frontend/src/i18n/locales/en.json')); print('OK')"`
Expected : `OK`.

- [ ] **Step 4 : Commit**

```bash
git add frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "i18n(observatory): siblings panel labels (fr/en)"
```

---

### Task 8 : Vérification finale + build front

**Files:** aucun (vérification)

- [ ] **Step 1 : Rebuild front + back**

Run: `docker compose up -d --build frontend backend`
Expected : conteneurs `Up (healthy)`.

- [ ] **Step 2 : Matrice de vérification API**

```bash
curl -s "http://localhost:49514/api/v1/observatory/piezo/stations/00073X0043%2FF1/siblings?level=nappe"   | python3 -c "import sys,json;d=json.load(sys.stdin);print('piezo nappe   ', d['nb_stations'], d['code_bdlisa'])"
curl -s "http://localhost:49514/api/v1/observatory/piezo/stations/00073X0043%2FF1/siblings?level=systeme" | python3 -c "import sys,json;d=json.load(sys.stdin);print('piezo systeme ', d['nb_stations'], d['code_bdlisa'])"
curl -s "http://localhost:49514/api/v1/observatory/hydro/stations/1232000101/siblings?level=site"         | python3 -c "import sys,json;d=json.load(sys.stdin);print('hydro site    ', d['nb_stations'])"
curl -s "http://localhost:49514/api/v1/observatory/hydro/stations/1232000101/siblings?level=cours_eau"    | python3 -c "import sys,json;d=json.load(sys.stdin);print('hydro cours   ', d['nb_stations'])"
```
Expected : 4 lignes cohérentes ; `systeme` ≥ `nappe`, `cours_eau` ≥ `site`.

- [ ] **Step 3 : Vérification UI manuelle**

Ouvrir `http://localhost:49513/station/piezo/00073X0043%2FF1` : section « Stations du même
groupe » avec toggle Nappe/Système, liens cliquables, pastilles de classe. Basculer le
toggle → la liste change. Ouvrir une station hydro multi-stations → toggle Site/Cours d'eau.
Ouvrir une station piézo sans BDLISA → message « non rattachée ». Cliquer une voisine →
navigation, **la carte/le reste de l'UI ne sont pas filtrés**.

- [ ] **Step 4 : Lancer la suite de tests backend**

Run: `docker compose exec backend pytest tests/test_observatory_siblings.py -v`
Expected : PASS.

- [ ] **Step 5 : Commit final (si ajustements)**

```bash
git add -A && git commit -m "chore(observatory): finalize grouped station lists"
```

---

## Self-Review

- **Couverture spec (Partie B)** : endpoint piézo nappe/système → Task 2 ; hydro site/cours
  d'eau → Task 3 ; composant unique + toggle + liens opt-in + masquage/non-rattachée →
  Task 5/6 ; i18n → Task 7. ✓
- **Écart assumé vs spec** : la spec disait « panneau masqué si 0 voisine » ; avec le toggle,
  on garde le panneau visible (avec message « aucune autre station ») quand la station a une
  identité de groupe, sinon le toggle serait inutilisable. Masquage total seulement si pas de
  rendu de données. Justifié par l'ajout du sélecteur.
- **Placeholders** : aucun — code complet à chaque étape.
- **Cohérence des types** : `usePiezoSiblings(code, level)`, `useHydroSiblings(code, level)`,
  `PiezoBdlisaSiblings`/`HydroSiteSiblings`, helpers `_bdlisa_primary`/`_bdlisa_system_prefix`
  utilisés de façon identique entre tâches. ✓
- **Risque route catch-all** : siblings piézo déclaré avant `get_station` (`:path`) — noté
  Task 2 Step 3. ✓
