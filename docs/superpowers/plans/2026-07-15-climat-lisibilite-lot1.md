# Climat — Lot 1 (langage visuel & lisibilité) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre le module Climat lisible pour des gestionnaires eau : une seule grammaire couleur (« rouge = préoccupant », Normal neutre unique), classes nommées partout, hiérarchie claire, sans changement d'entrepôt.

**Architecture:** Frontend React/TS (Vitest + Testing Library) + micro-ajustement backend Python (pytest). Classification statistique z→classe = **backend seul** (`api/era5_anomaly.py`, source unique déjà consommée via `index_class`). Présentation (couleur/label) = **frontend seul** (`era5-colors.ts` → à repeindre). Le binning physique du bilan (mm) est une décision de présentation → frontend.

**Tech Stack:** React 18, TypeScript, MapLibre GL, i18next, Vitest, Testing Library ; FastAPI, pytest.

## Global Constraints

- Spec de référence : `docs/superpowers/specs/2026-07-15-climat-module-lisibilite-lot1-design.md`.
- **Aucune touche entrepôt** (marts gold inchangés). Backend = lecture seule.
- **Classification statistique = backend seul** : le front ne réimplémente jamais les seuils z→classe. Il consomme `index_class` de l'API et mappe classe→couleur/label.
- **Palette d'anomalie = RdBu ColorBrewer (colorblind-safe)** : `#b2182b #ef8a62 #fddbc7 #f7f7f7 #d1e5f0 #67a9cf #2166ac`. Neutre `NORMAL = #f7f7f7` identique sur tous les indicateurs d'anomalie.
- **Direction couleur** : sec / déficit / chaud = **rouge** ; humide / surplus / froid = **bleu**.
- **Seuil sécheresse = −0.84 σ** (frontière de la classe « Modérément sec »), plus jamais `−1.0`.
- Classes McKee (clés) : `EXTREMEMENT_BAS TRES_BAS BAS NORMAL HAUT TRES_HAUT EXTREMEMENT_HAUT` (+ `UNKNOWN`). Labels FR déjà en i18n `observatory.spi.*` / `observatory.sti.*`.
- Tests front : `vitest run` ; style = `i18n.changeLanguage('fr')` en `beforeAll`, requêtes par rôle ARIA. Tests back : `pytest` (`testpaths=["tests"]`).
- Commits fréquents, un par tâche minimum.

---

### Task 1 : Backend — centraliser & recaler le seuil sécheresse (−1.0 → −0.84)

**Files:**
- Modify: `api/era5_anomaly.py` (ajouter la constante)
- Modify: `api/routers/observatory_climat.py` (lignes ~127 et ~286 : remplacer `-1.0`)
- Test: `tests/test_situation_router.py` (ou nouveau `tests/test_drought_threshold.py`)

**Interfaces:**
- Produces: `api.era5_anomaly.DROUGHT_SPI_THRESHOLD: float = -0.84`
- Consumes: `classify_index` (existant, inchangé)

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# tests/test_drought_threshold.py
from api.era5_anomaly import DROUGHT_SPI_THRESHOLD
from api.routers.observatory_climat import _build_situation_summary
from datetime import date

def test_drought_threshold_is_moderate_dry_boundary():
    assert DROUGHT_SPI_THRESHOLD == -0.84

def test_pct_secheresse_counts_below_moderate_dry():
    # 4 cellules : -1.0 (sec), -0.9 (sec), -0.5 (normal), 0.2 (normal)
    rows = [
        {"spi": -1.0, "era5_latitude": 48.0, "era5_longitude": 2.0},
        {"spi": -0.9, "era5_latitude": 48.1, "era5_longitude": 2.0},
        {"spi": -0.5, "era5_latitude": 48.2, "era5_longitude": 2.0},
        {"spi":  0.2, "era5_latitude": 48.3, "era5_longitude": 2.0},
    ]
    out = _build_situation_summary(rows, [], date(2026, 6, 1), 3)
    # -1.0 et -0.9 sont < -0.84 → 2/4 = 50 %
    assert out["pct_secheresse"] == 50.0
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `pytest tests/test_drought_threshold.py -v`
Expected: FAIL (`ImportError: cannot import name 'DROUGHT_SPI_THRESHOLD'`)

- [ ] **Step 3 : Implémenter**

Dans `api/era5_anomaly.py`, après les seuils :
```python
# Frontière de la classe « Modérément sec » (BAS) : un mois est « en sécheresse »
# dès que son SPI passe sous ce seuil. Aligné sur la légende McKee (pas de -1.0 arbitraire).
DROUGHT_SPI_THRESHOLD = -0.84
```
Dans `api/routers/observatory_climat.py`, importer et remplacer les deux `-1.0` :
```python
from api.era5_anomaly import classify_index, DROUGHT_SPI_THRESHOLD
# ...
# _build_situation_summary (~ligne 127) :
n_drought = sum(1 for v in values if v < DROUGHT_SPI_THRESHOLD)
# _build_drought_episodes (~ligne 286) : remplacer le -1.0 par DROUGHT_SPI_THRESHOLD
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `pytest tests/test_drought_threshold.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5 : Commit**

```bash
git add api/era5_anomaly.py api/routers/observatory_climat.py tests/test_drought_threshold.py
git commit -m "fix(climat): seuil sécheresse recalé sur la frontière de classe (-0.84), centralisé"
```

---

### Task 2 : Frontend — repeindre les palettes SPI/STI en RdBu cohérent

**Files:**
- Modify: `frontend/src/lib/era5-colors.ts` (`SPI_CLASS_COLORS` ~:164, `STI_CLASS_COLORS` ~:96)
- Test: `frontend/src/lib/era5-colors.test.ts` (créer)

**Interfaces:**
- Produces: `SPI_CLASS_COLORS` et `STI_CLASS_COLORS` avec `NORMAL === '#f7f7f7'` identiques ; SPI sec = rouge, STI chaud = rouge.

- [ ] **Step 1 : Écrire le test qui échoue**

```ts
// frontend/src/lib/era5-colors.test.ts
import { describe, it, expect } from 'vitest'
import { SPI_CLASS_COLORS, STI_CLASS_COLORS } from './era5-colors'

describe('palettes d\'anomalie cohérentes', () => {
  it('Normal est le même gris neutre pour SPI et STI', () => {
    expect(SPI_CLASS_COLORS.NORMAL).toBe('#f7f7f7')
    expect(STI_CLASS_COLORS.NORMAL).toBe(SPI_CLASS_COLORS.NORMAL)
  })
  it('SPI : sec = rouge, humide = bleu', () => {
    expect(SPI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#b2182b') // très sec = rouge
    expect(SPI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#2166ac') // très humide = bleu
  })
  it('STI : chaud = rouge, froid = bleu (axe inversé, même rouge = préoccupant)', () => {
    expect(STI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#b2182b') // très chaud = rouge
    expect(STI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#2166ac')  // très froid = bleu
  })
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: FAIL (couleurs actuelles BrBG/cold-hot)

- [ ] **Step 3 : Implémenter (repeindre)**

Dans `frontend/src/lib/era5-colors.ts` :
```ts
// SPI — divergent RdBu : sec = rouge, Normal = gris neutre, humide = bleu.
export const SPI_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#b2182b', TRES_BAS: '#ef8a62', BAS: '#fddbc7',
  NORMAL: '#f7f7f7',
  HAUT: '#d1e5f0', TRES_HAUT: '#67a9cf', EXTREMEMENT_HAUT: '#2166ac',
  UNKNOWN: '#6b7280',
}
// STI — RdBu inversé : chaud (HAUT) = rouge, Normal = gris neutre, froid (BAS) = bleu.
export const STI_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#2166ac', TRES_BAS: '#67a9cf', BAS: '#d1e5f0',
  NORMAL: '#f7f7f7',
  HAUT: '#fddbc7', TRES_HAUT: '#ef8a62', EXTREMEMENT_HAUT: '#b2182b',
  UNKNOWN: '#6b7280',
}
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/lib/era5-colors.ts frontend/src/lib/era5-colors.test.ts
git commit -m "feat(climat): palettes SPI/STI en RdBu cohérent (rouge=préoccupant, Normal neutre unique)"
```

---

### Task 3 : Frontend — barre de synthèse (distribution 7 classes) remplace les chips lat/lon

**Files:**
- Modify: `frontend/src/components/climat/SituationBanner.tsx`
- Modify: `frontend/src/lib/climat-situation-format.ts` (retirer la construction `mostAffected`)
- Modify: `frontend/src/i18n/locales/fr.json` + `en.json` (`climat.banner.droughtPct`)
- Test: `frontend/src/components/climat/SituationBanner.test.tsx` (créer)

**Interfaces:**
- Consumes: `ClimatSituationSummary` (`classes_pct: Record<string,number>`, `pct_secheresse`, `driest_since_year`, `available`).
- Produces: un `<div role="img" aria-label>` barre empilée + phrase ; plus aucune chip `top5_cellules_seches`.

- [ ] **Step 1 : Écrire le test qui échoue**

```tsx
// frontend/src/components/climat/SituationBanner.test.tsx
import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { SituationBanner } from './SituationBanner'

const summary = {
  month: '2026-06', window: 3, n_cells: 100,
  classes_pct: { EXTREMEMENT_BAS: 10, TRES_BAS: 15, BAS: 20, NORMAL: 40, HAUT: 10, TRES_HAUT: 5, EXTREMEMENT_HAUT: 0 },
  pct_secheresse: 45, median_spi: -0.7, driest_since_year: 2011, is_driest_on_record: false,
  top5_cellules_seches: [], available: true,
}

describe('SituationBanner', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('affiche le % en sécheresse et la barre de distribution', () => {
    render(<SituationBanner summary={summary} isLoading={false} />)
    expect(screen.getByText(/45\s*%/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /distribution/i })).toBeInTheDocument()
  })

  it('n\'affiche plus de coordonnées lat/lon brutes', () => {
    render(<SituationBanner summary={summary} isLoading={false} />)
    expect(screen.queryByText(/°N|°O|°E|°S/)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/components/climat/SituationBanner.test.tsx`
Expected: FAIL (barre absente / chips encore présentes)

- [ ] **Step 3 : Implémenter**

Remplacer le rendu de `SituationBanner.tsx` (garder la signature `Props { summary, isLoading }`). Barre empilée pilotée par `SPI_CLASS_ORDER` + `SPI_CLASS_COLORS`, phrase via i18n :
```tsx
import { SPI_CLASS_ORDER, SPI_CLASS_COLORS } from '@/lib/era5-colors'
import { useTranslation } from 'react-i18next'
import type { ClimatSituationSummary } from '@/types/observatory-types'

interface Props { summary: ClimatSituationSummary | undefined; isLoading: boolean }

export function SituationBanner({ summary, isLoading }: Props) {
  const { t } = useTranslation()
  if (isLoading) return <div className="climat-banner">{t('climat.banner.loading')}</div>
  if (!summary?.available) return <div className="climat-banner">{t('climat.banner.indicesUnavailable')}</div>
  return (
    <div className="climat-banner">
      <p>
        {t('climat.banner.droughtPct', { pct: summary.pct_secheresse })}
        {summary.driest_since_year != null && ` · ${t('climat.banner.driestSince', { year: summary.driest_since_year })}`}
      </p>
      <div role="img" aria-label={t('climat.banner.distributionAria')} className="climat-banner-dist">
        {SPI_CLASS_ORDER.filter(c => (summary.classes_pct[c] ?? 0) > 0).map(c => (
          <span key={c} title={`${summary.classes_pct[c]} %`}
            style={{ width: `${summary.classes_pct[c]}%`, background: SPI_CLASS_COLORS[c], display: 'inline-block', height: 10 }} />
        ))}
      </div>
    </div>
  )
}
```
Dans `climat-situation-format.ts` : retirer la construction/export de `mostAffected` (n'est plus consommé). Dans `fr.json` : `climat.banner.droughtPct` → `"{{pct}} % du territoire en sécheresse (≤ Modérément sec)"` ; ajouter `climat.banner.distributionAria` → `"Distribution du territoire par classe de sévérité"`. Miroir dans `en.json`. Supprimer la clé `climat.banner.mostAffected`.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd frontend && npx vitest run src/components/climat/SituationBanner.test.tsx`
Expected: PASS (2 tests)

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/components/climat/SituationBanner.tsx frontend/src/lib/climat-situation-format.ts frontend/src/components/climat/SituationBanner.test.tsx frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): barre de synthèse par classe remplace les chips lat/lon"
```

---

### Task 4 : Frontend — VariablePicker en 2 familles + libellés de fenêtre parlants

**Files:**
- Modify: `frontend/src/components/climat/VariablePicker.tsx`
- Modify: `frontend/src/i18n/locales/fr.json` + `en.json` (`climat.picker.*`, libellés de fenêtre)
- Test: `frontend/src/components/climat/VariablePicker.test.tsx` (étendre l'existant)

**Interfaces:**
- Consumes: props inchangées `{ variable, onVariableChange, window, onWindowChange }`.
- Produces: deux groupes visuels — *Anomalie* (`spi`, `sti`, `bilan_hydrique`) et *Absolu* (`precipitation`, `temperature`, `etp`) ; fenêtres labellisées.

- [ ] **Step 1 : Écrire le test (étendre l'existant)**

Ajouter à `VariablePicker.test.tsx` :
```tsx
it('groupe les variables en Anomalie et Absolu', () => {
  render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
  expect(screen.getByText('Anomalie')).toBeInTheDocument()
  expect(screen.getByText('Valeur absolue')).toBeInTheDocument()
})
it('labellise les fenêtres SPI (court terme / saisonnier / long terme / nappe)', () => {
  render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
  expect(screen.getByRole('radio', { name: /Court terme/ })).toBeInTheDocument()
  expect(screen.getByRole('radio', { name: /Nappe/ })).toBeInTheDocument()
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/components/climat/VariablePicker.test.tsx`
Expected: FAIL (« Anomalie » absent, fenêtres numériques nues)

- [ ] **Step 3 : Implémenter**

Dans `VariablePicker.tsx` : scinder `CLIMAT_VARIABLE_ORDER` en deux constantes locales `ANOMALY_VARS = ['spi','sti','bilan_hydrique']` et `ABSOLUTE_VARS = ['precipitation','temperature','etp']`, rendues sous deux `<fieldset>` avec `<legend>{t('climat.picker.familyAnomaly')}</legend>` / `familyAbsolute`. Pour la fenêtre, mapper la valeur → libellé :
```tsx
const WINDOW_LABELS: Record<number, string> = {
  1: t('climat.picker.window1'), 3: t('climat.picker.window3'),
  6: t('climat.picker.window6'), 12: t('climat.picker.window12'),
}
// bouton fenêtre : aria-label = `${w} — ${WINDOW_LABELS[w]}`
```
i18n `fr.json` `climat.picker` : ajouter `familyAnomaly:"Anomalie"`, `familyAbsolute:"Valeur absolue"`, `window1:"Court terme (1 mois)"`, `window3:"Saisonnier (3 mois)"`, `window6:"Long terme (6 mois)"`, `window12:"Nappe (12 mois)"`. Miroir `en.json`.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd frontend && npx vitest run src/components/climat/VariablePicker.test.tsx`
Expected: PASS (tous, y compris les tests existants)

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/components/climat/VariablePicker.tsx frontend/src/components/climat/VariablePicker.test.tsx frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): picker en 2 familles (anomalie/absolu) + fenêtres labellisées"
```

---

### Task 5 : Frontend — bilan hydrique en classes nommées (binning mm, présentation)

**Files:**
- Create: `frontend/src/lib/climat-scale.ts` (binning bilan + resolver classe→couleur/label)
- Modify: `frontend/src/components/climat/ClimatLegend.tsx` (légende bilan par classes)
- Test: `frontend/src/lib/climat-scale.test.ts` (créer)

**Interfaces:**
- Produces: `classifyBilan(mm: number): string` (retourne une clé McKee `EXTREMEMENT_BAS…EXTREMEMENT_HAUT`, réutilise SPI_CLASS_COLORS/labels — déficit = rouge). Bandes mm : `< -150`, `[-150,-75)`, `[-75,-20)`, `[-20,20]`, `(20,75]`, `(75,150]`, `> 150`.
- Note : binning physique = présentation (frontend), distinct de la classification statistique z (backend).

- [ ] **Step 1 : Écrire le test qui échoue**

```ts
// frontend/src/lib/climat-scale.test.ts
import { describe, it, expect } from 'vitest'
import { classifyBilan } from './climat-scale'
import { SPI_CLASS_COLORS } from './era5-colors'

describe('classifyBilan (binning mm, déficit = rouge)', () => {
  it('déficit sévère → EXTREMEMENT_BAS (rouge)', () => {
    expect(classifyBilan(-200)).toBe('EXTREMEMENT_BAS')
    expect(SPI_CLASS_COLORS[classifyBilan(-200)]).toBe('#b2182b')
  })
  it('équilibré → NORMAL (neutre)', () => {
    expect(classifyBilan(0)).toBe('NORMAL')
    expect(SPI_CLASS_COLORS[classifyBilan(0)]).toBe('#f7f7f7')
  })
  it('fort surplus → EXTREMEMENT_HAUT (bleu)', () => {
    expect(classifyBilan(200)).toBe('EXTREMEMENT_HAUT')
  })
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/lib/climat-scale.test.ts`
Expected: FAIL (`classifyBilan` inexistant)

- [ ] **Step 3 : Implémenter**

```ts
// frontend/src/lib/climat-scale.ts
// Présentation : binning du bilan hydrique (mm) en classes nommées réutilisant la
// palette d'anomalie (déficit = rouge = préoccupant). Distinct de la classification
// statistique z→classe qui reste backend (api/era5_anomaly.classify_index).
const BILAN_BANDS: [number, string][] = [
  [-150, 'EXTREMEMENT_BAS'], [-75, 'TRES_BAS'], [-20, 'BAS'],
  [20, 'NORMAL'], [75, 'HAUT'], [150, 'TRES_HAUT'],
]
export function classifyBilan(mm: number): string {
  for (const [hi, cls] of BILAN_BANDS) { if (mm < hi) return cls }
  return 'EXTREMEMENT_HAUT'
}
```
(Note : `-20` supérieur inclusif géré par la cascade `< hi` ; `0` tombe dans `< 20` → NORMAL.)
Dans `ClimatLegend.tsx` : pour `variable === 'bilan_hydrique'`, rendre une légende 7 classes (comme index) avec labels dédiés — ajouter en i18n `climat.bilanClasses.*` : `EXTREMEMENT_BAS:"Déficit sévère" … NORMAL:"Équilibré" … EXTREMEMENT_HAUT:"Fort surplus"`.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd frontend && npx vitest run src/lib/climat-scale.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/lib/climat-scale.ts frontend/src/lib/climat-scale.test.ts frontend/src/components/climat/ClimatLegend.tsx frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): bilan hydrique en classes nommées (déficit/équilibré/surplus)"
```

---

### Task 6 : Frontend — carte bilan en classes discrètes cohérentes

**Files:**
- Modify: `frontend/src/lib/climat-colors.ts` (`bilan_hydrique` : passer d'un gradient continu à un `match` par classe via `classifyBilan`)
- Modify: `frontend/src/components/climat/ClimatMap.tsx` si besoin (routing raw→index-like pour bilan)
- Test: `frontend/src/lib/climat-colors.test.ts` (créer)

**Interfaces:**
- Consumes: `classifyBilan` (Task 5), `SPI_CLASS_COLORS`.
- Produces: `climatBilanColorExpression()` (MapLibre) mappant `value` (mm) → couleur de classe, cohérente carte ↔ légende.

- [ ] **Step 1 : Écrire le test qui échoue**

```ts
// frontend/src/lib/climat-colors.test.ts
import { describe, it, expect } from 'vitest'
import { climatBilanColorExpression } from './climat-colors'

describe('climatBilanColorExpression', () => {
  it('produit une expression MapLibre step/case non vide', () => {
    const expr = climatBilanColorExpression()
    expect(Array.isArray(expr)).toBe(true)
    expect(JSON.stringify(expr)).toContain('#b2182b') // déficit sévère présent
    expect(JSON.stringify(expr)).toContain('#f7f7f7') // neutre présent
  })
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/lib/climat-colors.test.ts`
Expected: FAIL (`climatBilanColorExpression` inexistant)

- [ ] **Step 3 : Implémenter**

Dans `climat-colors.ts`, ajouter une expression `step` MapLibre sur `value` alignée sur les bandes de `classifyBilan`, avec les couleurs `SPI_CLASS_COLORS` :
```ts
import { SPI_CLASS_COLORS } from './era5-colors'
export function climatBilanColorExpression() {
  const C = SPI_CLASS_COLORS
  return [
    'step', ['get', 'value'],
    C.EXTREMEMENT_BAS,          // value < -150
    -150, C.TRES_BAS,
    -75,  C.BAS,
    -20,  C.NORMAL,
    20,   C.HAUT,
    75,   C.TRES_HAUT,
    150,  C.EXTREMEMENT_HAUT,
  ]
}
```
Router `bilan_hydrique` vers cette expression dans `ClimatMap.updateLayer` (au lieu de `climatRawColorExpression`).

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd frontend && npx vitest run src/lib/climat-colors.test.ts`
Expected: PASS

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/lib/climat-colors.ts frontend/src/lib/climat-colors.test.ts frontend/src/components/climat/ClimatMap.tsx
git commit -m "feat(climat): carte bilan hydrique en classes discrètes cohérentes avec la légende"
```

---

### Task 7 : Frontend — extraire l'état de ClimatPage dans `useClimatState`

**Files:**
- Create: `frontend/src/hooks/useClimatState.ts`
- Modify: `frontend/src/pages/ClimatPage.tsx` (consommer le hook)
- Test: `frontend/src/hooks/useClimatState.test.ts` (créer)

**Interfaces:**
- Produces: `useClimatState(): { variable, setVariable, window, setWindow, month, setMonth, day, setDay, isIndex, isDaily }` — extrait les `useState` + dérivés `isIndex`/`isDaily` de `ClimatPage`. Le wiring des overlays reste dans `ClimatPage`.

- [ ] **Step 1 : Écrire le test qui échoue**

```ts
// frontend/src/hooks/useClimatState.test.ts
import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useClimatState } from './useClimatState'

describe('useClimatState', () => {
  it('défaut = spi, fenêtre 3, isIndex vrai', () => {
    const { result } = renderHook(() => useClimatState())
    expect(result.current.variable).toBe('spi')
    expect(result.current.window).toBe(3)
    expect(result.current.isIndex).toBe(true)
    expect(result.current.isDaily).toBe(false)
  })
  it('passer à tmax rend isDaily vrai', () => {
    const { result } = renderHook(() => useClimatState())
    act(() => result.current.setVariable('tmax'))
    expect(result.current.isDaily).toBe(true)
    expect(result.current.isIndex).toBe(false)
  })
})
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd frontend && npx vitest run src/hooks/useClimatState.test.ts`
Expected: FAIL (`useClimatState` inexistant)

- [ ] **Step 3 : Implémenter**

```ts
// frontend/src/hooks/useClimatState.ts
import { useState } from 'react'
import type { ClimatVariable } from '@/lib/climat-colors'
import { isClimatIndexVariable, isClimatDailyVariable } from '@/lib/climat-colors'

export function useClimatState() {
  const [variable, setVariable] = useState<ClimatVariable>('spi')
  const [window, setWindow] = useState(3)
  const [month, setMonth] = useState('')
  const [day, setDay] = useState('')
  return {
    variable, setVariable, window, setWindow, month, setMonth, day, setDay,
    isIndex: isClimatIndexVariable(variable),
    isDaily: isClimatDailyVariable(variable),
  }
}
```
Dans `ClimatPage.tsx` : remplacer les 4 `useState` + les dérivés `isIndex`/`isDaily` par `const s = useClimatState()` et propager (`s.variable`, `s.setVariable`, …). Aucun changement de comportement (les hooks data et le wiring des overlays restent).

- [ ] **Step 4 : Lancer, vérifier le succès (+ non-régression)**

Run: `cd frontend && npx vitest run`
Expected: PASS (toute la suite Climat verte)

- [ ] **Step 5 : Commit**

```bash
git add frontend/src/hooks/useClimatState.ts frontend/src/hooks/useClimatState.test.ts frontend/src/pages/ClimatPage.tsx
git commit -m "refactor(climat): extraire l'état de ClimatPage dans useClimatState"
```

---

### Task 8 : Cohérence transverse IPS — décision documentée (PAS de code en Lot 1)

**Constat** : la cohérence de **direction** (« rouge = préoccupant ») entre le module Climat et l'IPS piézo est **déjà atteinte après la Task 2**, sans toucher l'IPS :
- IPS piézo = palette RdYlGn où **rouge = nappe basse = préoccupant** (déjà conforme).
- Climat après Task 2 = RdBu où **rouge = sec/chaud/déficit = préoccupant**.
→ Un utilisateur lit « rouge = ça va mal » de façon cohérente sur les deux modules.

**Ce qui reste incohérent (cosmétique, non traité en Lot 1)** : la couleur du `Normal` (gris `#f7f7f7` climat vs jaune pâle `#ffffbf` IPS) et la famille de palette (RdBu vs RdYlGn). **Ne pas** forcer le `Normal` de l'IPS à gris : cela casserait la continuité d'une rampe RdYlGn (rouge→…→gris→…→vert) et dégraderait probablement le rendu. Une vraie unification (repeindre l'IPS en RdBu) pose une question de sémantique propre — « nappe haute = bien » se traduit-il par bleu comme « humide » ? — qui mérite son propre cadrage.

**Décision** : réconciliation complète de la palette IPS ↔ Climat = **chantier séparé** (brainstorming dédié), hors Lot 1. Le Lot 1 livre la cohérence de direction + la cohérence interne au Climat (Normal neutre unique sur SPI/STI/bilan via Tasks 2/5/6).

*(Aucun code, aucun commit pour cette tâche.)*

---

## Vérification finale (après toutes les tâches)

- [ ] `cd frontend && npx vitest run` → toute la suite verte.
- [ ] `pytest tests/ -k "drought or situation or classify"` → vert.
- [ ] Passe manuelle sur `/climat` : basculer SPI → STI → Bilan, vérifier que **Normal est le même gris** partout, que le **rouge = préoccupant** (sec/chaud/déficit), que le **% du bandeau tombe pile sur la frontière « Modérément sec »** de la légende, et que **plus aucune coordonnée lat/lon brute** n'apparaît.
- [ ] **Flush cache Redis junon** (`junon:*` situation-summary) après déploiement backend — le recalage du seuil change les valeurs servies. Voir mémoire junon-backend / procédure de flush ciblé `junon:obs_*` déjà utilisée.

## Notes de portée

- **Hors Lot 1** (→ Lot 2, spec dédié) : agrégation par territoire, géocodage inverse (noms de lieux), flèches de tendance, grille-au-zoom.
- **Décision ouverte confirmée** : les bandes mm du bilan (±150/±75/±20) sont un premier calage — à ajuster sur la distribution réelle si le rendu paraît déséquilibré (Task 5/6, purement frontend, sans risque).
