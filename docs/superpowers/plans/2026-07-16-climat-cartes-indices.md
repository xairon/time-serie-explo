# Climat — cartes = indices, valeurs absolues = chiffres — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retirer les 3 couches carte en valeur absolue mensuelle (précipitation, température, ETP) du module Climat, et exposer ces faits comme des chiffres exacts dans le PointPanel.

**Architecture:** Frontend uniquement. On rétrécit l'union `ClimatVariable` à `spi | sti | bilan_hydrique | tmax | tmin | tmean` — `tsc` sert de filet : toute occurrence oubliée casse le build. Le picker perd sa famille « Valeur absolue ». Le PointPanel gagne un bloc « bilan du mois » alimenté par la série **déjà chargée** (`useClimatPointSeries`), sans aucun appel réseau supplémentaire.

**Tech Stack:** React 18 + TypeScript, vite, vitest + @testing-library/react, react-i18next, MapLibre.

**Spec:** `docs/superpowers/specs/2026-07-16-climat-etp-echelle-temperature-design.md`

## Global Constraints

- **Doctrine produit** : soit un **vrai indicateur** (IPS, SPLI, SPI, STI — standard reconnu, seuils publiés), soit une **vraie valeur** (°C/mm). Jamais un intermédiaire inventé. Toute échelle de couleur ré-ancrée sur la climatologie EST un indice maison → interdit.
- **Support** : carte choroplèthe = indicateur ; nombre = vraie valeur.
- **Aucune modification backend/entrepôt.** Les params `temperature`/`precipitation`/`etp` de `/grid-monthly` restent supportés côté serveur, simplement plus appelés. Ne pas nettoyer l'API (YAGNI).
- **Ne pas toucher** : `tmax`/`tmin`/`tmean`, `DailyTempBanner`, `DayStepper`, SPI/STI/bilan, `PrecipNormalChart`.
- **Ne pas confondre les modules** : `lib/era5-colors.ts` et `lib/era5-zones.ts` (+ leurs tests) appartiennent à l'**Observatory** et utilisent le type `Era5Variable`. Leurs occurrences `'temperature'`/`'precipitation'` sont **hors périmètre**. Seul `lib/climat-colors.ts` (type `ClimatVariable`) est concerné.
- **UI en français** (public BRGM). Toute chaîne visible passe par i18n (`fr.json` **et** `en.json`).
- **`npm run build` est obligatoire avant tout commit** : vitest ne typecheck pas. Seul `tsc -b` attrape les erreurs de type — et c'est lui qui casse le build de l'image Docker et la CI.
- Commandes lancées depuis `frontend/`.

---

### Task 1: Rétrécir le type `ClimatVariable` et retirer la famille « Absolu » du picker

Retire les 3 variables du modèle **et** du picker en une seule tâche : `tsc` refuserait de compiler un état intermédiaire où le type a perdu `'etp'` mais où `ABSOLUTE_VARS` le référence encore. Les deux fichiers forment un seul déliverable cohérent.

**Files:**
- Modify: `frontend/src/lib/climat-colors.ts:10-12` (type), `:42-89` (`CLIMAT_VARIABLES`), `:92-94` (`CLIMAT_VARIABLE_ORDER`)
- Modify: `frontend/src/components/climat/VariablePicker.tsx:20` (`ABSOLUTE_VARS`), `:59` (rendu du groupe)
- Modify: `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json` (clé `climat.picker.familyAbsolute`)
- Test: `frontend/src/lib/climat-colors.test.ts:69-70`, `frontend/src/components/climat/VariablePicker.test.tsx:9`

**Interfaces:**
- Consumes: rien (première tâche).
- Produces: `export type ClimatVariable = 'spi' | 'sti' | 'bilan_hydrique' | 'tmax' | 'tmin' | 'tmean'` ; `CLIMAT_VARIABLE_ORDER: ClimatVariable[]` valant `['spi', 'sti', 'bilan_hydrique']`. `CLIMAT_VARIABLES`, `DAILY_TEMP_VARIABLE_ORDER`, `isClimatIndexVariable`, `isClimatDailyVariable`, `climatFormatValue`, `climatRawColorExpression`, `climatGradientCss`, `climatRawDomain`, `climatBilanColorExpression` gardent leurs signatures **inchangées**.

- [ ] **Step 1: Mettre à jour les tests existants qui référencent les variables retirées**

Dans `frontend/src/lib/climat-colors.test.ts`, remplacer les lignes 69-70 (qui utilisent `'temperature'`, sur le point de disparaître de l'union) par le cas équivalent sur une variable conservée :

```ts
    expect(isClimatDailyVariable('bilan_hydrique')).toBe(false)
    expect(isClimatIndexVariable('bilan_hydrique')).toBe(false)
```

Dans `frontend/src/components/climat/VariablePicker.test.tsx`, remplacer le test de la ligne 9 (`renders all 6 climat variables with SPI first`) par :

```tsx
  it('renders only the 3 index variables, SPI first, and no absolute family', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'SPI (précipitations)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'STI (température)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Bilan hydrique' })).toBeInTheDocument()
    // La famille « Valeur absolue » a été retirée (doctrine : cartes = indicateurs).
    expect(screen.queryByRole('radio', { name: 'Précipitations' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: 'Température' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: 'ETP' })).not.toBeInTheDocument()
    // Les journalières restent (domaine absolu fixe, cf. spec §2.1).
    expect(screen.getByRole('radio', { name: 'Tx (max)' })).toBeInTheDocument()
  })
```

- [ ] **Step 2: Lancer les tests pour les voir échouer**

Run: `npx vitest run src/components/climat/VariablePicker.test.tsx -t "renders only the 3 index"`
Expected: FAIL — `queryByRole('radio', { name: 'Précipitations' })` trouve encore le bouton (la famille Absolu est toujours rendue).

- [ ] **Step 3: Rétrécir le type et la config dans `climat-colors.ts`**

Remplacer le type (lignes 10-12) par :

```ts
export type ClimatVariable =
  | 'spi' | 'sti' | 'bilan_hydrique'
  | 'tmax' | 'tmin' | 'tmean'
```

Dans `ClimatVarConfig`, restreindre `monthlyParam` (seul `bilan_hydrique` l'utilise encore) :

```ts
  /** `variable` query param for GET /observatory/climat/grid-monthly (raw vars only). */
  monthlyParam?: 'bilan_hydrique'
```

Supprimer les 3 entrées `precipitation`, `temperature`, `etp` de `CLIMAT_VARIABLES` (lignes 59-73) — **conserver** `spi`, `sti`, `bilan_hydrique`, `tmax`, `tmin`, `tmean` tels quels.

Remplacer `CLIMAT_VARIABLE_ORDER` (lignes 92-94) par :

```ts
/** Ordered for the picker UI: SPI first (default), then STI, then the water balance.
 *  Les valeurs absolues mensuelles (précip/température/ETP) ne sont plus des couches :
 *  une carte porte un indicateur, un nombre porte la valeur (cf. spec 2026-07-16). */
export const CLIMAT_VARIABLE_ORDER: ClimatVariable[] = ['spi', 'sti', 'bilan_hydrique']
```

Mettre à jour le commentaire d'en-tête du fichier (lignes 4-5) qui cite les variables retirées :

```ts
// same warehouse classification via api/era5_anomaly.py::classify_index — no duplication
// of the class→colour mapping). Le bilan hydrique et les températures journalières
// (Tx/Tn/Tmoy) gardent leurs échelles ici car les endpoints Climat
// (api/routers/observatory_climat.py) renvoient une propriété générique `value`.
```

- [ ] **Step 4: Retirer la famille « Absolu » du picker**

Dans `frontend/src/components/climat/VariablePicker.tsx`, remplacer le commentaire + les constantes (lignes 16-20) par :

```tsx
/** SPI/STI/bilan hydrique are deviations from a 1991-2020 normal — "is this month
 *  unusual?". Ce sont les seules couches mensuelles : les valeurs absolues
 *  (précipitation/température/ETP) sont des CHIFFRES dans le PointPanel, pas des
 *  cartes (cf. spec 2026-07-16 — une carte porte un indicateur, un nombre porte
 *  la valeur). Les journalières ci-dessous font exception : domaine absolu fixe. */
const ANOMALY_VARS: ClimatVariable[] = ['spi', 'sti', 'bilan_hydrique']
```

Supprimer la ligne 59 : `{renderVariableGroup(ABSOLUTE_VARS, 'climat.picker.familyAbsolute')}`

- [ ] **Step 5: Retirer la clé i18n devenue inutilisée**

Supprimer la clé `familyAbsolute` de l'objet `climat.picker` dans `frontend/src/i18n/locales/fr.json` **et** `frontend/src/i18n/locales/en.json`.

**Conserver** `climat.variables.temperature`, `climat.variables.etp`, `climat.variables.bilanHydrique`, `climat.variables.precipitation` : elles ne sont plus atteintes via `CLIMAT_VARIABLES` mais la Task 2 les référence directement.

- [ ] **Step 6: Lancer les tests + le build**

Run: `npx vitest run src/components/climat/VariablePicker.test.tsx src/lib/climat-colors.test.ts`
Expected: PASS

Run: `npm run build`
Expected: succès. **Si `tsc` signale une occurrence résiduelle** de `'precipitation'`/`'temperature'`/`'etp'` typée `ClimatVariable`, la corriger — c'est le filet qui fait son travail. Ne PAS toucher `lib/era5-colors.ts` / `lib/era5-zones.ts` (type `Era5Variable`, autre module).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/climat-colors.ts frontend/src/lib/climat-colors.test.ts \
        frontend/src/components/climat/VariablePicker.tsx \
        frontend/src/components/climat/VariablePicker.test.tsx \
        frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): les cartes ne portent plus que des indicateurs

Retire précipitation/température/ETP de l'union ClimatVariable et la
famille « Valeur absolue » du picker. Une carte choroplèthe répond à « où
est-ce anormal ? » (indicateur) ; « combien ? » est le travail d'un chiffre,
ajouté au PointPanel dans le commit suivant."
```

---

### Task 2: Bloc « bilan du mois » dans le PointPanel

**Files:**
- Modify: `frontend/src/components/climat/PointPanel.tsx` (imports, helper, rendu après `PrecipNormalChart`)
- Modify: `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json` (clés `climat.pointPanel.balance*`)
- Test: `frontend/src/components/climat/PointPanel.test.tsx`

**Interfaces:**
- Consumes: `ClimatPointSeriesEntry` (`@/lib/observatory-types:282`) — champs `month`, `temperature_moyenne`, `precipitation_totale`, `etp_totale`, `bilan_hydrique`, tous `number | null`. `classifyBilan(mm: number): string` (`@/lib/climat-scale`) → une des 7 clés `EXTREMEMENT_BAS|TRES_BAS|BAS|NORMAL|HAUT|TRES_HAUT|EXTREMEMENT_HAUT`. `era5SpiClassColor(cls: string): string` (`@/lib/era5-colors:181`) → couleur CSS de la classe, repli gris `UNKNOWN` si inconnue. La série est déjà chargée dans `PointPanel` : `const series = pointData?.series ?? []` (ligne 38).
- Produces: rien (feuille).

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter dans `frontend/src/components/climat/PointPanel.test.tsx`. Le fichier mocke déjà `@/hooks/useClimat` — s'appuyer sur ce mock existant en lui faisant renvoyer une série. Adapter le mock de `useClimatPointSeries` pour qu'il retourne :

```tsx
  it('affiche le bilan du mois avec les vraies valeurs du dernier mois', () => {
    vi.mocked(useClimatPointSeries).mockReturnValue({
      data: { series: [
        { month: '2026-05', temperature_moyenne: 15.1, precipitation_totale: 70,
          etp_totale: 90, bilan_hydrique: -20 },
        { month: '2026-06', temperature_moyenne: 18.3, precipitation_totale: 40,
          etp_totale: 120, bilan_hydrique: -80 },
      ] },
      isLoading: false, isError: false,
    } as any)

    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)

    expect(screen.getByText('18.3 °C')).toBeInTheDocument()
    expect(screen.getByText('40 mm')).toBeInTheDocument()
    expect(screen.getByText('120 mm')).toBeInTheDocument()
    expect(screen.getByText('−80 mm')).toBeInTheDocument()   // U+2212, pas un tiret ASCII
    expect(screen.getByText('Déficit')).toBeInTheDocument()   // classifyBilan(-80) -> TRES_BAS
  })

  it('rend « — » sur les champs nuls d’un mois partiel sans masquer le bloc', () => {
    vi.mocked(useClimatPointSeries).mockReturnValue({
      data: { series: [
        { month: '2026-07', temperature_moyenne: null, precipitation_totale: null,
          etp_totale: null, bilan_hydrique: null },
      ] },
      isLoading: false, isError: false,
    } as any)

    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)

    expect(screen.getByText('Bilan du mois')).toBeInTheDocument()
    expect(screen.getAllByText('—')).toHaveLength(4)
  })
```

Si `useClimatPointSeries` n'est pas encore importé dans le test, l'ajouter : `import { useClimatPointSeries } from '@/hooks/useClimat'`.

- [ ] **Step 2: Lancer le test pour le voir échouer**

Run: `npx vitest run src/components/climat/PointPanel.test.tsx -t "bilan du mois"`
Expected: FAIL — `Unable to find an element with the text: 18.3 °C` (le bloc n'existe pas).

- [ ] **Step 3: Ajouter les clés i18n**

Dans `frontend/src/i18n/locales/fr.json`, objet `climat.pointPanel`, ajouter :

```json
"balanceTitle": "Bilan du mois",
"balanceHint": "Valeurs mesurées sur la maille ERA5 la plus proche."
```

Dans `frontend/src/i18n/locales/en.json`, même objet :

```json
"balanceTitle": "Monthly balance",
"balanceHint": "Values measured on the nearest ERA5 cell."
```

- [ ] **Step 4: Implémenter le bloc**

Dans `frontend/src/components/climat/PointPanel.tsx`, ajouter aux imports :

```tsx
import { classifyBilan } from '@/lib/climat-scale'
import { era5SpiClassColor } from '@/lib/era5-colors'
```

Ajouter au-dessus du composant `PointPanel` (formateur local : `climatFormatValue` n'est plus utilisable, `temperature`/`precipitation`/`etp` ayant quitté l'union `ClimatVariable`) :

```tsx
/** Formateur local du bloc « bilan du mois ». `climatFormatValue` ne convient pas :
 *  il indexe CLIMAT_VARIABLES, dont temperature/precipitation/etp ont été retirés
 *  (ce ne sont plus des couches). Rend le vrai chiffre, ou — s'il manque. */
function fmtValue(v: number | null | undefined, unit: string, digits = 0): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v.toFixed(digits)} ${unit}`
}

/** Idem, mais signé avec un vrai U+2212 (cohérent avec climatFormatValue). */
function fmtSigned(v: number | null | undefined, unit: string): string {
  if (v == null || Number.isNaN(v)) return '—'
  const s = Math.abs(Math.round(v)).toString()
  return `${v < 0 ? `−${s}` : `+${s}`} ${unit}`
}
```

Dans le corps du composant, après `const series = pointData?.series ?? []` (ligne 38), ajouter :

```tsx
  // Dernier mois de la série = le plus récent. Le panneau est une fiche de lieu :
  // il ne suit pas le MonthStepper de la carte (pas de prop `month` — YAGNI).
  const lastEntry = series.length > 0 ? series[series.length - 1] : undefined
  const bilan = lastEntry?.bilan_hydrique
  const bilanClass = bilan != null && !Number.isNaN(bilan) ? classifyBilan(bilan) : undefined
```

Dans le JSX, insérer **juste avant** `<PrecipNormalChart series={series} />` (ligne 89) :

```tsx
              {lastEntry && (
                <div>
                  <h3 className="text-sm font-semibold text-text-primary mb-2">
                    {t('climat.pointPanel.balanceTitle')}
                  </h3>
                  <dl className="rounded-lg border border-white/10 divide-y divide-white/5">
                    {[
                      { k: 'climat.variables.temperature', v: fmtValue(lastEntry.temperature_moyenne, '°C', 1) },
                      { k: 'climat.variables.precipitation', v: fmtValue(lastEntry.precipitation_totale, 'mm') },
                      { k: 'climat.variables.etp', v: fmtValue(lastEntry.etp_totale, 'mm') },
                    ].map(({ k, v }) => (
                      <div key={k} className="flex items-center justify-between px-3 py-1.5">
                        <dt className="text-xs text-text-secondary">{t(k)}</dt>
                        <dd className="text-xs font-medium text-text-primary tabular-nums">{v}</dd>
                      </div>
                    ))}
                    <div className="flex items-center justify-between px-3 py-1.5">
                      <dt className="text-xs text-text-secondary">{t('climat.variables.bilanHydrique')}</dt>
                      <dd className="text-xs font-medium text-text-primary tabular-nums flex items-center gap-1.5">
                        {fmtSigned(bilan, 'mm')}
                        {bilanClass && (
                          <span
                            className="text-[10px] px-1.5 py-0.5 rounded"
                            style={{ backgroundColor: `${era5SpiClassColor(bilanClass)}33`, color: era5SpiClassColor(bilanClass) }}
                          >
                            {t(`climat.bilanClasses.${bilanClass}`, { defaultValue: bilanClass })}
                          </span>
                        )}
                      </dd>
                    </div>
                  </dl>
                  <p className="text-[10px] text-text-secondary mt-1">{t('climat.pointPanel.balanceHint')}</p>
                </div>
              )}
```

- [ ] **Step 5: Lancer les tests + le build**

Run: `npx vitest run src/components/climat/PointPanel.test.tsx`
Expected: PASS (les 2 nouveaux tests + les existants)

Run: `npm run build`
Expected: succès.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/climat/PointPanel.tsx \
        frontend/src/components/climat/PointPanel.test.tsx \
        frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(climat): bloc « bilan du mois » (T/P/ETP/bilan) dans le PointPanel

Les valeurs absolues quittent la carte mais pas l'app : elles réapparaissent
en chiffres exacts là où on les lit vraiment. L'ETP y retrouve son seul usage
défendable — expliquer le bilan hydrique (déficit -80 = P 40 vs ETP 120).
Aucun appel réseau ajouté : la série est déjà chargée par useClimatPointSeries."
```

---

### Task 3: Vérification de bout en bout dans l'app réelle

Les tests unitaires ne prouvent pas que la carte fonctionne encore. Cette tâche exerce le vrai parcours.

**Files:** aucun (vérification).

**Interfaces:**
- Consumes: le module Climat complet, tel que modifié par les Tasks 1-2.
- Produces: rien.

- [ ] **Step 1: Suite complète + build**

Run: `npx vitest run src/ && npm run build`
Expected: tous les tests PASS, build OK.

- [ ] **Step 2: Lancer l'app et exercer le parcours**

Run: `npm run dev` puis ouvrir `/climat`.

Vérifier, dans cet ordre :
1. Le picker n'affiche que **Anomalie** (SPI, STI, Bilan hydrique) + **Températures journalières** (Tx, Tn, T moy). Aucune famille « Valeur absolue ».
2. SPI, STI et Bilan hydrique s'affichent toujours en 7 classes nommées avec leur légende.
3. Tx/Tn/T moy s'affichent toujours (rampe météo) et le DayStepper fonctionne.
4. Cliquer une maille → le PointPanel s'ouvre → le bloc **« Bilan du mois »** affiche 4 lignes (Température, Précipitations, ETP, Bilan hydrique + pastille de classe).
5. Le graphe « Précipitations vs normale » est toujours là sous le bloc.

- [ ] **Step 3: Vérifier une maille de mois partiel**

Le mois courant est souvent partiel (`mois_complet=false`). Ouvrir le PointPanel sur n'importe quelle maille et confirmer que le bloc reste affiché même si des lignes valent `—` (aucun crash, aucun bloc masqué).

- [ ] **Step 4: Commit (si des correctifs ont été nécessaires)**

```bash
git add -A
git commit -m "fix(climat): correctifs issus de la vérification end-to-end"
```

Si aucun correctif n'a été nécessaire, ne rien commiter — la vérification est passée.
