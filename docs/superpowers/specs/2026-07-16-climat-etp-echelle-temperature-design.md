# Climat — les cartes sont des indices, les valeurs absolues sont des chiffres

**Date** : 2026-07-16
**Statut** : design validé, prêt pour plan d'implémentation
**Repo** : `time-serie-explo` (frontend junon uniquement — aucune modification API/entrepôt)
**Amende** : `2026-07-15-climat-module-lisibilite-lot1-design.md` (§4.1 tableau des familles, §5 VariablePicker)

## 1. Contexte & problème

Retour utilisateur direct après la livraison du Lot 1 :

1. « Pourquoi on affiche l'ETP ? On s'en fout non ? »
2. « Les cartes de température sont inutiles : pour juin c'est juste rouge uni, sans info. »
3. « N'invente rien, utilise toujours des index. »

### 1.1 Diagnostic

Le point 2 est **fondé et structurel**. La couche `temperature` utilise une échelle fixe **−10 → +35 °C** (`climat-colors.ts:67`), dimensionnée pour l'année entière. Or, mesuré sur l'entrepôt (`gold.fct_era5_monthly_grid`, 11 496 mailles × 30 ans), la France de juin s'étale sur **12 → 22 °C** (p5/p95, 1991-2020) : **~22 % de la rampe**. Toute la variance spatiale tombe dans une tranche de couleur. Le défaut se reproduit chaque mois.

Le point 1 (ETP) n'est pas un bug mais un problème d'usage : l'ETP est une **quantité intermédiaire**, dont le sens n'existe que relativement à la précipitation — c'est-à-dire dans le bilan hydrique, déjà exposé.

### 1.2 La fausse piste, et pourquoi le point 3 la tue

Une première version de ce design proposait de **ré-ancrer** l'échelle de température sur un domaine climatologique par mois (juin → [12, 22]). Elle a été **rejetée**, et c'est le cœur de la décision :

- Ré-ancrer une rampe par mois transforme la couleur en **encodage relatif au mois** — « plus chaud que la moyenne nationale de juin ». Or *un encodage relatif à la climatologie du mois, c'est la définition d'un indice*.
- Cet indice serait **maison** : un p5/p95 national, grossier, non standard. Le **STI** fait déjà exactement ce travail, mais **rigoureusement** : z-score **par maille** contre la référence 1991-2020, seuils **WMO/McKee**.
- Dériver proprement les bornes d'un indicateur inventé ne le rend pas standard : **ça reste un indicateur inventé**.

Conclusion : toute tentative de rendre lisible une carte mensuelle en valeur absolue **revient à réinventer un STI en moins bon**. La carte absolue mensuelle n'a pas de design défendable — il ne faut donc pas la réparer, il faut la retirer.

## 2. Décision — la règle transverse

Doctrine projet (énoncée par l'utilisateur, transverse à junon — cf. l'IPS/SPLI piézo) :

> **Soit un vrai indicateur** (IPS, SPLI, SPI, STI… : standard reconnu, seuils publiés),
> **soit une vraie valeur** (la mesure, en °C ou en mm).
> **Jamais un intermédiaire inventé.**

C'est exactement le trou dans lequel tombait l'échelle ré-ancrée de §1.2 : ni un vrai indicateur (pas de z-score par maille, pas de seuils WMO — juste un p5/p95 national bricolé), ni une vraie valeur (la couleur ne signifiait plus rien en absolu, 12 °C virant au violet en juin). Un entre-deux qui ment sur les deux tableaux.

Appliquée au support, la doctrine se traduit ainsi :

> **Une carte choroplèthe répond à « où est-ce anormal ? » → ce sont des indicateurs.
> Un nombre répond à « combien ? » → c'est la vraie valeur, dans le PointPanel.**

Le module confondait les deux dans une même couche ; c'est la cause racine des trois retours. On sépare.

| | Carte | Chiffre au point |
|---|---|---|
| SPI, STI, bilan hydrique | ✅ indices / classes nommées | ✅ (déjà) |
| Précipitation, Température, ETP | ✂ **retirées** | ✅ **nouveau bloc** |
| Tx / Tn / T moy (journalières) | ✅ **conservées** (voir §2.1) | — |

**Cadrage produit préservé** : le module reste un *observatoire climatique complet*. L'utilisateur veut toujours les faits bruts (« il a fait 18,3 °C », « il est tombé 40 mm) — ils ne disparaissent pas, ils **changent de support** : du dégradé illisible vers un chiffre exact, là où on le lit vraiment.

### 2.1 Les journalières : l'exception qui confirme la règle

`tmax` / `tmin` / `tmean` restent en carte, et c'est **cohérent, pas dérogatoire** :

- Leur domaine est **réellement absolu et fixe** (`DAILY_TEMP_STOPS`, −10→42 °C) : un jour à 35 °C est rouge **partout et toujours**. Aucun ré-ancrage, donc aucun indice déguisé.
- Elles sont **comparables d'un jour à l'autre** — c'est précisément la propriété que le ré-ancrage détruisait.
- **Aucun indice journalier standardisé n'existe** dans l'entrepôt : il n'y a pas d'alternative « index » à leur opposer.

La règle est donc : *domaine absolu fixe → carte absolue légitime ; domaine qui devrait être ré-ancré → utiliser l'indice standard qui existe déjà*.

## 3. Changements

### 3.1 Type & configuration (`lib/climat-colors.ts`)

`ClimatVariable` perd `'precipitation'`, `'temperature'`, `'etp'` :

```ts
export type ClimatVariable = 'spi' | 'sti' | 'bilan_hydrique' | 'tmax' | 'tmin' | 'tmean'
```

Retraits correspondants dans `CLIMAT_VARIABLES` et `CLIMAT_VARIABLE_ORDER` (qui devient `['spi', 'sti', 'bilan_hydrique']`). Le type union étant exhaustif, **`tsc` attrape toute occurrence oubliée** — un retrait incomplet casse le build plutôt que de passer en silence.

Effet de bord assumé : la voie « raw gradient » (`climatRawColorExpression`, `climatGradientCss`, `climatRawDomain`) ne sert plus que les **journalières** ; `bilan_hydrique` garde sa voie discrète (`climatBilanColorExpression`). Aucune signature ne change — le paramètre `month` envisagé dans la version précédente n'a plus lieu d'être.

### 3.2 Picker (`components/climat/VariablePicker.tsx`)

La famille « Valeur absolue » **disparaît** : suppression de `ABSOLUTE_VARS` (ligne 20) et de son `renderVariableGroup(...)` (ligne 59). Restent le groupe *Anomalie* et la section *Températures journalières*. La clé i18n `climat.picker.familyAbsolute` devient inutilisée et est **supprimée** (fr + en).

### 3.3 PointPanel — bloc « bilan du mois » (`components/climat/PointPanel.tsx`)

Nouveau bloc alimenté par la série **déjà chargée** par `useClimatPointSeries`. Vérifié : `ClimatPointSeriesEntry` (`observatory-types.ts:282`) contient déjà `precipitation_totale`, `etp_totale`, `bilan_hydrique`, `temperature_moyenne` → **coût backend nul**.

```
Bilan du mois — juin 2026
  Température      18,3 °C
  Précipitations     40 mm
  ETP               120 mm
  Bilan hydrique    −80 mm   (Déficit)
```

- **Mois affiché** : celui de la carte. `PointPanel` reçoit une prop `month` et cherche l'entrée correspondante dans la série.

  *Correction du 2026-07-17 — la première version de ce spec disait l'inverse* (« le dernier de la série […] on n'ajoute pas de prop pour ça — YAGNI »). C'était faux : sans le mois, la carte affiche mai pendant que le panneau annonce « Bilan du mois — juil. », et les deux se contredisent à l'écran. Le panneau suit donc la période active, exactement comme `ClimatLegend` le fait déjà (`month={s.isDaily ? s.day : s.month}`) ; en mode journalier il prend le mois du jour affiché. Mois absent de la série pour cette maille → bloc masqué, plutôt que des valeurs tirées d'un autre mois.
- **Classe du bilan** : réutilise `classifyBilan` (`lib/climat-scale.ts`) + les libellés i18n `climat.bilanClasses.*` + `SPI_CLASS_COLORS` — aucun nouveau système de classes.
- **Formatage** : `climatFormatValue` **n'est pas utilisable ici** — il indexe `CLIMAT_VARIABLES[variable]`, or `temperature`/`precipitation`/`etp` quittent l'union (§3.1). Le bloc utilise un **formateur local** au PointPanel, qui rend `—` sur `null`/`NaN`.
- **Valeurs `null`** (mois partiel) : chaque ligne rend `—` indépendamment ; le bloc reste affiché.
- **Libellés i18n** : les clés `climat.variables.temperature` / `.etp` / `.bilanHydrique` **restent** dans `fr.json`/`en.json` (elles ne sont plus atteintes via `CLIMAT_VARIABLES` mais référencées directement). Ne pas les supprimer.

`PrecipNormalChart` est **conservé** : c'est déjà le fait « précipitation » au point, sous forme de graphe vs normale.

## 4. Hors périmètre

- SPI / STI / bilan hydrique en classes : livrés au Lot 1, **on ne touche pas**.
- Journalières `tmax` / `tmin` / `tmean`, `DailyTempBanner`, `DayStepper` : **on ne touche pas** (§2.1).
- API / entrepôt : les paramètres `temperature`, `precipitation`, `etp` de `/grid-monthly` restent supportés côté serveur — simplement plus appelés. **Aucune modification backend**, aucun nettoyage d'endpoint (YAGNI).
- Agrégation par territoire, noms de lieux → **Lot 2**.

## 5. Réconciliation avec le Lot 1

Le Lot 1 (§4.1) rangeait les 6 indicateurs en deux familles et prescrivait pour les absolus une échelle « séquentielle mono-teinte, désaturée ». Il avait **lui-même relevé la redondance** : « précip et température brutes sont quasi redondantes avec SPI et STI (leurs versions standardisées via la climatologie déjà calculée) ».

Ce document **pousse ce constat à sa conclusion** : si les brutes sont redondantes avec leurs versions standardisées, elles n'ont pas à occuper une couche de carte. La famille « Absolu » du Lot 1 est donc **supprimée**, pas redessinée. Le §4.1 du Lot 1 est amendé en conséquence.

## 6. Tests

- `climat-colors.test.ts` : `CLIMAT_VARIABLE_ORDER` ne contient que les 3 indices ; les tests existants portant sur `'precipitation'` / `'temperature'` sont supprimés ; les tests journalières (`'tmax'`, `'tmean'`, `'tmin'`) et `climatBilanColorExpression` restent **inchangés**.
- `VariablePicker.test.tsx` : la famille « Valeur absolue » n'est plus rendue ; les 3 indices et les 3 journalières le sont.
- `PointPanel.test.tsx` : le bloc affiche T/P/ETP/bilan + la classe du bilan ; gère les `null` d'un mois partiel sans disparaître.
- **`npm run build` obligatoire** : vitest ne typecheck pas — seul `tsc -b` attrape les erreurs de type, et c'est lui qui casse le build de l'image Docker et la CI (leçon du 2026-07-16).

## 7. Risques

- **Faible.** Périmètre confiné : les consommateurs de `climat-colors` sont tous dans `components/climat/` + `hooks/useClimatState` + `pages/ClimatPage`. Vérifié : **Pastas et Observatory n'importent pas `climat-colors`** (leurs occurrences `tmax`/`tmean` sont des noms de séries d'entrée de modèle, sans rapport) → aucune régression hors module.
- Le rétrécissement du type `ClimatVariable` est attrapé à la compilation (union exhaustive).
- **État par défaut : vérifié, aucun risque.** `useClimatState:9` initialise `useState<ClimatVariable>('spi')` — un `useState` simple, sans persistance URL ni localStorage. Aucune valeur retirée ne peut donc être restaurée au chargement, et aucun repli n'est nécessaire.
