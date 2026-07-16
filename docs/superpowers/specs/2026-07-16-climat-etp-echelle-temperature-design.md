# Climat — sort de l'ETP & échelle de la température mensuelle

**Date** : 2026-07-16
**Statut** : design validé, prêt pour plan d'implémentation
**Repo** : `time-serie-explo` (frontend junon uniquement — aucune modification API/entrepôt)
**Amende** : `2026-07-15-climat-module-lisibilite-lot1-design.md` (§4.1 tableau des familles, §5 VariablePicker)

## 1. Contexte & problème

Retour utilisateur direct après la livraison du Lot 1, sur deux couches de la famille « Absolu » :

1. **« Pourquoi on affiche l'ETP ? On s'en fout non ? »**
2. **« Les cartes de température sont inutiles : pour juin c'est juste rouge uni, sans info. »**

### 1.1 Diagnostic technique

Le point 2 est **fondé et structurel**. La couche `temperature` utilise une échelle **fixe −10 → +35 °C** (`climat-colors.ts:67`), dimensionnée pour l'année entière. Or en juin la France s'étale sur ~15→22 °C, soit **~15 % de la rampe** : toute la variance spatiale tombe dans une seule tranche de couleur. Le défaut est général — *une échelle absolue annuelle écrase la variance spatiale d'un mois donné* — et il se reproduit à chaque mois.

Nuance importante, vérifiée : **le défaut ne touche que la mensuelle.**

- `precipitation` (0→200 mm fixe) va bien : la précip mensuelle parcourt réellement cette plage entre régions.
- Les journalières `tmax`/`tmin`/`tmean` utilisent `DAILY_TEMP_STOPS` (−10→42 °C), rampe calibrée météo (28 °C orange, 34 °C rouge, 38 °C rouge sombre). Un Tx de juin à 30-38 °C **montre** de la structure. Leur caractère absolu est **voulu** : un jour à 35 °C doit être rouge partout et toujours, c'est ce qui les rend comparables d'un jour à l'autre.

Le point 1 (ETP) n'est pas un bug mais une question d'usage : l'ETP est une **quantité intermédiaire**. Seule, une carte d'ETP ne dit rien d'exploitable ; son sens n'existe que **relativement à la précipitation** — c'est-à-dire dans le bilan hydrique, qui est déjà exposé.

### 1.2 Ce que disait le Lot 1

Le Lot 1 avait **déjà identifié la redondance** (§4.1) : « précip et température brutes sont quasi redondantes avec SPI et STI (leurs versions standardisées via la climatologie déjà calculée) ». Il a néanmoins choisi de **garder les 6 indicateurs**, en rangeant les absolus en « contexte, à la demande ». Le présent document **révise ce choix pour l'ETP uniquement**, sur retour utilisateur.

## 2. Cadrage produit (validé)

**Rôle du module Climat : observatoire climatique complet.** L'utilisateur veut aussi les faits bruts (« il a fait 32 °C », « il est tombé 40 mm), pas seulement l'anomalie. Conséquence directe : on **répare** la carte mensuelle de température **au lieu de la supprimer**, et on garde les journalières.

Ce cadrage exclut deux options écartées :

- *Tout ramener au STI* (supprimer les températures absolues) — rejeté : le module n'est pas qu'un détecteur d'anomalies.
- *Classes nommées par mois pour la température* — rejeté : on perdrait le « il a fait 18,3 °C », et le STI **est déjà** la version en classes.

## 3. Décision 1 — ETP : couche carte → chiffre au point

L'ETP quitte le picker et réapparaît là où elle a du sens : **en explication du bilan hydrique**, à l'endroit où l'utilisateur se pose la question.

**Retraits** (`climat-colors.ts`, `VariablePicker.tsx`) :

- `'etp'` du type `ClimatVariable`, de `CLIMAT_VARIABLES`, de `CLIMAT_VARIABLE_ORDER`, et de `ABSOLUTE_VARS` (`VariablePicker.tsx:20`).
- Le paramètre `etp` reste supporté côté API (`observatory_climat.py`, mapping `"etp": "etp_totale"`) : **on ne casse rien**, on cesse simplement de l'exposer comme couche. Aucune modification backend.

**Ajout** (`PointPanel.tsx`) : un bloc « bilan du mois » alimenté par la série **déjà chargée** par `useClimatPointSeries`. Vérifié : `ClimatPointSeriesEntry` (`observatory-types.ts:282`) contient déjà `precipitation_totale`, `etp_totale`, `bilan_hydrique` → **coût backend nul**.

```
Précipitations    40 mm
ETP              120 mm
Bilan hydrique   −80 mm   (déficit)
```

La classe du bilan (`déficit` / `équilibré` / `surplus`) réutilise `classifyBilan` (`climat-scale.ts`) — pas de nouveau système de classes.

**Gestion des `null`** : le mois courant est souvent partiel (`mois_complet=false`) et les champs peuvent être `null`. Le bloc affiche `—` par champ manquant via `climatFormatValue`, sans masquer le bloc entier.

## 4. Décision 2 — Température mensuelle : échelle climatologique par mois

**Principe** : **ré-ancrer le domaine sur le mois affiché**, et adapter le langage de couleur en conséquence.

### 4.0 Règle transverse : le langage de couleur suit la sémantique du domaine

C'est le cœur du design, et il corrige une incohérence détectée en relecture (l'ancienne §5.1) :

> - **Domaine absolu fixe → couleurs à sémantique absolue** (rampe météo arc-en-ciel). Un jour à 35 °C est rouge, partout, toujours.
> - **Domaine ré-ancré par mois → encodage relatif → rampe séquentielle**, pâle → soutenu. La couleur dit « plus frais / plus chaud *dans ce mois* » ; ce sont la **légende et le point** qui portent la valeur absolue.

**Pourquoi c'est obligatoire, et pas une préférence** : ré-ancrer une rampe à sémantique absolue lui fait dire des faussetés. Avec la rampe actuelle ré-ancrée, juin [12, 26] peindrait **12 °C en violet** (couleur du grand froid) et **26 °C en magenta** (couleur du record) ; janvier [−2, 12] peindrait **12 °C en rouge**, alors que 12 °C en janvier est *doux*. Une rampe séquentielle mono-teinte n'a pas ce défaut : « pâle → soutenu » ne prétend rien sur le froid absolu.

Cette règle **unifie** le module au lieu d'y créer des exceptions :

| Couche | Domaine | Langage de couleur |
|---|---|---|
| `tmax` / `tmin` / `tmean` (journalières) | absolu fixe (−10→42) | arc-en-ciel météo — **inchangé, et désormais justifié** |
| `temperature` (mensuelle) | ré-ancré par mois | **séquentielle mono-teinte** |
| `precipitation` | absolu fixe (0→200) | séquentielle mono-teinte — inchangé |

Elle **rend caduc** l'écart signalé face au Lot 1 §4.1 (« séquentielle mono-teinte, désaturée » pour la famille Absolu) : la mensuelle s'y conforme désormais, non par obéissance mais parce que le ré-ancrage l'exige. Aucun amendement du Lot 1 n'est nécessaire sur ce point.

```ts
TEMP_RAMP_COLORS: string[]                        // rampe séquentielle mono-teinte, sans valeurs
TEMP_MONTHLY_DOMAIN: Record<number, [number, number]>  // 1..12 → [min, max]
climatTempStops(month: string): Array<[number, string]>
```

**Choix des teintes** : rampe séquentielle chaude (« plus chaud = plus soutenu »), perceptuellement régulière et lisible en déficience de vision des couleurs — donc **pas** un dérivé de `jet`/arc-en-ciel, qui crée des bandes fantômes et n'est pas monotone en luminance. Les valeurs exactes des stops sont à fixer à l'implémentation **en s'appuyant sur le skill `dataviz`** (palettes séquentielles, validation du contraste) plutôt qu'à l'intuition.

Les couleurs sont réparties uniformément sur `[min, max]` du mois calendaire extrait de `month` (`'YYYY-MM'`).

**Changements de signature** — `climatRawColorExpression(variable, month)`, `climatGradientCss(variable, month)`, `climatRawDomain(variable, month)`. `ClimatMap` et `ClimatLegend` ont déjà `month` en portée ; les autres variables ignorent le paramètre (domaine fixe inchangé).

**Propriétés obtenues** :

- Structure spatiale lisible **toute l'année** (Bretagne fraîche vs Provence chaude, en juin comme en janvier).
- **Juin 2026 comparable à juin 2003** : même mois → même échelle, toutes années. Sert directement `CompareYearsSection`, qui existe déjà.
- Comparaison **inter-mois** cassée (juin vs janvier) — assumé : c'est le rôle du **STI**, déjà présent.

**Saturation** : les valeurs hors domaine se clampent aux couleurs extrêmes (comportement naturel de `interpolate`). Un juin exceptionnel tape le haut de rampe — **c'est l'information, pas un bug**. La légende l'assume en affichant `≤ 12` / `≥ 26` plutôt que des bornes sèches.

Cette notation `≤`/`≥` s'applique **à la seule couche `temperature`**, dont le domaine est désormais serré autour du mois et donc réellement saturable. Les autres variables à dégradé gardent leurs bornes affichées telles quelles (`precipitation` : la borne 200 mm est un maximum de confort rarement atteint, pas un seuil de saturation signifiant ; les journalières : rampe météo inchangée). Élargir la notation aux autres couches serait un changement de périmètre non demandé.

### 4.1 Provenance des 12 domaines (point non négociable)

Les bornes **ne sont pas inventées**. Elles sont dérivées **une fois** par requête sur l'entrepôt — p5/p95 de `temperature_moyenne` par mois calendaire sur la référence **1991-2020**, sur l'emprise France — puis figées en constantes dans `climat-colors.ts`, accompagnées d'un commentaire indiquant **l'origine, la date de calcul et la requête de régénération**.

Justification du figement (plutôt qu'un calcul à la volée) : le cadrage exige une échelle **identique d'une année sur l'autre** pour un même mois. Une échelle recalculée à chaque requête serait adaptative — l'option explicitement rejetée (l'échelle bougerait, deux cartes ne se compareraient plus, et l'utilisateur sur-interpréterait un simple ré-étalonnage).

Le choix p5/p95 (et non min/max) évite qu'une seule maille extrême n'écrase la rampe pour tout le pays.

## 5. Hors périmètre

- `precipitation` (0→200 mm) : plage réellement parcourue, **on ne touche pas**.
- Journalières `tmax`/`tmin`/`tmean` : rampe absolue **volontairement** comparable d'un jour à l'autre, **on ne touche pas**.
- SPI/STI, bilan hydrique en classes : livrés au Lot 1, **on ne touche pas**.
- Agrégation par territoire, noms de lieux → **Lot 2** (spec dédié).

### 5.1 Point tranché (ex-point ouvert)

La première version de ce spec laissait ouvert le sort de la rampe `temperature` : garder l'arc-en-ciel (et amender le Lot 1 §4.1, qui prescrit « séquentielle mono-teinte, désaturée » pour les absolus), ou l'appliquer.

**Tranché en faveur de la rampe séquentielle** — voir §4.0. Le motif n'est pas doctrinal mais logique : le ré-ancrage par mois est **incompatible** avec une rampe à sémantique absolue, qui peindrait 12 °C en violet en juin et 12 °C en rouge en janvier. Le Lot 1 §4.1 se trouve donc respecté sans amendement, et les journalières gardent leur arc-en-ciel sans devenir une exception — les deux découlent de la même règle.

## 6. Tests

- `climat-colors.test.ts` : domaine de juin ≠ domaine de janvier ; stops ordonnés et bornés au domaine du mois ; clamp hors domaine ; les variables non-température ignorent le paramètre `month`.
- `PointPanel.test.tsx` : le bloc affiche P/ETP/bilan ; gère les `null` d'un mois partiel sans disparaître.
- Non-régression : `'etp'` n'apparaît plus dans le picker ; les journalières et la précip sont inchangées.
- **`npm run build` obligatoire** : vitest ne typecheck pas — seul `tsc -b` attrape les erreurs de type, et c'est lui qui casse le build de l'image Docker et la CI (leçon du 2026-07-16).

## 7. Risques

- **Faible.** Périmètre confiné : les 7 consommateurs de `climat-colors` sont tous dans `components/climat/` + `hooks/useClimatState` + `pages/ClimatPage`. Vérifié : **Pastas et Observatory n'importent pas `climat-colors`** (les occurrences `tmax`/`tmean` y sont des noms de séries d'entrée de modèle, sans rapport) → aucun risque de régression hors module.
- Le retrait de `'etp'` du type `ClimatVariable` est attrapé à la compilation par `tsc` (union exhaustive) — toute occurrence oubliée casse le build plutôt que de passer en silence.
