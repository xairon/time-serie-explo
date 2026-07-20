# Climat — retirer la synthèse narrée du bandeau

**Date** : 2026-07-17
**Statut** : design validé, implémentation directe en TDD (pas de pipeline multi-agents — décision utilisateur, changement petit)
**Repo** : `time-serie-explo` (frontend uniquement — **aucune modification backend/entrepôt**)

## 1. Problème

Le bandeau en haut de la carte Climat affiche une **phrase auto-générée** qui « fait trop LLM » (retour utilisateur) :

- SPI/STI (`SituationBanner`) : « X % du territoire ≤ Modérément sec · mois le plus sec depuis AAAA », suivie d'une barre de distribution 7 classes.
- Journalières (`DailyBanner`) : « Pluie max France : X mm · N cellules ≥ 50 mm » (phrase seule, pas de barre).

Le ton de résumé narré — surtout « mois le plus sec depuis AAAA » — est ce qui sonne généré.

## 2. Décision

**Retirer les phrases générées des deux familles ; garder la barre de distribution SPI/STI** (c'est un graphique, pas du texte narré — et le % y reste lisible, cf. le recalage du seuil du 2026-07-17 : la part sèche = somme des 3 classes les plus sèches, lisible sur la barre).

- `SituationBanner` → **barre seule** (+ ses états loading / indisponible, + son `aria-label`).
- `DailyBanner` → **supprimé** (il n'avait que la phrase). En journalier, plus rien en haut : carte + légende + DayStepper suffisent.
- Pas de titre statique ajouté à la barre (l'utilisateur a validé la barre nue ; `aria-label` conservé pour l'accessibilité).

## 3. Périmètre des fichiers (vérifié par grep)

**Supprimer** (modules devenus entièrement morts) :
- `components/climat/DailyBanner.tsx` + `.test.tsx` (seul consommateur : `ClimatPage`).
- `lib/climat-daily-format.ts` + `.test.ts` (`buildDailyBannerData`/`formatOneDecimal`/`HEAT_CELL_THRESHOLD_C` : consommés uniquement par `DailyBanner`).
- `lib/climat-situation-format.ts` + `.test.ts` (`buildSituationBannerData`/`formatDroughtPct`/`SituationBannerData`/`FormatLocale` : consommés uniquement par la phrase de `SituationBanner` et par `climat-daily-format`, tous deux supprimés ; `climat-episodes.ts` n'y fait référence que dans un **commentaire**, pas un import).

**Modifier** :
- `components/climat/SituationBanner.tsx` : retirer le bloc phrase (`droughtPct` + `driestSince`), l'appel `buildSituationBannerData` et son import. Conserver la barre, les états loading/indisponible, l'`aria-label`.
- `components/climat/SituationBanner.test.tsx` : la barre reste asserted ; les assertions sur la phrase partent.
- `pages/ClimatPage.tsx` : `{s.isDaily ? <DailyBanner…/> : <SituationBanner…/>}` → rien en journalier, `SituationBanner` sinon. Retirer l'import `DailyBanner`. **Ne pas toucher** aux hooks de données journalières (`tempPoints`/`precipPoints`/`dailyPoints`/`dailyLoading`) : ils alimentent la carte, pas seulement le bandeau supprimé.
- `pages/ClimatPage.test.tsx` : retirer le `vi.mock('@/components/climat/DailyBanner', …)`.
- `i18n/locales/fr.json` + `en.json` : retirer les clés `climat.banner.droughtPct`, `driestSince`, `dailyTempSummary_one/_other`, `dailyPrecipSummary_one/_other`, `dailyTempUnavailable`. **Conserver** `loading`, `indicesUnavailable`, `distributionAria` (utilisées par la barre). **Ne pas toucher** `climat.picker.dailyTempInfo` (tooltip du picker, sans rapport).

## 4. Backend — on ne touche pas

`_build_situation_summary` continue de calculer `pct_secheresse` / `driest_since_year` : calcul désormais non affiché, mais l'info survit dans la barre, et ces champs sont épinglés par des tests (dont le recalage du 2026-07-17). Les arracher casserait ce test pour un gain nul. Hors périmètre.

## 5. Vérification

- `tsc -b` (via `npm run build`) attrape tout import mort oublié après suppression — c'est le filet principal.
- Frontend : `SituationBanner.test.tsx` (barre présente, phrase absente) ; suite complète verte ; build OK.
- End-to-end : sur `/climat`, en SPI/STI le haut n'affiche que la barre ; en Tx/Tn/Pluie, rien en haut ; la carte, la légende et le DayStepper fonctionnent.

## 6. Risque

Faible. Périmètre frontend confiné ; `tsc` garantit qu'aucun import mort ne subsiste. Aucun autre module ne consomme les fichiers supprimés (grep). Le seul point d'attention : ne pas emporter par erreur les hooks de données journalières de `ClimatPage`, qui servent la carte.
