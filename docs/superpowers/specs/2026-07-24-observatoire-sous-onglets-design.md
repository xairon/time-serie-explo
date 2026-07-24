# Observatoire — sous-onglets par compartiment (Nappes & rivières / Climat)

**Date** : 2026-07-24
**Statut** : design validé, prêt pour plan d'implémentation
**Repo** : `time-serie-explo` (frontend uniquement — aucune modification API/entrepôt)

## 1. Problème

La barre de navigation mélange **deux axes différents** :

| Axe | Entrées actuelles |
|---|---|
| *Domaines qu'on observe* | Observatoire, Climat |
| *Outils qu'on applique* | Comparer, Pastas Lab, AI Lab |

`Climat` est aujourd'hui un **pair** d'`Observatoire` (`TopNav.tsx:28-32`) alors que ce n'en
est pas une activité distincte : c'est **la moitié météorologique de la même observation**.
Les deux partagent le même modèle mental — choisir un indicateur standardisé → choisir une
période → lire une choroplèthe → cliquer une maille/station pour le détail — et jusqu'à la
même famille d'indices (7 classes McKee, mêmes conventions de palette, cf. le SPEI livré le
2026-07-24 qui réutilise la palette SPI).

**Coût concret** : la question la plus naturelle d'un hydrogéologue — « cette sécheresse
est-elle encore météorologique (SPI/SPEI) ou a-t-elle atteint la nappe (IPS/SPLI) ? » —
impose aujourd'hui un changement d'onglet de premier niveau qui **perd le contexte de la
carte**. La navigation dresse un mur exactement là où passe la question.

## 2. Décision

`Observatoire` devient un **conteneur à sous-onglets par compartiment** :

```
Observatoire ─┬─ Nappes & rivières   (carte actuelle : piézo/hydro, IPS/SPLI/SSFI,
              │                       BDLISA, secteurs BSH)
              └─ Climat              (module actuel : SPI/STI/SPEI, Situation /
                                      Point-Zone / Comparaison)

Comparer · Pastas Lab · AI Lab   → restent au premier niveau (ce sont des OUTILS)
```

La barre de navigation retrouve ainsi **un seul axe** : un domaine observé, puis des outils.

### 2.1 Ce qu'on ne fait PAS (et pourquoi)

**Les deux cartes restent séparées.** Fusionner les moteurs de carte (superposer le SPI sur
les stations piézo dans une carte unique) est la version « puissante », mais :

- l'Observatoire a déjà son propre système de couches (BDLISA, secteurs, grille ERA5) et le
  module Climat son `VariablePicker` — les réunir demande de trancher un modèle de couches
  commun, c'est un vrai chantier, pas un ajustement de navigation ;
- **cette IA a déjà fait des allers-retours** : la météo-des-nappes a été fondue dans
  l'Observatoire le 2026-06-09, puis un `/meteo` autonome est revenu le 2026-06-10. Une
  refonte lourde de plus, sans besoin utilisateur mesuré, risque un nouveau va-et-vient.

On corrige donc **le défaut d'architecture de l'information** (peu coûteux, réversible) et on
**diffère le pari coûteux** (carte partagée) jusqu'à ce qu'un besoin réel le justifie.

**`/meteo` reste hors périmètre** : c'est délibérément un clone BRGM plein écran, hors
`Layout` (cf. `project-meteo-clone-fidele`). Il ne rentre pas dans les sous-onglets.

## 3. Changements

### 3.1 Routage

- L'Observatoire porte des routes enfants : `/` (Nappes & rivières, index) et
  `/climat` **rendu à l'intérieur de l'Observatoire** plutôt qu'en page sœur.
- **`/climat` reste une URL valide et inchangée** — c'est déjà un lien profond utilisé
  (deep-link depuis la popup de maille de l'Observatoire, cf. README §Climat). Aucune
  redirection, aucun lien cassé : seule la *coquille* qui l'entoure change.
- Les sous-vues Climat (`Situation` / `Point-Zone` / `Comparaison`) sont **inchangées**.

### 3.2 `components/layout/TopNav.tsx`

- Retirer `{ to: '/climat', … }` de `navItems` (ligne 29). Il reste : Observatoire,
  Comparer, Pastas Lab, AI Lab.
- L'ancre de visite guidée `tour: 'nav-climat'` déménage sur le sous-onglet Climat —
  **à ne pas supprimer** sans vérifier le scénario de la visite guidée (`nav-climat` est
  référencé par le tour).

### 3.3 Nouveau `components/observatory/ObservatoryTabs.tsx`

Barre de sous-onglets rendue en haut de la coquille Observatoire, deux entrées
(`NavLink` vers `/` en `end` et `/climat`). Style **visuellement distinct** de la `TopNav`
(plus discret : soulignement plutôt que pastille pleine) pour que la hiérarchie se lise et
qu'on ne croie pas à deux barres de même niveau.

### 3.4 Profondeur — le point de vigilance

Le module Climat a déjà 3 sous-vues. L'empilement donnerait
`Observatoire › Climat › Situation`, soit **trois niveaux**. Traitement retenu : les 3
sous-vues Climat passent en **contrôle segmenté** (segmented control) à l'intérieur de la
page, et non en 3ᵉ rangée d'onglets. Deux rangées d'onglets maximum à l'écran.

### 3.5 i18n

`nav.climat` existe déjà et est réutilisé pour le sous-onglet. Nouvelle clé
`observatory.tabs.groundwater` (fr « Nappes & rivières » / en « Groundwater & rivers »).

## 4. Tests

- `TopNav.test.tsx` : `Climat` n'est plus dans la barre principale ; les 4 entrées restantes
  sont rendues.
- `ObservatoryTabs.test.tsx` : les 2 sous-onglets sont rendus ; l'onglet actif suit la route.
- Un test de non-régression sur le **deep-link** : `/climat` rend bien le module Climat (à
  l'intérieur de la coquille Observatoire).
- `npm run build` (tsc -b) obligatoire.

## 5. Hors périmètre

- Fusion des moteurs de carte / couches partagées (§2.1) — différé.
- `/meteo` (clone plein écran) — inchangé.
- Les sous-vues internes du Climat et toute la logique d'indices — inchangées.
- Aucun changement API/entrepôt.

## 6. Risques

- **Faible et réversible** : le changement est une coquille de routage + une barre
  d'onglets ; aucun état métier ne bouge (`useClimatState` et l'état de l'Observatoire
  restent indépendants, ce qui est justement pourquoi les cartes ne fusionnent pas).
- Vigilance visite guidée : `nav-climat` change d'emplacement (§3.2).
- Vigilance deep-link : `/climat` doit rester fonctionnel (§3.1, couvert par un test).
