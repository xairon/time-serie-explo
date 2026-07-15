# Redesign du module Climat — Lot 1 : langage visuel & lisibilité

**Date** : 2026-07-15
**Statut** : design validé (approche A retenue), prêt pour plan d'implémentation
**Repo** : `time-serie-explo` (frontend junon + API)

## 1. Contexte & problème

Le module Climat (route `/climat`, ~35 composants `climat-*`/`era5-*`, backend `api/routers/observatory_climat.py`) expose 6 indicateurs ERA5 (SPI, STI, bilan hydrique, précipitation, température, ETP) sur une carte grille 0.1° plein écran avec overlays. Les données viennent des marts gold de l'entrepôt (`fct_era5_monthly_grid`, `fct_era5_climatology_grid`, `fct_era5_indices_grid`) ; l'API est une couche de lecture (cache Redis).

Problèmes constatés (audit code + retour utilisateur) :

- **Bandeau « zones les plus touchées »** : affiche les 5 cellules les plus sèches en **lat/lon bruts** (`"48.2°N, 1.7°O"`), sans nom de lieu (pas de géocodage inverse). Faible valeur d'usage.
- **Incohérence sémantique des couleurs** :
  - SPI en palette BrBG (brun = sec, **pas de rouge**), STI en froid→chaud (**rouge = chaud**) → le rouge ne veut pas dire la même chose d'une carte à l'autre.
  - La classe **Normal** est gris quasi-blanc sur SPI mais **vert vif** sur STI.
  - Conflit transverse avec l'IPS piézo : **rouge = nappe basse** (piézo) vs **rouge = chaud** (STI climat).
- **Hétérogénéité de traitement** : SPI/STI en 7 classes nommées ; bilan/précip/temp/ETP en gradients continus sans classes ni repères de sévérité.
- **Seuil incohérent** : le bandeau compte « SPI < −1 » alors que la carte classe en McKee à −0.84/−1.28 → le « % sécheresse » ne correspond à aucune frontière de la légende.
- **Dette technique** : `ClimatPage.tsx` est un god-component d'état ; logique couleur dupliquée (`climat-colors.ts` ⇄ `era5-colors.ts`) ; classification z→classe dupliquée frontend (`classifyIndex`) / backend (`classify_index`), synchronisée à la main.

## 2. Public cible & décision produit (cadrage validé)

- **Public prioritaire** : gestionnaires eau / techniciens — *suivre et anticiper* (trajectoire, comparaison à la normale, fenêtres 1/3/6/12 mois, distinguer sécheresse flash vs de fond).
- **Maille spatiale cible** : territoires en vue d'ensemble → grille au zoom → point au clic (**ambition Lot 2** ; Lot 1 garde la grille + drawer actuels).
- **Hiérarchie des indicateurs** : garder les 6, mais **hiérarchie fixe + une seule logique de couleur + classes nommées partout**.

## 3. Décomposition en lots

Le chantier est décomposé en deux sous-projets indépendamment livrables (chacun aura son spec → plan → implémentation) :

- **Lot 1 (ce document)** : le **langage visuel & la lisibilité**. Frontend + un micro-ajustement backend (recalage seuil). Aucune touche entrepôt. C'est la victoire d'interprétation, livrée en premier.
- **Lot 2 (spec ultérieur)** : **agrégation par territoire** (département / bassin / nappe), vue d'ensemble par zone + flèches de tendance + grille-au-zoom. Nécessite une couche d'agrégation spatiale entrepôt/API (inexistante aujourd'hui).

## 4. Modèle couleur & classification unifié (cœur du Lot 1)

### 4.1 Principe : une grammaire, deux familles

Insight de cadrage : **précip et température brutes sont quasi redondantes avec SPI et STI** (leurs versions standardisées via la climatologie déjà calculée). On ne force donc pas 6 indicateurs dans un moule unique ; on les range en deux familles partageant **la même grammaire visuelle** :

| Famille | Indicateurs | Échelle | Rôle |
|---|---|---|---|
| **Anomalie (héros)** | SPI, STI, bilan hydrique | Divergente ancrée sur Normale/0, **7 classes nommées**, **rouge = préoccupant**, bleu = opposé | Lecture première |
| **Absolu (contexte)** | précipitation (mm), température (°C), ETP (mm) | Séquentielle, libellée « valeur absolue », désaturée | À la demande |

### 4.2 Règles transverses

- **Normal = un seul gris neutre** pour tous les indicateurs d'anomalie (fin du gris-SPI vs vert-STI).
- **Rouge = la direction préoccupante**, systématiquement : SPI sec, STI chaud, bilan en déficit. Bleu = l'autre extrême.
- **7 classes nommées** en clair, plus de z-scores nus sur la carte :
  - Axe eau (SPI, bilan) : `Très sec · Sec · Modérément sec · Normal · Modérément humide · Humide · Très humide`.
  - Axe température (STI) : `Très chaud · Chaud · Modérément chaud · Normal · Modérément froid · Froid · Très froid` (rouge = chaud).
- **Seuils** : conservés WMO/McKee pour SPI/STI (±0.84 / ±1.28 / ±1.75 σ). Bilan hydrique : divergent ancré sur 0, classes par bandes de mm (seuils à définir dans le plan, ex. ±150/±75/±25 mm), libellés `Déficit sévère … Équilibré … Surplus`.
- **Recalage du bandeau** : « % en sécheresse » = part des classes ≤ *Modérément sec* (frontière −0.84σ), qui correspond à une limite visible de la légende. Un seul micro-changement backend dans `observatory_climat.py::_build_situation_summary` (remplacer le seuil `-1.0` codé en dur).
- **Indicateurs absolus** : échelle séquentielle mono-teinte, légende « valeur absolue » explicite ; ils ne prétendent pas être des anomalies.

### 4.3 Réconciliation avec l'IPS piézo (transverse junon)

La grammaire eau (**rouge = sec/déficit**) coïncide déjà avec l'IPS piézo (**rouge = nappe basse = stress**). On extrait donc **un contrat de classification/couleur partagé** consommé par le module Climat ET le module piézo :

- Contrat partagé = **enum 7 classes + palette + direction « rouge = stress » + labels i18n**.
- **Les seuils restent propres à chaque domaine** (BRGM ±0.25/0.84/1.28 pour l'IPS piézo, WMO ±0.84/1.28/1.75 pour SPI/STI) — légitime, domaines différents. On unifie **couleurs + direction + langage de classes**, pas les frontières numériques.
- Le STI (rouge = chaud) reste sur son axe température, clairement étiqueté « température » dans sa légende.

## 5. Hiérarchie de l'information & layout (Lot 1)

On conserve la carte grille 0.1° plein écran + le drawer point (la partie qui fonctionne), on retravaille les overlays.

- **Barre de synthèse** (remplace le bandeau lat/lon) : distribution des 7 classes en **barre empilée** (% du territoire par classe, couleurs unifiées) + phrase claire (« X % du territoire en sécheresse (≤ Modérément sec) · mois le plus sec depuis AAAA »). Suppression des chips lat/lon (le « où » nommé arrive au Lot 2).
- **VariablePicker en 2 familles** : *Anomalie* (SPI, STI, Bilan) en primaire, *Absolu* (précip, temp, ETP) en secondaire replié. Sélecteur de fenêtre SPI/STI avec **libellés parlants** : `Court terme (1 mois) · Saisonnier (3) · Long terme (6) · Nappe (12)`.
- **Légende unifiée** pilotée par le module d'échelle partagé : 7 pastilles nommées pour l'anomalie, barre de gradient « valeur absolue » pour le contexte. Toujours visible, auto-explicative (« Rouge = plus sec que la normale »).
- **Drawer point** : fonctions inchangées (série temporelle SPI/STI, épisodes, comparaison d'années), **re-skiné** aux couleurs unifiées → carte ↔ drawer ↔ légende = un seul langage. Libellés de fenêtre parlants repris.

## 6. Architecture

- **Séparation nette classification (backend) / présentation (frontend)** — c'est ce qui tue la double source de vérité :
  - **Classification = backend seul** (`api/era5_anomaly.py`). Les **seuils** (WMO SPI/STI, bandes mm du bilan) vivent uniquement là. Le backend renvoie déjà `index_class` sur `/grid-indices` ; on étend ce principe (classe renvoyée aussi pour le bilan et, à terme, les indicateurs concernés). Le frontend **supprime** `classifyIndex` — il ne re-classe plus jamais.
  - **Présentation = frontend seul** : nouveau module `frontend/src/lib/climat-scale.ts` exposant l'**enum des 7 classes + `classToColor` + `classToLabel` (clé i18n)** — pas de seuils, pas de `classify()`. Il **fusionne** `climat-colors.ts` et `era5-colors.ts` (fin de la double source de vérité couleur). Le front reçoit une classe de l'API et la mappe en couleur/label.
- **Découpe du god-component** : extraire l'état (variable/fenêtre/mois/jour/cellule) dans un hook `useClimatState` ; chaque overlay (barre de synthèse, picker, légende, drawer) devient un composant focalisé consommant l'état + le module d'échelle. Suppression des ternaires imbriqués index/raw/daily au profit d'un résolveur de famille explicite.
- **Contrat partagé IPS ↔ Climat** : extraire l'enum de classes + la palette + les labels dans un module réutilisable par `lib/ips.ts` et `lib/climat-scale.ts`. `lib/ips.ts` conserve ses seuils BRGM mais consomme la palette partagée.

## 7. Tests & vérification

- **Golden classification (backend)** : table de frontières (valeur → classe) testée unitairement côté `api/era5_anomaly.py`, sur le modèle du contrat IPS cross-repo existant, pour chaque axe (eau, température, bilan). La classification étant backend-only, c'est là que le contrat est verrouillé.
- **Mapping présentation (frontend)** : test que le module `climat-scale.ts` couvre **toutes** les classes que l'API peut renvoyer (aucune classe orpheline sans couleur/label), et que couleur/label sont déterministes.
- **Rendu composants** : tests de rendu picker (2 familles), légende (7 classes nommées + gradient absolu), barre de synthèse (distribution empilée, seuil recalé).
- **Non-régression visuelle** : passe manuelle sur la page Climat après re-skin (carte/drawer/légende cohérents) ; vérifier que le « % sécheresse » du bandeau tombe pile sur une frontière de la légende.

## 8. Risques & décisions ouvertes

- **Bandes de classes du bilan hydrique** : les seuils mm (ex. ±150/±75/±25) sont à caler sur la distribution réelle — à figer dans le plan d'implémentation à partir des données.
- **Changement de couleurs côté piézo** : l'adoption de la palette partagée peut légèrement modifier les teintes IPS existantes ; à valider visuellement (l'IPS est déjà « rouge = bas », donc impact faible).
- **Cache Redis junon** : le recalage du seuil `pct_secheresse` change les valeurs de `/situation-summary` → flush du cache `junon:*` concerné au déploiement.
- **STI vs axe eau** : le STI garde « rouge = chaud » (axe température), distinct de « rouge = sec » (axe eau). Assumé et étiqueté ; risque résiduel de confusion pour un utilisateur qui superpose mentalement les deux — mitigé par des légendes explicites et distinctes.

## 9. Hors périmètre (rappel)

Agrégation par territoire, noms de lieux (géocodage inverse), flèches de tendance par zone, grille-au-zoom → **Lot 2** (spec dédié).
