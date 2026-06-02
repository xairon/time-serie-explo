# Script de démo JUNON — fil conducteur « nappe de Beauce »

Démo pensée pour montrer l'utilité de l'outil en ~15 min, sur un cas d'usage réel :
**suivi et prévision de la nappe de Beauce** (aquifère emblématique de la gestion
sécheresse). Tout est pré-calculé : le jour J, on **navigue**, on ne calcule pas.

URL locale : http://dib-2019006065:49513 (ou l'URL publique une fois sur K8s DSI).

---

## 0. Pré-vol (à vérifier avant la présentation)

- [ ] Les 3 stacks tournent (`docker ps` : junon-frontend / junon-backend / junon-mlflow + brgm-postgres).
- [ ] Comptes prêts : compte **admin** connecté (identifiants hors de ce dépôt).
- [ ] Artefacts de démo présents (déjà en base, ne pas re-supprimer) :
  - IA : **NHiTS mono-station, 2 par station** (univarié + multivarié météo) pour Ruan & Engenville (MAE 0,18–0,29 m)
  - Pastas : Ruan `03276X0009/P` + Engenville `03282X0043/S1`, **2 modèles chacun** —
    *complet* (toutes données, opérationnel, NSE ≈ 0,70–0,72) + *validation* (holdout 80/20,
    NSE validation Ruan +0,51 — généralise ; Engenville négatif — station à forte dérive)
- [ ] Connexion réseau front → back OK (ouvrir une page atelier loggué = données qui chargent).
- [ ] Onglet déjà ouvert sur l'observatoire, zoomé sur la région Centre-Val de Loire.

Station pivot : **Ruan `03276X0009/P`** (Loiret, ~59 ans). Station bis : **Engenville `03282X0043/S1`**.

---

## 1. OBSERVER — l'observatoire national (public, ~3 min)

1. Page d'accueil = carte. *« 28 660 stations : 22 400 piézo + 6 250 hydro, données Hub'Eau/BRGM, 1967→2026. »*
2. Lancer la **frise chronologique animée** (TimelineSlider, en bas) → jouer **2019→2023**.
   *« On voit la sécheresse 2022 colorer la France en rouge, mois par mois. »* → effet visuel fort.
3. Filtrer : zone Beauce / département Loiret (45). *« On se concentre sur la nappe de Beauce. »*
4. Cliquer la station **Ruan** sur la carte → le panneau station s'ouvre.

**Message** : un observatoire complet, public, sans login — la donnée brute pour tous.

---

## 2. ZOOMER — la fiche station (~3 min)

Sur la page station Ruan (`/station/piezo/03276X0009/P`) :
1. **Fiche technique** : ~59 ans, liens ADES / Hub'Eau / BDLISA (montrer qu'on relie aux référentiels officiels).
2. **Panneau Situation** : l'indice **SPLI/IPS** du mois (classe Météo-France) → *« où en est la nappe par rapport à sa normale de saison. »*
3. **Séries temporelles** : basculer jour / mois / an ; superposer la **pluie**.
4. **Indices sécheresse** : SPLI (nappe) + SPI (pluie) côte à côte.

**Message** : pour chaque station, lecture immédiate de l'état + historique + indices standardisés.

---

## 3. COMPRENDRE — modèle physique Pastas (~3 min)

Depuis la fiche station → bouton **« Analyser dans Pastas »** (ou page Pastas, charger le modèle Ruan déjà calibré).
1. Montrer l'ajustement **observé vs simulé** (modèle complet, NSE ≈ 0,72) : *« un modèle physique relie la pluie efficace au niveau de nappe. »*
2. **Réponse impulsionnelle / contributions** : *« combien de temps la nappe met à réagir à la pluie, part recharge vs prélèvement. »*
3. **Honnêteté méthodo** (si public technique) : 2 modèles par station — un *complet* (opérationnel) et un *validation* (holdout temporel). La validation révèle la non-stationnarité (dérive long terme) ; d'où l'usage d'une tendance + modèle de bruit.
4. (Si le temps le permet) un **scénario** : « et si la pluie baissait de 20 % ? ».

**Message** : au-delà de l'observation, on **explique** la dynamique — interprétable pour un hydrogéologue.

---

## 4. PRÉVOIR — l'IA (~4 min)

### 4a. Prévision sur une station
Page IA → modèle **NHiTS univarié** Ruan → **prévision à 30 jours** + intervalle.
*« MAE 0,25 m sur une nappe qui varie d'environ 1 m — robuste. »* (Engenville : MAE 0,19 m.)

### 4b. Deux modèles par station (univarié + multivarié)
Chaque station de démo a **2 modèles NHiTS mono-station** :
- **univarié** (niveau seul) — la prévision opérationnelle ;
- **multivarié** (+ pluie/ETP/température) — pour l'explicabilité.

*« Bien entraînés, les deux sont au coude-à-coude (Ruan 0,25 vs 0,29 ; Engenville 0,19 vs 0,18) — la météo n'apporte pas de gain de précision net ici, mais ouvre l'explicabilité. »*

### 4c. Explicabilité / contrefactuel (sur le modèle multivarié)
« Ce qui pilote le niveau de nappe » : importance **pluie / ETP / température** + scénario « et si la pluie… ».

**Message** : prévision opérationnelle (univarié) + interprétation des moteurs météo (multivarié).

---

## 5. (Optionnel) la rigueur sous le capot — pour un public technique

Si questions « c'est sérieux ? » :
- **Comparaison honnête de 6 architectures** (NHiTS, NBEATS, DLinear, **TSMixer, TiDE, TFT**) dans plusieurs régimes (mono-station, multivarié pluie, covariable calendaire future, global multi-stations). **Le modèle léger NHiTS gagne partout** — conforme à la littérature (Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023).
- **Métriques auditées** : split temporel strict, scalers fit sur train seul, covariables passées (pas de fuite du futur), backtest `retrain=False`. Deux bugs de métrique trouvés et corrigés (NSE par-fenêtre, agrégation globale).
- **Qualité des données** : bug de précision de coordonnées ERA5 détecté et corrigé (1300 grilles re-raccordées à leur historique 1950→2026).

**Message** : démarche scientifique, pas une boîte noire.

---

## Plan B (si un module rame en live)

- Le réseau front→back coupe → montrer l'**observatoire** (public, robuste) + captures d'écran des modèles.
- Pastas/IA lents → ce sont des **résultats pré-calculés** : on ouvre la page, on ne relance rien.
- Tout le pipeline tourne aussi en local sur dib (`:49513`) indépendamment de la DSI.

## Phrase de clôture
*« JUNON, c'est la chaîne complète sur la nappe de Beauce : observer → comprendre → prévoir,
dans un seul outil, sur données officielles, avec une démarche validée. »*
