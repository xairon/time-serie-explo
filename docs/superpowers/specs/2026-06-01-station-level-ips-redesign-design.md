# Refonte du système niveau / alerte des stations (basé IPS/SSFI)

Date : 2026-06-01
Statut : design validé (en attente de relecture spec)
Périmètre : observatoire `time-serie-explo` (drawer carte + page station détail) + une étape données.

## Problème

Le système actuel mélange trois axes incompréhensibles pour l'utilisateur :

1. **Classement « Bas/Haut »** : percentile de la **moyenne annuelle** vs l'historique de la station
   (`fct_yearly_*`, 5 classes, seuils 0,2/0,4/0,6/0,8). Non standard, et seulement 5 classes.
2. **« Alerte : Normal »** : en réalité la **pente de régression sur 5 ans** (`niveau_alerte` dans
   `dim_piezo_stations`) — une tendance pluriannuelle, pas le niveau. Affichée dans une **boîte rouge
   même quand c'est « Normal »** → trompeur.
3. **Incohérence 5 vs 7 classes** : marqueurs/badges = 5 classes annuelles, mais la légende carte et
   la timeline « sécheresse » = 7 classes mensuelles. La légende ne correspond pas aux marqueurs.
4. **Aucune explication** de « m NGF » ni de la base du classement.

## Objectif

Un indicateur de niveau **unique, standard, factuel**, intuitif pour un newbie comme pour un pro,
**sans texte généré** (pas de phrases auto-rédigées). Sobre, efficace.

## Décisions (validées)

- **Base = indices standardisés déjà implémentés** dans `dashboard/utils/drought.py` :
  - Nappes → **IPS/SPLI** (KDE → loi normale, méthode BRGM).
  - Rivières → **SSFI** (loi gamma).
  - **7 classes Météo-France/ADES** (`_THRESHOLDS_7`, seuils z = ±0,84 / ±1,28 / ±1,75).
  - Classe du **dernier mois** par station via `classify_latest_spli` / `classify_latest_ssfi`.
- **Champ « Alerte » (tendance) supprimé** du détail (drawer + page).
- **Unification sur 7 classes** partout : marqueurs, légende, stats nationales, page Alertes, détail.
- **Périmètre** : drawer (clic carte) **+** page station détail.
- Historique insuffisant (`< 60 mois` ou `< 10 obs/mois calendaire`) → **« Non classé »** explicite (gris).
- Page **Alertes** : on garde la logique « années consécutives » telle quelle (option a) ; seule la
  classe d'en-tête devient IPS/SSFI. Pas de refonte de la logique d'alerte dans cette itération.

## Architecture données

Le « niveau actuel » doit être disponible **en masse** (≈ 22 400 piézo + 6 250 hydro) pour les marqueurs
et les stats, pas seulement par requête station. `gold.dim_*_stations` est géré par dbt (pipeline BRGM,
rafraîchi chaque nuit — cf. `daily-transform-chain-fix`). On l'alimente ainsi :

- Une **étape Python nocturne dans le pipeline BRGM** (Dagster) calcule, par station, l'indice
  standardisé du dernier mois + sa classe 7 niveaux, en **réutilisant la méthodologie de `drought.py`**
  (`classify_latest_spli` pour piézo, `classify_latest_ssfi` pour hydro), à partir de
  `gold.fct_monthly_chroniques` / `gold.fct_monthly_hydro`.
- Résultat écrit dans une table dédiée **`gold.station_current_index`** :
  `code`, `type` ('piezo'|'hydro'), `index_name` ('IPS'|'SSFI'), `index_value` (z, arrondi 2 déc.),
  `index_class` (7 classes ou 'UNKNOWN'), `ref_month` (date), `baseline_start`, `baseline_end`,
  `computed_at`. Table **non gérée par dbt** (évite le conflit de propriété) mais peuplée dans la même
  chaîne nightly.
- `dim_piezo_stations` / `dim_hydro_stations` exposent `index_value` + `index_class` via **LEFT JOIN**
  sur cette table (colonnes ajoutées au SELECT final, pas de recalcul dbt).
- L'API (`observatory_*`) et la couche carte lisent `index_class` / `index_value` au lieu de
  `classification_derniere_annee` / `classification_resultat_dern_annee` pour le « niveau actuel ».

Réf. d'implémentation = `dashboard/utils/drought.py` (source de vérité de la math). La portion réutilisée
dans le pipeline BRGM sera un petit module miroir (≈ 60 lignes scipy) ou un import partagé — arbitrage
laissé au plan. `classification_derniere_annee` (annuel) reste pour les vues annuelles des graphiques.

## UI — panneau « Situation »

Sobre, factuel, **zéro phrase générée**. Composé uniquement de champs structurés :

```
┌ NIVEAU DE LA NAPPE ─────────────────────┐
│ ● Bas                            IPS ⓘ  │
│ Très bas ▁▂▃▄[▄]▅▆▇ Très haut    −0.97   │
│ Mesure : 103,5 m NGF ⓘ                   │
│ mai 2026 · référence 1994–2026           │
└──────────────────────────────────────────┘
```

- **Classe** : pastille couleur (palette `CLASSIFICATION_COLORS` existante) + libellé i18n.
- **Échelle 7 niveaux** : barre segmentée Très bas→Très haut, segment courant mis en évidence.
- **Valeur de l'indice** : `IPS −0.97` (piézo) / `SSFI −0.97` (hydro), avec infobulle ⓘ.
- **Mesure brute** : `103,5 m NGF` (piézo) / `… m³/s` ou `… m` (hydro), avec infobulle ⓘ NGF (piézo).
- **Mois de référence + période** de l'historique.
- Si `index_class = 'UNKNOWN'` → bandeau gris « Non classé — historique insuffisant (< 5 ans) »,
  pas d'échelle.

Composant React réutilisable `SituationPanel` partagé entre `StationDrawer` et `StationPage`.

### Infobulles (texte FIXE, i18n — jamais généré)

- **ⓘ IPS** : « Indicateur Piézométrique Standardisé (IPS/SPLI). Compare le niveau de ce mois à tous les
  mois équivalents passés de cette station, ramené à une échelle standard (méthode BRGM / Météo-France).
  0 = médiane ; négatif = plus bas que d'habitude, positif = plus haut. »
- **ⓘ SSFI** : analogue pour le débit des rivières (loi gamma).
- **ⓘ NGF** : « Nivellement Général de la France : altitude de référence nationale (≈ niveau moyen de la
  mer). 103,5 m NGF = altitude de la surface de la nappe. »

## Conséquences (cohérence globale)

- **Marqueurs carte** (`ObservatoryMap`), **légende**, **stats nationales** (`/stats/national`),
  **page Alertes** : classe = `index_class` (IPS/SSFI 7 niveaux). La légende 7 niveaux correspond enfin.
- **Boîte « Alerte »** supprimée de `StationDrawer` (et de la page si présente).
- **`niveau_alerte`** : retiré de l'affichage. (La colonne dbt peut rester ; on cesse de l'exposer.)
- **`/alerts`** : `active_only` + sévérité basculent sur `index_class` ; logique « années consécutives »
  inchangée.

## Hors périmètre (YAGNI)

- Refonte de la logique d'alerte « années consécutives » de la page Alertes.
- Nouveaux indices (SPI précip. en en-tête, etc.) — l'IPS/SSFI suffit pour le niveau.
- Garder/déplacer la tendance pluriannuelle ailleurs (supprimée pour cette itération).
- Bascule Simple/Expert (rejetée au profit d'infobulles à texte fixe).

## Critères de réussite

- Une station active affiche : classe 7 niveaux IPS/SSFI cohérente avec la timeline, l'échelle, la valeur
  d'indice, la mesure (m NGF/m³/s), le mois de réf., et des infobulles fixes (IPS, NGF).
- Plus aucune boîte rouge « Alerte : Normal ».
- Marqueurs et légende utilisent la même échelle 7 niveaux.
- Stations à historique court → « Non classé », pas de fausse classe.
