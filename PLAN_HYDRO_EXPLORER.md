# Plan : Page Exploration Hydrogéologique

## Objectif

Nouvelle page Streamlit permettant de visualiser sur une carte interactive les piézomètres
et stations hydrométriques regroupés par masse d'eau souterraine (entité BDLISA),
avec affichage des séries temporelles au clic.

## Source de données

Toutes les données sont déjà dans l'entrepôt PostgreSQL (`gold` schema) :

| Table | Contenu |
|-------|---------|
| `gold.hubeau_daily_chroniques` | Piézométrie + météo ERA5 (code_bss, code_eh, libelle_eh, lat/lon, niveau_nappe, précip, température…) |
| `gold.hydro_daily_chroniques` | Hydrométrie — stations rivières (débit, hauteur) |
| Tables ERA5 | Données climatiques grillées |

Champs clés pour le regroupement : `code_eh`, `libelle_eh`, `nature_eh`, `niveau_eh`.

## Architecture de la page

```
Sidebar :
  - Sélecteur masse d'eau (code_eh / libelle_eh)
  - Filtres : nature_eh, département, période

Main :
  ┌─────────────────────────────┐
  │  Carte interactive (Leafmap)│
  │  - Markers piézos (bleu)    │
  │  - Markers hydro (vert)     │
  │  - Contour masse d'eau      │
  └─────────────────────────────┘
  Stats : N piézomètres · M stations hydro

  ┌─────────────────────────────┐
  │  Série temporelle (Plotly)  │
  │  au clic sur un point       │
  │  niveau_nappe + précip      │
  └─────────────────────────────┘
```

## Stack technique

- **Leafmap** (`streamlit-leafmap`) : carte interactive, fond OSM/IGN, GeoJSON natif
- **GeoPandas** : manipulation spatiale (bounding box, convex hull, jointure spatiale)
- **Plotly** : séries temporelles (déjà utilisé dans le projet)
- **SQLAlchemy/psycopg2** : connexion BDD (déjà en place via `postgres_connector.py`)

### Optionnel / phase 2

- **WhiteboxTools** : calcul de bassins versants à partir d'un MNT (BD ALTI IGN 25m)
- **Polygones BDLISA officiels** : GeoJSON BRGM pour les vrais contours de masses d'eau
  (en phase 1, un convex hull des piézos suffit comme approximation)

## Étapes d'implémentation

### Phase 1 — Carte + piézos

1. Requête SQL : `SELECT DISTINCT code_eh, libelle_eh, nature_eh, count(DISTINCT code_bss)
   FROM gold.hubeau_daily_chroniques GROUP BY 1,2,3` → dropdown sidebar
2. Sélection masse d'eau → requête des piézos (code_bss, lat, lon, commune, altitude)
3. Affichage carte Leafmap avec markers piézos
4. Convex hull GeoPandas pour le contour approximatif de la masse d'eau
5. Clic sur un piézo → série temporelle (niveau_nappe + précipitations) en Plotly

### Phase 2 — Stations hydro

6. Vérifier structure de `gold.hydro_daily_chroniques` (colonnes lat/lon)
7. Requête stations hydro proches (filtre bounding box autour de la masse d'eau)
8. Markers hydro sur la carte (couleur/icône différente)
9. Clic sur station hydro → série débit/hauteur

### Phase 3 — Enrichissements

10. Polygones BDLISA officiels (téléchargement unique, stockage local ou en base)
11. Fond de carte IGN via tuiles WMS
12. Sélection multiple de piézos pour comparaison de séries
13. WhiteboxTools : calcul bassin versant au clic sur un exutoire
14. Corrélation croisée nappe ↔ rivière ↔ précipitations

## Dépendances à installer

```
pip install leafmap streamlit-leafmap geopandas
```

(WhiteboxTools uniquement si phase 3)
