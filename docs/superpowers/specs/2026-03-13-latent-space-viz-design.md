# Spec : Visualisation Interactive de l'Espace Latent + Page Laboratoire

**Date** : 2026-03-13
**Statut** : Draft
**Auteur** : Claude + ringuet

---

## 1. Contexte

Le pipeline hubeau_data_integration produit des embeddings SoftCLT (TS2Vec + soft contrastive loss) pour ~5 470 stations hydrologiques (2 935 piézo + 2 535 hydro). Les vecteurs 320d sont stockés dans PostgreSQL via pgvector :

| Table | Granularité | Volume |
|-------|------------|--------|
| `ml.piezo_station_embeddings` | 1 par station (mean-pool) | ~2 935 |
| `ml.piezo_window_embeddings` | fenêtre 365j / stride 90j | ~208K |
| `ml.hydro_station_embeddings` | 1 par station | ~2 535 |
| `ml.hydro_window_embeddings` | fenêtre 365j / stride 90j | ~161K |

HDBSCAN est pré-calculé nightly (cluster_id stocké). L'objectif est de fournir une interface interactive pour explorer cet espace latent avec UMAP paramétrable, clustering configurable, et coloration par attributs hydrogéologiques.

En parallèle, les features expérimentales (contrefactuel, détection pompage) sont regroupées sous une page "Laboratoire".

## 2. Objectifs

1. Créer une page **Laboratoire** regroupant les features expérimentales sous des onglets (nested routes)
2. Construire un onglet **Espace Latent** avec scatter UMAP 2D/3D interactif
3. Permettre le **filtrage par attributs hydrogéologiques** (nappe, milieu, état, thème, géographie)
4. Supporter **HDBSCAN et K-Means** avec paramétrage utilisateur
5. Proposer un **drill-down temporel** : stations → window embeddings par année/saison
6. Pré-calculer les coordonnées UMAP 2D/3D nightly pour un affichage instantané

## 3. Architecture

### 3.1 Navigation — Page Laboratoire

**TopNav** : Une seule entrée "Laboratoire" (icône `FlaskConical`) remplace les entrées séparées Counterfactual et Pumping Detection.

**Routes (nested, React Router 7)** :
```
/lab                    → Navigate redirect vers /lab/latent-space
/lab/latent-space       → LatentSpacePage (nouveau)
/lab/counterfactual     → CounterfactualPage (déplacé depuis /counterfactual)
/lab/pumping-detection  → PumpingDetectionPage (déplacé depuis /pumping-detection)
```

**LabLayout.tsx** : Wrapper avec barre d'onglets horizontale (`border-b-2 border-accent-cyan` pour l'actif). Contient un `<Outlet />` pour le contenu.

**Redirects de rétrocompatibilité** :
```
/counterfactual       → /lab/counterfactual
/pumping-detection    → /lab/pumping-detection
```

### 3.2 Backend — API Espace Latent

**Router** : `api/routers/latent_space.py` — préfixe `/api/v1/latent-space/`

> **Note** : Toutes les requêtes SQL de ce router utilisent la connexion BRGM (`get_brgm_db()`), car les schémas `ml.*` et `gold.*` sont tous les deux sur brgm-postgres.

#### Endpoints

**`GET /stations/{domain}`** — Scatter initial (pré-calculé)
- `domain` : `piezo` | `hydro`
- Jointure SQL : `ml.{domain}_station_embeddings` JOIN `gold.dim_piezo_stations` + TME (via `gold.int_station_era5_mapping` ou `gold.stations_piezo_carte`)
- Réponse :
```json
{
  "stations": [{
    "id": "02062X0001",
    "umap_2d": [1.23, -0.45],
    "umap_3d": [1.23, -0.45, 0.78],
    "cluster_id": 3,
    "n_windows": 24,
    "metadata": {
      "milieu_eh": "Karstique",
      "theme_eh": "Sédimentaire",
      "etat_eh": "Libre seul",
      "nature_eh": "Système aquifère",
      "libelle_eh": "Craie du Sénonien",
      "departement": "45",
      "nom_departement": "Loiret",
      "region": "Centre-Val de Loire",
      "altitude": 120.0
    }
  }]
}
```

**`POST /windows/{domain}`** — Window embeddings pour drill-down
- Endpoint **interne** (appelé par `/compute`, pas directement par le frontend)
- Body :
```json
{
  "filters": {
    "station_ids": ["..."],
    "libelle_eh": "Craie du Sénonien",
    "milieu_eh": "Karstique",
    "theme_eh": "Sédimentaire",
    "etat_eh": "Libre seul",
    "departement": "45",
    "region": "Centre-Val de Loire",
    "cluster_id": 3
  },
  "year_min": 2018,
  "year_max": 2024,
  "season": "DJF"
}
```
- Tous les filtres sont optionnels, combinés en AND
- Retourne les embeddings 320d + metadata par fenêtre (station_id, window_start, window_end)
- **Les vecteurs 320d ne sont jamais envoyés au frontend** — cet endpoint est utilisé en interne par `/compute` qui retourne uniquement les coordonnées UMAP projetées + labels cluster

**`POST /compute`** — UMAP + clustering on-demand
- Body :
```json
{
  "domain": "piezo",
  "embeddings_type": "stations",
  "filters": {
    "station_ids": ["..."],
    "libelle_eh": "...",
    "milieu_eh": "...",
    "theme_eh": "...",
    "etat_eh": "...",
    "departement": "...",
    "region": "...",
    "cluster_id": 3
  },
  "year_min": 2018,
  "year_max": 2024,
  "season": "DJF",
  "umap": {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine"
  },
  "clustering": {
    "method": "hdbscan",
    "n_umap_dims": 10,
    "hdbscan": { "min_cluster_size": 10, "min_samples": 5 },
    "kmeans": { "n_clusters": 8 }
  }
}
```

**Pipeline de calcul** :
1. Charger les embeddings 320d depuis pgvector (filtrés selon `filters` + `year_min`/`year_max`/`season`)
2. **Cap à 15 000 points** : si le résultat dépasse, sous-échantillonnage aléatoire stratifié (par station) avec warning dans la réponse
3. UMAP 320d → `n_components` (2 ou 3) pour les coordonnées scatter
4. Si clustering HDBSCAN : UMAP 320d → `n_umap_dims` (défaut 10) → HDBSCAN
5. Si clustering K-Means : K-Means directement sur les coordonnées UMAP `n_components`
6. Retourne : coordonnées UMAP + labels cluster + metadata par point

- Temps estimé : 2-15s (stations ~5K), 10-30s (windows ~15K max)
- Timeout API : 120s pour cet endpoint

**`GET /similar/{domain}/{station_id}?k=10`** — K plus proches voisins
- Utilise l'index HNSW pgvector (instantané)
- Retourne les K stations les plus similaires avec distance cosinus

#### Backend utils

**`dashboard/utils/latent_space.py`** :
- `load_station_embeddings(domain, filters=None)` — query pgvector + join metadata gold
- `load_window_embeddings(domain, filters, year_min, year_max, season)` — query windows filtrées
- `compute_umap(embeddings, n_components, n_neighbors, min_dist, metric)` — appel umap-learn
- `compute_clustering(coords_or_embeddings, method, params)` — HDBSCAN ou KMeans
- `find_similar(domain, station_id, k)` — query HNSW cosine

**Schemas** : `api/schemas/latent_space.py` — Pydantic models pour request/response

### 3.3 Frontend — LatentSpacePage

#### Layout

```
┌──────────────────────────────────────────────────────┐
│ [Piézométrie | Hydrométrie]  switch domaine           │
├───────────┬──────────────────────────────────────────┤
│ FILTRES   │  SCATTER PLOT (Plotly Scattergl/Scatter3d)│
│           │                                           │
│ Nappe ▼   │                                           │
│ Milieu ▼  │                                           │
│ Thème ▼   │                                           │
│ État ▼    │                                           │
│ Dépt ▼    │                                           │
│ Région ▼  │                                           │
│ Cluster ▼ │                                           │
│           ├──────────────────────────────────────────┤
│ ────────  │  Couleur: [Cluster ▼]  Vue: [2D | 3D]    │
│ Couleur:  │  Niveau: [Stations | Windows]             │
│ [milieu▼] │                                           │
│           │  UMAP: neighbors=[15] min_dist=[0.1]      │
│ ────────  │  Clustering: [HDBSCAN ▼] min_cluster=[10] │
│ ACTIONS   │                                           │
│ Recentrer │  [Recalculer]                             │
│ Reset     │                                           │
└───────────┴──────────────────────────────────────────┘
```

#### Composants

| Composant | Rôle |
|-----------|------|
| `LatentSpacePage.tsx` | Orchestration, state management, layout |
| `EmbeddingScatter.tsx` | Plotly `Scattergl` (2D) / `Scatter3d` (3D), coloration dynamique |
| `FilterPanel.tsx` | Sidebar : dropdowns filtres + sélection attribut de couleur |
| `UMAPControls.tsx` | Params UMAP (n_neighbors, min_dist, n_umap_dims) + clustering (method, params) + bouton recalculer |
| `StationDetail.tsx` | Panneau KNN affiché au clic sur un point |

#### Hook `useLatentSpace.ts`

```typescript
useStationEmbeddings(domain: string)           // GET /stations/{domain} — pré-calculé
useWindowEmbeddings(domain: string, filters)   // POST /windows/{domain}
useComputeUMAP()                               // POST /compute — mutation
useSimilarStations(domain: string, id: string) // GET /similar/{domain}/{id}
```

#### Interactions

**Filtrage** :
- Les dropdowns filtrent les points. Mode **highlight** par défaut : points filtrés en couleur, reste en gris (opacité 15%).
- Bouton "Recentrer UMAP" : recalcule la projection uniquement sur la sélection filtrée via `POST /compute`.
- Bouton "Reset" : revient à la vue globale pré-calculée.

**Coloration** :
- Dropdown pour choisir l'attribut : `cluster_id` (défaut), `milieu_eh`, `theme_eh`, `etat_eh`, `nature_eh`, `libelle_eh` (top N + "Autre"), `departement`, `region`, `altitude` (échelle continue viridis).

**Vue 2D/3D** :
- 2D : `Scattergl` (WebGL, performant 5K+ points)
- 3D : `Scatter3d` avec rotation orbitale

**Drill-down temporal** (switch Stations → Windows) :
- Charge les window embeddings des stations filtrées
- UMAP recalculé on-demand sur ce sous-ensemble
- Colorations supplémentaires : **année**, **saison** (DJF/MAM/JJA/SON)
- Range slider année min/max

**Hover** : Station ID + nappe + attributs. En mode windows : + dates fenêtre.

**Clic** : Sélectionne un point → affiche panneau KNN (K plus proches voisins via HNSW).

### 3.4 Pré-calcul — Modifications DB & Dagster

#### Schema SQL

Nouvelles colonnes dans `ml.piezo_station_embeddings` et `ml.hydro_station_embeddings` :

```sql
ALTER TABLE ml.piezo_station_embeddings
    ADD COLUMN umap_2d_x FLOAT,
    ADD COLUMN umap_2d_y FLOAT,
    ADD COLUMN umap_3d_x FLOAT,
    ADD COLUMN umap_3d_y FLOAT,
    ADD COLUMN umap_3d_z FLOAT;

ALTER TABLE ml.hydro_station_embeddings
    ADD COLUMN umap_2d_x FLOAT,
    ADD COLUMN umap_2d_y FLOAT,
    ADD COLUMN umap_3d_x FLOAT,
    ADD COLUMN umap_3d_y FLOAT,
    ADD COLUMN umap_3d_z FLOAT;
```

#### Dagster asset modifié

`ml_{domain}_clusters` (existant dans hubeau_data_integration) :
1. Load station embeddings 320d (déjà fait)
2. UMAP 320d → 10d pour HDBSCAN (déjà fait)
3. HDBSCAN → cluster_id (déjà fait)
4. **Nouveau** : UMAP 320d → 2d (`n_neighbors=15, min_dist=0.1, metric=cosine`)
5. **Nouveau** : UMAP 320d → 3d (mêmes hyperparamètres)
6. UPDATE colonnes umap_2d_x/y, umap_3d_x/y/z, cluster_id

Les window embeddings ne sont PAS pré-projetées (volume trop important, projection dépend du sous-ensemble).

### 3.5 Dépendances

Ajouts au `pyproject.toml` de time-serie-explo :
- `umap-learn>=0.5.0`

Pour HDBSCAN, utiliser `sklearn.cluster.HDBSCAN` (disponible depuis scikit-learn 1.3, déjà en dépendance) plutôt que le package standalone `hdbscan` — évite une dépendance supplémentaire.

### 3.6 Métadonnées par domaine

**Piézométrie** : Attributs TME complets — `milieu_eh`, `theme_eh`, `etat_eh`, `nature_eh`, `libelle_eh` + géographie (`departement`, `region`, `altitude`). JOIN via `gold.dim_piezo_stations` + `gold.int_station_era5_mapping`.

**Hydrométrie** : Pas d'attributs TME (les entités hydrogéologiques ne s'appliquent qu'aux nappes). Attributs disponibles : `nom_cours_eau`, `code_departement`, `nom_departement`, `statut_station`. JOIN via `gold.dim_hydro_stations`. Le panneau de filtres s'adapte au domaine sélectionné (moins de filtres en hydro).

### 3.7 États d'erreur et edge cases

- **BRGM inaccessible** : Message "Base de données BRGM indisponible" avec retry
- **Aucun résultat après filtrage** : État vide "Aucune station ne correspond aux filtres"
- **UMAP échoue** (trop peu de points pour `n_neighbors`) : Erreur 422 avec message explicatif, le frontend affiche un toast
- **Chargement initial** : Skeleton loader pendant le `GET /stations/{domain}`
- **Compute long** : Spinner avec texte "Calcul UMAP en cours..." sur le scatter
- **Sous-échantillonnage** : Badge warning "15 000 / 42 000 points (sous-échantillonné)" quand le cap est atteint
- **`libelle_eh` coloration** : Top 12 valeurs les plus fréquentes + catégorie "Autre", groupement fait côté serveur

## 4. Hors périmètre

- Carte géographique des stations
- Export des clusters / résultats
- Comparaison piezo vs hydro sur le même scatter (espaces latents incompatibles)
- Pré-calcul UMAP sur les window embeddings
- Modification de l'encodeur SoftCLT / re-training
- Données `tendance_classification` / `niveau_alerte` (non disponibles)

## 5. Fichiers impactés

### Nouveaux fichiers
- `frontend/src/pages/LabLayout.tsx`
- `frontend/src/pages/LatentSpacePage.tsx`
- `frontend/src/components/latent-space/EmbeddingScatter.tsx`
- `frontend/src/components/latent-space/FilterPanel.tsx`
- `frontend/src/components/latent-space/UMAPControls.tsx`
- `frontend/src/components/latent-space/StationDetail.tsx`
- `frontend/src/hooks/useLatentSpace.ts`
- `api/routers/latent_space.py`
- `api/schemas/latent_space.py`
- `dashboard/utils/latent_space.py`

### Fichiers modifiés
- `frontend/src/routes.tsx` — nested routes `/lab/*`
- `frontend/src/components/layout/TopNav.tsx` — remplacer Counterfactual + Pumping par "Laboratoire"
- `frontend/src/lib/api.ts` — namespace `latentSpace`
- `api/main.py` — include latent_space router
- `pyproject.toml` — dépendances umap-learn, hdbscan

### Fichiers modifiés (hubeau_data_integration)
- `docker/postgres/init.sql` — colonnes umap_2d/3d
- `src/hubeau_pipeline/ml/latent_space/persistence.py` — persist umap coords
- `src/hubeau_pipeline/assets/ml_assets.py` — compute umap 2d/3d dans cluster asset
