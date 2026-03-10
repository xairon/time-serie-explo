# Plan : Espace Latent pour Stations Piézométriques

> **Date** : 2026-03-10
> **Statut** : Validé
> **Méthode** : TS2Vec (apprentissage contrastif)
> **Périmètre** : 500+ stations depuis PostgreSQL, embeddings dim=320, stockage pgvector

---

## 1. Contexte et objectifs

### Problème

Les chroniques piézométriques de nappes phréatiques sont analysées station par station. Il n'existe pas de représentation vectorielle permettant de comparer, regrouper ou transférer des connaissances entre stations à l'échelle d'un réseau de 500+ piézomètres.

### Objectifs

| Objectif | Description | Priorité |
|----------|-------------|----------|
| **Clustering typologique** | Grouper les stations par comportement hydrodynamique (réactif/inertiel, libre/captif, alluvial/karstique) | P0 |
| **Enrichissement forecasting** | Utiliser les embeddings comme features pour améliorer les prévisions existantes (TFT, LSTM...) | P0 |
| **Recherche par similarité** | Trouver les k stations les plus proches dans l'espace latent | P1 |
| **Détection d'anomalies** | Identifier les stations dont l'embedding dérive (changement de régime, capteur défaillant) | P1 |
| **Détection de changement de régime** | Trajectoires temporelles dans l'espace latent | P2 |

### Non-objectifs (v1)

- Génération synthétique de chroniques (nécessiterait un VAE/décodeur)
- Interpolation spatiale de niveaux piézométriques
- Interface Streamlit (refactoring React+FastAPI en cours)

---

## 2. Choix technique : TS2Vec

### Pourquoi TS2Vec

TS2Vec (Yue et al., 2022) est un framework d'apprentissage de représentations pour séries temporelles basé sur l'apprentissage contrastif hiérarchique. Il a été retenu pour les raisons suivantes :

| Critère | TS2Vec | VRAE (alternative) | TFT extract (alternative) |
|---------|--------|---------------------|---------------------------|
| Labels nécessaires | Non | Non | Oui (forecasting) |
| Multivarié natif | Oui | Oui | Oui |
| Multi-granularité (fenêtre + instance) | Oui | Non | Non |
| Scalabilité (500+ stations) | Excellente (batch) | Moyenne | Limitée |
| Compatibilité PyTorch | Native | Native | Via Darts |
| Génération synthétique | Non | Oui | Non |
| Effort d'intégration | Moyen | Élevé | Faible |

### Principe

TS2Vec apprend des représentations en maximisant la concordance entre deux vues augmentées d'une même série temporelle, à deux niveaux :

1. **Niveau instance** : cohérence entre sous-séries de la même fenêtre
2. **Niveau temporel** : cohérence entre timestamps adjacents

La perte contrastive hiérarchique opère sur des représentations à différentes échelles temporelles (via dilated convolutions), ce qui permet de capturer à la fois les patterns haute fréquence (précipitations) et basse fréquence (cycles saisonniers, tendances pluriannuelles).

### Références

- Yue, Z. et al. (2022). *TS2Vec: Towards Universal Representation of Time Series*. AAAI 2022.
- GitHub : [github.com/yuezhihan/ts2vec](https://github.com/yuezhihan/ts2vec)

---

## 3. Architecture du module

### Structure des fichiers

```
dashboard/utils/latent_space/
├── __init__.py              # Exports publics, __all__
├── types.py                 # TypedDict : EmbeddingResult, ClusterResult, SimilarityResult
├── encoder.py               # TS2VecEncoder : fit, encode, save/load, MLflow integration
├── clustering.py            # HDBSCAN clustering + métriques (silhouette, Davies-Bouldin)
├── similarity.py            # Recherche kNN via pgvector, distance cosine/euclidienne
├── aggregation.py           # Fenêtre → station : mean pooling, weighted pooling
├── persistence.py           # pgvector CRUD : store, fetch, update, search embeddings
└── viz.py                   # UMAP/t-SNE projection, export JSON pour frontend React
```

### Diagramme de flux

```
┌─────────────────────────────────────────────────────────────────┐
│                        PostgreSQL                                │
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────────┐  │
│  │ stations_ts   │  │ station_window_emb   │  │ station_emb   │  │
│  │ (séries brutes│  │ (pgvector, dim=320)  │  │ (agrégé,      │  │
│  │  journalières)│  │                      │  │  dim=320)     │  │
│  └──────┬───────┘  └──────────▲───────────┘  └──────▲────────┘  │
│         │                     │                      │           │
└─────────┼─────────────────────┼──────────────────────┼───────────┘
          │                     │                      │
          ▼                     │                      │
┌─────────────────┐    ┌───────┴────────┐    ┌────────┴────────┐
│ preprocessing.py │    │ persistence.py │    │ aggregation.py  │
│ (StandardScaler) │    │ (pgvector I/O) │    │ (mean pooling)  │
└────────┬────────┘    └───────▲────────┘    └────────▲────────┘
         │                     │                      │
         ▼                     │                      │
┌─────────────────┐    ┌───────┴────────┐    ┌────────┴────────┐
│   encoder.py     │───▶│  Embeddings    │───▶│  Embeddings     │
│   (TS2Vec)       │    │  par fenêtre   │    │  par station    │
│   fit() + encode │    │  (N×W, 320)    │    │  (N, 320)       │
└─────────────────┘    └────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────┐       ┌──────────────┐
                       │ clustering.py│       │ similarity.py│
                       │ (HDBSCAN)    │       │ (kNN pgvec)  │
                       └──────┬───────┘       └──────┬───────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────────────────────────────┐
                       │             viz.py                    │
                       │  UMAP/t-SNE → JSON pour React         │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │         FastAPI Endpoints             │
                       │  /api/latent-space/*                   │
                       └──────────────────────────────────────┘
```

---

## 4. Types de données

### types.py

```python
from typing import TypedDict, Optional, List, Dict
import numpy as np

class EmbeddingResult(TypedDict, total=False):
    station_id: str
    window_start: str                  # ISO date
    window_end: str                    # ISO date
    embedding: List[float]             # dim=320
    method: str                        # "ts2vec"
    model_run_id: str                  # MLflow run_id

class StationEmbedding(TypedDict, total=False):
    station_id: str
    embedding: List[float]             # dim=320, agrégé
    cluster_id: int
    cluster_label: str                 # ex: "reactif_alluvial"
    metadata: Dict                     # nappe, coords, profondeur...

class ClusterResult(TypedDict):
    n_clusters: int
    silhouette_score: float
    davies_bouldin_index: float
    labels: List[int]                  # cluster_id par station
    cluster_sizes: Dict[int, int]      # cluster_id → nb stations

class SimilarityResult(TypedDict):
    query_station: str
    neighbors: List[Dict]              # [{station_id, distance, metadata}]

class ProjectionResult(TypedDict):
    method: str                        # "umap" | "tsne"
    coordinates: List[Dict]            # [{station_id, x, y, cluster_id, metadata}]
    params: Dict                       # n_neighbors, min_dist, perplexity...
```

---

## 5. Composants détaillés

### 5.1 encoder.py — TS2VecEncoder

```python
class TS2VecEncoder:
    """Wrapper autour de TS2Vec pour l'encodage de séries piézométriques."""

    def __init__(
        self,
        input_dims: int = 4,           # niveau + temp + precip + evap
        embedding_dim: int = 320,
        hidden_dim: int = 320,
        depth: int = 10,               # champ réceptif = 2^10 = 1024 jours
        device: str = "auto",          # cpu/cuda/xpu
    ): ...

    def fit(
        self,
        train_series: List[np.ndarray],  # [(T_i, 4)] par station
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 16,
        early_stopping_patience: int = 20,
        mlflow_run_id: Optional[str] = None,
    ) -> "TS2VecEncoder": ...

    def encode(
        self,
        series: List[np.ndarray],
        window_size: int = 365,
        stride: int = 90,
        encoding_window: str = "full_series",  # ou "multiscale"
    ) -> Dict[str, np.ndarray]:
        """Retourne {station_id: (n_windows, 320)}"""
        ...

    def encode_full(
        self,
        series: np.ndarray,
    ) -> np.ndarray:
        """Encode une série complète → (1, 320). Pour embedding station."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "TS2VecEncoder": ...
```

**Hyperparamètres** :

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `embedding_dim` | 320 | Suffisant pour discriminer 500+ stations avec marge |
| `hidden_dim` | 320 | Aligné sur embedding_dim (standard TS2Vec) |
| `depth` | 10 | Champ réceptif de 1024 jours (~3 ans de contexte) |
| `window_size` | 365 jours | Un cycle hydrologique complet |
| `stride` | 90 jours | ~4 embeddings par an par station, overlap 75% |
| `batch_size` | 16 | Adapté à 500+ stations en mémoire |
| `lr` | 1e-3 | Default TS2Vec |
| `n_epochs` | 200 | Avec early stopping patience=20 |
| `input_dims` | 4 | niveau_nappe_eau + temperature_2m + total_precipitation + potential_evaporation |

### 5.2 clustering.py

```python
def cluster_embeddings(
    embeddings: np.ndarray,            # (N, 320)
    method: str = "hdbscan",           # ou "kmeans"
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> ClusterResult: ...

def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz."""
    ...

def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = range(2, 20),
) -> int:
    """Elbow method + silhouette pour k-means."""
    ...
```

**HDBSCAN** est préféré car :
- Pas besoin de spécifier k a priori
- Détecte les outliers naturellement (stations atypiques)
- Adapté aux clusters de densité variable (nappes alluviales denses vs. karst épars)

### 5.3 similarity.py

```python
def find_similar_stations(
    station_id: str,
    k: int = 10,
    metric: str = "cosine",            # ou "euclidean"
    db_engine: Engine = None,
) -> SimilarityResult:
    """Recherche kNN via pgvector."""
    ...

def compute_pairwise_distances(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Matrice de distance (N, N). Pour heatmap."""
    ...
```

### 5.4 aggregation.py

```python
def pool_windows_to_station(
    window_embeddings: np.ndarray,     # (n_windows, 320)
    method: str = "mean",              # "mean", "last", "weighted"
    weights: Optional[np.ndarray] = None,  # pour weighted
) -> np.ndarray:
    """(n_windows, 320) → (320,)"""
    ...

def compute_trajectory(
    window_embeddings: np.ndarray,     # (n_windows, 320)
    window_dates: List[str],
    projection_method: str = "umap",
) -> List[Dict]:
    """Retourne [{date, x, y}] pour visualisation de trajectoire."""
    ...
```

Deux niveaux de granularité :
- **Fenêtre** : un embedding par fenêtre glissante de 365j → trajectoires temporelles, détection de régime
- **Station** : agrégation des fenêtres → clustering global, similarité entre stations

### 5.5 persistence.py

```python
class EmbeddingStore:
    """Interface pgvector pour le stockage et la recherche d'embeddings."""

    def __init__(self, engine: Engine): ...

    def init_tables(self) -> None:
        """Crée les tables et index pgvector si inexistants."""
        ...

    def store_window_embeddings(
        self,
        station_id: str,
        embeddings: np.ndarray,
        window_dates: List[Tuple[str, str]],
        method: str = "ts2vec",
        model_run_id: str = None,
    ) -> int:
        """Stocke les embeddings fenêtre. Retourne nb inséré."""
        ...

    def store_station_embedding(
        self,
        station_id: str,
        embedding: np.ndarray,
        cluster_id: int = None,
        cluster_label: str = None,
        method: str = "ts2vec",
        model_run_id: str = None,
    ) -> None: ...

    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        metric: str = "cosine",
    ) -> List[Dict]: ...

    def get_station_embedding(self, station_id: str) -> Optional[np.ndarray]: ...
    def get_all_embeddings(self, method: str = "ts2vec") -> Dict[str, np.ndarray]: ...
    def get_window_trajectory(self, station_id: str) -> List[Dict]: ...
```

### 5.6 viz.py

```python
def project_embeddings(
    embeddings: np.ndarray,            # (N, 320)
    method: str = "umap",              # "umap" | "tsne"
    n_components: int = 2,
    n_neighbors: int = 15,             # UMAP
    min_dist: float = 0.1,             # UMAP
    perplexity: int = 30,              # t-SNE
) -> np.ndarray:
    """(N, 320) → (N, 2). Projection 2D."""
    ...

def build_projection_payload(
    station_ids: List[str],
    projections: np.ndarray,           # (N, 2)
    cluster_labels: np.ndarray,
    metadata: Dict[str, Dict],         # station_id → {nappe, coords, ...}
) -> ProjectionResult:
    """Construit le JSON pour le frontend React."""
    ...

def build_trajectory_payload(
    window_projections: np.ndarray,    # (n_windows, 2)
    window_dates: List[str],
    station_id: str,
) -> List[Dict]:
    """[{date, x, y}] pour animation de trajectoire."""
    ...
```

---

## 6. Schéma PostgreSQL

```sql
-- Prérequis : extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings par fenêtre temporelle
CREATE TABLE station_window_embeddings (
    id              SERIAL PRIMARY KEY,
    station_id      TEXT NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    embedding       vector(320),
    method          TEXT NOT NULL DEFAULT 'ts2vec',
    model_run_id    TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(station_id, window_start, method)
);

-- Embeddings agrégés par station
CREATE TABLE station_embeddings (
    station_id      TEXT PRIMARY KEY,
    embedding       vector(320),
    cluster_id      INT,
    cluster_label   TEXT,
    method          TEXT NOT NULL DEFAULT 'ts2vec',
    model_run_id    TEXT,
    n_windows       INT,               -- nb de fenêtres agrégées
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour recherche par similarité (cosine)
CREATE INDEX idx_window_emb_cosine
    ON station_window_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX idx_station_emb_cosine
    ON station_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20);

-- Index temporel pour trajectoires
CREATE INDEX idx_window_emb_station_date
    ON station_window_embeddings (station_id, window_start);
```

---

## 7. Endpoints FastAPI

### Router : `/api/latent-space/`

En cohérence avec l'architecture décrite dans `docs/PLAN_API.md` :

```
api/routers/latent_space.py   →  api/services/latent_space.py  →  dashboard/utils/latent_space/
```

| Méthode | Endpoint | Description | Async |
|---------|----------|-------------|-------|
| `POST` | `/train` | Lancer l'entraînement TS2Vec | Oui (SSE) |
| `GET` | `/status` | Statut de l'entraînement en cours | Non |
| `GET` | `/models` | Lister les modèles d'embedding (MLflow) | Non |
| `POST` | `/encode` | Encoder des stations → embeddings | Oui (SSE) |
| `GET` | `/embeddings` | Récupérer tous les embeddings + projection UMAP | Non |
| `GET` | `/embeddings/{station_id}` | Embedding d'une station + trajectoire | Non |
| `GET` | `/similar/{station_id}` | k stations les plus similaires | Non |
| `POST` | `/clusters/run` | Lancer un clustering HDBSCAN | Non |
| `GET` | `/clusters` | Résultats du dernier clustering | Non |
| `POST` | `/reduce` | Projection UMAP/t-SNE à la demande | Non |

### Schémas Pydantic

```python
# Requêtes
class TrainRequest(BaseModel):
    embedding_dim: int = 320
    depth: int = 10
    n_epochs: int = 200
    lr: float = 1e-3
    window_size: int = 365
    stride: int = 90
    station_ids: Optional[List[str]] = None  # None = toutes

class EncodeRequest(BaseModel):
    model_run_id: str                        # MLflow run_id du modèle à utiliser
    station_ids: Optional[List[str]] = None
    window_size: int = 365
    stride: int = 90

class ClusterRequest(BaseModel):
    method: str = "hdbscan"                  # "hdbscan" | "kmeans"
    min_cluster_size: int = 5
    n_clusters: Optional[int] = None         # pour kmeans

class ReduceRequest(BaseModel):
    method: str = "umap"                     # "umap" | "tsne"
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1

# Réponses
class EmbeddingsResponse(BaseModel):
    stations: List[StationEmbeddingSchema]
    clustering_metrics: Optional[ClusterMetricsSchema]
    projection_method: str
    model_run_id: str

class SimilarResponse(BaseModel):
    query_station: str
    neighbors: List[NeighborSchema]          # station_id, distance, metadata

class TrajectoryResponse(BaseModel):
    station_id: str
    points: List[TrajectoryPointSchema]      # date, x, y
```

---

## 8. Intégration MLflow

Chaque entraînement TS2Vec crée un run MLflow dans un experiment dédié `latent-space` :

**Params loggés** :
- `embedding_dim`, `hidden_dim`, `depth`, `n_epochs`, `lr`, `batch_size`
- `window_size`, `stride`, `n_stations`, `input_dims`
- `method` = "ts2vec"

**Metrics loggées** :
- `training_loss` (par epoch)
- `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`
- `n_clusters` (HDBSCAN)
- `wall_clock_seconds`

**Artifacts** :
- `model/` : poids du modèle TS2Vec (pour reload)
- `embeddings.npz` : embeddings station + fenêtre
- `clusters.json` : labels de clustering
- `projection.json` : coordonnées UMAP/t-SNE

---

## 9. Intégration Docker

### Modification docker-compose.yml

```yaml
# Remplacer postgres:15-alpine par l'image pgvector
postgres:
  image: pgvector/pgvector:pg15
  # ... reste identique
```

### Nouvelle dépendance pyproject.toml

```toml
[project]
dependencies = [
    # ... existantes ...
    "pgvector>=0.3.0",         # Python client pgvector
    "hdbscan>=0.8.33",         # Clustering
    "umap-learn>=0.5.0",       # Projection UMAP
    # ts2vec : à vendoriser ou installer depuis GitHub
]
```

### Stratégie TS2Vec

TS2Vec n'est pas sur PyPI. Deux options :
1. **Vendoriser** : copier les fichiers sources dans `dashboard/utils/latent_space/ts2vec/` (recommandé pour la stabilité)
2. **pip install git+** : `pip install git+https://github.com/yuezhihan/ts2vec.git`

Option 1 recommandée car TS2Vec est compact (~5 fichiers Python) et évite une dépendance Git à la construction Docker.

---

## 10. Protocole d'entraînement détaillé

### Étape 1 : Chargement des données

```python
# Charger toutes les stations depuis PostgreSQL
from dashboard.utils.postgres_connector import fetch_data

stations_data = {}
for station_id in all_station_ids:
    df = fetch_data(
        engine=engine,
        table="piezometric_data",
        columns=["date", "niveau_nappe_eau", "temperature_2m",
                 "total_precipitation", "potential_evaporation"],
        filters={"station_id": station_id},
        date_range=(start_date, end_date),
    )
    stations_data[station_id] = df
```

### Étape 2 : Prétraitement

```python
from dashboard.utils.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor(
    missing_values="interpolate",
    normalization="standard",
)

# Fit sur l'ensemble d'entraînement (toutes stations confondues)
all_train = concat([s[s.date < train_end] for s in stations_data.values()])
preprocessor.fit(all_train)

# Transformer chaque station
processed = {
    sid: preprocessor.transform(df).values  # → np.ndarray (T, 4)
    for sid, df in stations_data.items()
}
```

### Étape 3 : Entraînement TS2Vec

```python
from dashboard.utils.latent_space.encoder import TS2VecEncoder

encoder = TS2VecEncoder(
    input_dims=4,
    embedding_dim=320,
    hidden_dim=320,
    depth=10,
)

# Entraînement sur les portions train de toutes les stations
train_arrays = [s[:train_end_idx] for s in processed.values()]
encoder.fit(
    train_series=train_arrays,
    n_epochs=200,
    lr=1e-3,
    batch_size=16,
    early_stopping_patience=20,
    mlflow_run_id=run_id,
)
```

### Étape 4 : Encodage

```python
# Embeddings par fenêtre
window_embeddings = encoder.encode(
    series=list(processed.values()),
    window_size=365,
    stride=90,
)
# → {station_id: (n_windows, 320)}

# Embeddings par station (agrégation)
from dashboard.utils.latent_space.aggregation import pool_windows_to_station

station_embeddings = {
    sid: pool_windows_to_station(embs, method="mean")
    for sid, embs in window_embeddings.items()
}
# → {station_id: (320,)}
```

### Étape 5 : Clustering

```python
from dashboard.utils.latent_space.clustering import cluster_embeddings

emb_matrix = np.stack(list(station_embeddings.values()))
result = cluster_embeddings(emb_matrix, method="hdbscan", min_cluster_size=5)
# → ClusterResult(n_clusters=8, silhouette=0.72, ...)
```

### Étape 6 : Stockage pgvector

```python
from dashboard.utils.latent_space.persistence import EmbeddingStore

store = EmbeddingStore(engine)
store.init_tables()

for sid, embs in window_embeddings.items():
    store.store_window_embeddings(sid, embs, window_dates, model_run_id=run_id)

for sid, emb in station_embeddings.items():
    store.store_station_embedding(
        sid, emb,
        cluster_id=result.labels[idx],
        model_run_id=run_id,
    )
```

---

## 11. Cas d'usage downstream

### 11.1 Enrichissement du forecasting existant

Les embeddings station peuvent servir de **static covariates** dans les modèles Darts :

```python
from darts import TimeSeries

# Ajouter l'embedding comme attribut statique
series = series.with_static_covariates(
    pd.DataFrame([station_embeddings[station_id]], columns=[f"emb_{i}" for i in range(320)])
)

# Les modèles TFT, TiDE, TSMixer supportent nativement les static covariates
model = TFTModel(...)
model.fit(series=train, past_covariates=cov)
```

### 11.2 Transfer learning

```python
# Trouver les 5 stations les plus similaires à une station pauvre en données
neighbors = store.search_similar(station_embeddings["BSS_PAUVRE"], k=5)

# Pré-entraîner le modèle de forecasting sur les voisins riches en données
rich_series = [load_series(n["station_id"]) for n in neighbors]
model.fit(series=rich_series, ...)

# Fine-tuner sur la station cible
model.fit(series=[target_series], epochs=20, lr=1e-4)
```

### 11.3 Détection de changement de régime

```python
# Calculer la trajectoire d'une station dans l'espace latent
trajectory = store.get_window_trajectory("BSS001")

# Détecter les ruptures via distance inter-fenêtres
from scipy.signal import find_peaks
distances = [np.linalg.norm(t[i+1] - t[i]) for i in range(len(t)-1)]
peaks, _ = find_peaks(distances, height=threshold)
# peaks = indices des changements de régime
```

---

## 12. Plan d'implémentation (séquence)

| Phase | Tâches | Dépendances | Estimation |
|-------|--------|-------------|------------|
| **Phase 0** | Vendoriser TS2Vec, ajouter dépendances (pgvector, hdbscan, umap-learn) | Aucune | — |
| **Phase 1** | `types.py` + `encoder.py` (fit/encode/save/load) | Phase 0 | — |
| **Phase 2** | `persistence.py` (pgvector CRUD, init_tables) + migration Docker pgvector | Phase 1 | — |
| **Phase 3** | `clustering.py` + `aggregation.py` + `similarity.py` | Phase 1 | — |
| **Phase 4** | `viz.py` (UMAP/t-SNE, export JSON) | Phase 3 | — |
| **Phase 5** | `__init__.py` + intégration MLflow | Phase 1-4 | — |
| **Phase 6** | FastAPI router + service (`api/routers/latent_space.py`) | Phase 5 + API existante | — |
| **Phase 7** | Tests (unitaires + intégration) | Phase 6 | — |
| **Phase 8** | Intégration forecasting (static covariates) | Phase 5 | — |

### Chemin critique

```
Phase 0 → Phase 1 → Phase 2 → Phase 5 → Phase 6
                   → Phase 3 → Phase 4 ↗
```

Les phases 2-3-4 sont parallélisables après la phase 1.

---

## 13. Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| TS2Vec ne scale pas bien sur 500+ stations longues | Performance | Batch training + fenêtrage (pas besoin de charger tout en mémoire) |
| pgvector pas dispo sur le PostgreSQL existant | Bloquant | Image Docker `pgvector/pgvector:pg15` (drop-in replacement) |
| Embeddings de dim 320 trop grands pour le clustering | Qualité | UMAP pré-réduction à 50 dims avant HDBSCAN si nécessaire |
| Séries de longueurs très différentes entre stations | Qualité | Padding/truncation + masquage dans TS2Vec |
| Données manquantes sur certaines stations | Qualité | Interpolation linéaire (déjà dans le preprocessing) + seuil minimum de couverture |

---

## 14. Métriques de succès

| Métrique | Seuil | Mesure |
|----------|-------|--------|
| Silhouette score du clustering | > 0.4 | Les clusters sont séparés et cohérents |
| Cohérence avec la géologie connue | > 70% | Les clusters correspondent aux types de nappes connus |
| Amélioration forecasting avec embeddings | > 5% RMSE | Les static covariates améliorent les prévisions |
| Temps d'encodage 500 stations | < 10 min | Scalabilité acceptable |
| Temps de recherche kNN (pgvector) | < 100ms | Temps réel pour l'API |
