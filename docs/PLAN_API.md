# Plan : API-iser le projet Junon Time-Series

> **Dernière mise à jour** : 2026-02-20 — audit complet par 5 agents Claude Code
> **Version** : v2 (post-audit)

## Contexte

Le projet est une plateforme Streamlit de prévision piézométrique (4 pages). Un audit exhaustif de chaque fichier confirme que **91% du backend (`dashboard/utils/`) est pur Python** — 19 modules sur 25 n'importent pas Streamlit. Les 6 fichiers contaminés sont des cas simples (décorateurs `@st.cache_data` ou fonctions UI à déplacer).

Le travail consiste à :
1. Nettoyer les 6 fichiers contaminés (~2h)
2. Extraire ~1324 lignes de logique métier enfouie dans les pages Streamlit (~1 jour)
3. Créer le package `api/` (~20 fichiers) qui délègue à `dashboard/utils/`

---

## Résultat d'audit : État réel du code

### Modules 100% purs (aucune modification nécessaire)

| Module | Fonctions clés | Sérialisation API |
|--------|---------------|-------------------|
| `training.py` | `run_training_pipeline()`, `train_model()`, `evaluate_model()`, `calculate_metrics()` | Retourne `dict` → JSON direct |
| `callbacks.py` | `MetricsFileCallback`, `create_training_callbacks()` | `MetricsState.to_dict()` → JSON direct |
| `forecasting.py` | `generate_single_window_forecast()`, `generate_rolling_forecast()`, `generate_comparison_forecast()`, `generate_global_forecast()` | `TimeSeries` → `.to_dataframe().to_dict("records")` |
| `model_factory.py` | `ModelFactory.create_model()`, `get_device_info()` | `dict` → JSON direct |
| `preprocessing.py` | `TimeSeriesPreprocessor`, `split_train_val_test()`, `prepare_dataframe_for_darts()` | Darts objects → conversion nécessaire |
| `optuna_training.py` | `create_optuna_objective()`, `run_optuna_study()` | `Study.trials_dataframe().to_dict()` |
| `optuna_utils.py` | `plot_optuna_*()` (3 fonctions Plotly), `get_best_params()` | `go.Figure.to_json()` |
| `postgres_connector.py` | 14 fonctions (CRUD, schema, fetch) | `DataFrame` → `.to_dict("records")` |
| `dataset_registry.py` | `DatasetRegistry` (CRUD), `PreparedDataset.to_dict()` | JSON direct |
| `model_registry.py` | `ModelRegistry` (16 méthodes), `ModelEntry` | Pydantic schema nécessaire |
| `mlflow_client.py` | `MLflowManager` (singleton) | Interne, pas exposé |
| `robust_loader.py` | `load_model_safe()` | Interne, pas exposé |
| `models_config.py` | `get_model_info()`, `get_available_models()` | `dict` → JSON direct |
| `config.py` | Constantes, paths, device | Import lourd (torch + GPU init) — voir risques |
| `counterfactual/types.py` | `CounterfactualResult` (TypedDict) | Tensors → `.detach().cpu().tolist()` |
| `counterfactual/constants.py` | Constantes IPS, couleurs, presets | JSON direct |
| `counterfactual/perturbation.py` | `PerturbationLayer` (nn.Module) | `.to_interpretable()` → `dict` |
| `counterfactual/metrics.py` | 18 fonctions pures (`validity_ratio`, `proximity_theta`, `cc_compliance`, `build_paper_metrics`...) | `float`/`dict` → JSON direct (NaN → None) |
| `counterfactual/ips.py` | 15+ fonctions (`compute_ips_reference()`, `classify_prediction_monthly()`, `validate_ips_data()`...) | `dict`/`DataFrame` → conversion simple |
| `counterfactual/physcf_optim.py` | `generate_counterfactual()` | `CounterfactualResult` → serializer |
| `counterfactual/optuna_optim.py` | `generate_counterfactual_optuna()` | idem |
| `counterfactual/comte.py` | `generate_counterfactual_comte()` — CoMTE (Ates et al. 2021) | idem (feature swapping discret, prend `s_obs_norm` + `df_train_val` pour distracteurs) |
| `counterfactual/darts_adapter.py` | `DartsModelAdapter`, `StandaloneGRUAdapter` | Interne |
| `counterfactual/pastas_validation.py` | `PastasWrapper`, `validate_with_pastas()`, `run_dual_validation_for_results()` | `dict` → JSON (numpy → `.tolist()`) |
| `counterfactual/viz.py` | `plot_cf_overlay()`, `plot_theta_radar()`, `compute_seasonal_summary()`, `build_cf_export_df()` | `go.Figure.to_json()` ou data-only |
| `explainability/base.py` | `ExplainabilityResult.to_dict()`, `ModelType` | JSON direct via `.to_dict()` |
| `explainability/feature_importance.py` | `compute_correlation_importance()`, `compute_permutation_importance()`, `compute_shap_importance()` | `dict` (arrays → `.tolist()`) |
| `explainability/gradients.py` | `GradientExplainer`, `compute_gradient_attributions()` | `dict` (arrays → `.tolist()`) |
| `explainability/attention.py` | `TFTExplainer.explain()`, `compute_attention_summary()` | `dict` (drop `raw_result` key, arrays → `.tolist()`) |
| `explainability/decomposition.py` | `DecompositionAnalyzer`, `detect_seasonality_patterns()` | `dict` (pd.Series → `.tolist()`) |
| `explainability/model_specific.py` | `ModelExplainerFactory.get_explainer()` | `.to_dict()` intégré |
| `explainability/visualizations.py` | 10 fonctions `plot_*()` | `go.Figure.to_json()` |

### 6 fichiers contaminés (nettoyage requis)

| Fichier | Contamination | Occurrences | Sévérité | Action |
|---------|--------------|:-----------:|----------|--------|
| `plots.py` | `@st.cache_data` × 16 | 17 | **EASY** | Supprimer les 16 décorateurs + `import streamlit`. Fonctions retournent déjà des `go.Figure` purs. |
| `statistics.py` | `@st.cache_data` × 7 | 8 | **EASY** | Supprimer les 7 décorateurs + `import streamlit`. Fonctions retournent des `dict`/`tuple` purs. |
| `data_loader.py` | `@st.cache_data` × 1 sur `get_data_summary()` | 2 | **EASY** | Supprimer le décorateur + `import streamlit`. |
| `export.py` | `st.spinner`, `st.download_button`, `st.warning`, `st.error` dans `add_download_button()` | 5 | **MEDIUM** | Garder `create_model_archive()` (pur, retourne `bytes`). Déplacer `add_download_button()` vers `dashboard/components/`. |
| `training_monitor.py` | `st.session_state`, `st.rerun`, `st.columns`, `st.metric`, `@st.fragment` | 16 | **HARD** | Garder `TrainingMonitor.read_metrics()` (pur JSON reader). Déplacer `display_progress()` et `create_training_monitor_fragment()` vers page 2. |
| `state.py` | `st.session_state` × 9 | 10 | **HARD** | **Supprimer entièrement.** Remplacé par `TaskManager` côté API et `st.session_state` direct côté pages. |

**Contamination transitive : AUCUNE.** Aucun module propre n'importe un module contaminé.

---

## Architecture cible

```
api/                          <- NOUVEAU PACKAGE
  __init__.py
  main.py                     # FastAPI app, lifespan (init engine+mlflow), CORS
  settings.py                 # Pydantic BaseSettings (ports, URIs, clés) — PAS config.py
  routers/
    health.py                 # GET /health, GET /config
    datasets.py               # 7 endpoints CRUD + upload CSV + import DB
    training.py               # POST start, GET status, GET SSE stream, POST cancel, GET history
    models.py                 # GET list, GET details, GET download ZIP, DELETE
    forecasting.py            # POST single-window, rolling, comparison, global
    explainability.py         # POST feature-importance, attention, shap, summary
    counterfactual.py         # POST ips-reference, ips-classify, generate, generate-batch, pastas-validate
  schemas/                    # Pydantic models (request/response)
    common.py                 # Pagination, ErrorResponse, TaskStatus
    datasets.py               # DatasetCreate, DatasetResponse, PreviewResponse
    training.py               # TrainingRequest, TrainingStatus, SSEEvent
    models.py                 # ModelResponse (from ModelEntry), ModelDownload
    forecasting.py            # ForecastRequest, ForecastResponse, MetricsResponse
    explainability.py         # ExplainRequest, ExplainResponse (from ExplainabilityResult.to_dict())
    counterfactual.py         # CFRequest, CFResponse, IPSReference, PastasValidation
  services/
    task_manager.py           # Thread-safe registry (remplace st.session_state)
    serializers.py            # tensor→list, NaN→None, DataFrame→records, Figure→JSON
    dataset_service.py        # Upload, prepare, validate, data profiling
    training_service.py       # Thread management + SSE progress
    forecasting_service.py    # Model cache + wrap forecasting.py
    explainability_service.py # ModelExplainerFactory dispatch + background tasks
    counterfactual_service.py # CF orchestration + Pastas + background tasks
  sse/
    training_stream.py        # Async generator qui lit metrics.json (SSE)
  middleware/
    auth.py                   # API key optionnelle (header X-API-Key)
    error_handler.py          # Exception → JSONResponse
  tests/
    conftest.py               # TestClient, fixtures
    test_health.py
    test_datasets.py
    test_training.py
    test_models.py
    test_forecasting.py
    test_explainability.py
    test_counterfactual.py

dashboard/utils/              <- EXISTANT (partagé Streamlit + API)
  # Après nettoyage : 100% pur Python, zéro import Streamlit
```

**Principe** : `api/routers/` → `api/services/` → `dashboard/utils/` (aucune logique métier dans les routers)

**`api/settings.py` vs `dashboard/config.py`** : `settings.py` est léger (Pydantic `BaseSettings`, env vars). `config.py` reste pour les constantes ML (modèles, métriques, device) mais n'est importé qu'au besoin (lazy import) car il déclenche `import torch` + détection GPU au chargement.

---

## Endpoints (~29)

### Health & Config (2)
- `GET /api/v1/health` → device (via `get_device_info()`), mlflow_uri, postgres status, pastas_available
- `GET /api/v1/config` → modèles disponibles (via `get_available_models()`), métriques, variables, optuna_config

### Datasets (7)
- `POST /api/v1/datasets/upload` — Upload CSV + metadata (Form + File)
- `POST /api/v1/datasets/from-database` — Import depuis PostgreSQL (via `postgres_connector.fetch_data()`)
- `GET /api/v1/datasets` — Liste des datasets préparés (via `DatasetRegistry.scan_datasets()`)
- `GET /api/v1/datasets/{id}` — Détails (via `PreparedDataset.to_dict()`)
- `GET /api/v1/datasets/{id}/preview` — 50 premières lignes + stats (via `get_data_summary()`)
- `POST /api/v1/datasets/{id}/prepare` — Preprocess + save (via `prepare_station_data()` extrait)
- `DELETE /api/v1/datasets/{id}` — (via `DatasetRegistry.delete_dataset()`)

### Training (5)
- `POST /api/v1/training/start` → task_id (UUID) — lance `run_training_pipeline()` en thread
- `GET /api/v1/training/{task_id}/status` → epoch, losses, ETA (via `TrainingMonitor.read_metrics()`)
- `GET /api/v1/training/{task_id}/stream` → **SSE** (text/event-stream) — lit metrics.json async
- `POST /api/v1/training/{task_id}/cancel` — set `threading.Event`
- `GET /api/v1/training/history` — (via `MLflowManager.search_runs()`)

### Models (4)
- `GET /api/v1/models` — Liste (via `ModelRegistry.list_all_models()`)
- `GET /api/v1/models/{run_id}` — Détails + métriques (via `ModelRegistry.get_model()`)
- `GET /api/v1/models/{run_id}/download` → ZIP (via `create_model_archive()` + `StreamingResponse`)
- `DELETE /api/v1/models/{run_id}` — (via `ModelRegistry.delete_model()`)

### Forecasting (4)
- `POST /api/v1/forecasting/single-window` — (via `generate_single_window_forecast()`)
- `POST /api/v1/forecasting/rolling` — (via `generate_rolling_forecast()`)
- `POST /api/v1/forecasting/comparison` — (via `generate_comparison_forecast()`)
- `POST /api/v1/forecasting/global` — (via `generate_global_forecast()`)

### Explainability (4)
- `POST /api/v1/explainability/feature-importance` — (via `compute_correlation_importance()`, `compute_permutation_importance()`)
- `POST /api/v1/explainability/attention` — (via `TFTExplainer.explain()`, TFT uniquement)
- `POST /api/v1/explainability/shap` — (via `compute_shap_importance()`) — background task
- `POST /api/v1/explainability/summary` — (via `ModelExplainerFactory.get_explainer().explain_local().to_dict()`)

### Counterfactual (5)
- `POST /api/v1/counterfactual/ips-reference` — (via `compute_all_ips_references()`)
- `POST /api/v1/counterfactual/ips-classify` — (via `classify_prediction_monthly()`)
- `POST /api/v1/counterfactual/generate` — (via `generate_counterfactual()` / `_optuna()` / `_comte()`) — background task
- `POST /api/v1/counterfactual/generate-batch` — multi-méthode en parallèle — background task
- `POST /api/v1/counterfactual/pastas-validate` — (via `run_dual_validation_for_results()`) — background task

### Optuna (2 — bonus)
- `POST /api/v1/training/optuna/start` — (via `run_optuna_study()`) — background task + SSE
- `GET /api/v1/training/optuna/{task_id}/status`

---

## Extractions critiques (logique enfouie dans les pages Streamlit)

L'audit a identifié **~1324 lignes** de logique métier à extraire sur 5481 lignes totales (24%).

### Page 2 — Train Models (440 lignes, **priorité maximale**)

#### 1. `prepare_station_data()` → `dashboard/utils/preprocessing.py`
**Source** : `2_Train_Models.py` lignes 620-739 (~120 lignes)
Fonction pure qui : filtre par station, fill missing (4 méthodes), supprime doublons (`groupby.mean()`), convertit en Darts TimeSeries, split train/val/test, normalise via `TimeSeriesPreprocessor.fit_transform()`.
Remplacer `add_log()` par `logging.getLogger().info()`, `progress_bar.progress()` par `Optional[Callable[[float, str], None]]`.

#### 2. `run_training_session()` → `dashboard/utils/training_orchestrator.py`
**Source** : `2_Train_Models.py` lignes 871-1022 (~150 lignes)
Closure qui capture ~15 variables locales. Orchestre global vs independent training, gère le metrics JSON file entre stations, cleanup GPU.
Remplacer la closure par une classe `TrainingSession(config: TrainingSessionConfig)` avec méthode `.run()`.
Remplacer `_write_log_to_state(state_dict, ...)` par `TaskManager.update_task()`.
Ajouter paramètre `stop_event: Optional[threading.Event]` pour cancellation.

#### 3. `TrainingPhase` enum + state management → `dashboard/utils/training_orchestrator.py`
**Source** : `2_Train_Models.py` lignes 38-135 (~100 lignes)
Enum `IDLE/PREPARING/TRAINING/COMPLETED/ERROR`, fonctions `get_training_state()`, `reset_training_state()`, `add_log()`, `_write_log_to_state()`.

### Page 1 — Dataset Preparation (380 lignes)

#### 4. Data profiling → `dashboard/utils/data_profiler.py` (nouveau)
**Source** : `1_Dataset_Preparation.py` dans `render_exploration_tabs()` (~200 lignes)
- `get_schema(df) -> List[Dict]` — types, non-null, samples
- `compute_stats(df, cols) -> Dict` — `.describe()` + missing + zeros
- `compute_correlations(df, cols) -> Dict` — matrice corr + top-N pairs
- `missing_summary(df) -> Dict` — per-column missing count/pct/dtype
- `quality_metrics(df) -> Dict` — completeness %, duplicates, cols complete
- `detect_date_columns(df) -> List[str]` — heuristique keyword + parse

#### 5. Configuration heuristics → `dashboard/utils/preprocessing.py`
**Source** : `1_Dataset_Preparation.py` dans `render_configuration_ui()` (~100 lignes)
- `detect_date_column(df) -> Optional[str]`
- `detect_station_column(df, exclude) -> Optional[str]`
- `apply_date_filter(df, col, mode, ...) -> pd.DataFrame`
- `check_date_duplicates(df, date_col) -> Dict`

#### 6. Session state population → `DatasetConfig` dataclass
**Source** : `1_Dataset_Preparation.py` lignes 681-724 (~60 lignes)
Les 13 clés `st.session_state` (training_data, training_target_var, training_stations, etc.) deviennent les champs d'un `DatasetConfig` dataclass retourné par `DatasetService.validate_and_configure()`.

### Page 3 — Forecasting (280 lignes)

#### 7. Métriques hydro → `dashboard/utils/training.py` (ou `utils/metrics.py`)
**Source** : `3_Forecasting.py` lignes 102-172 (~70 lignes)
- `nash_sutcliffe_efficiency(actual, predicted) -> float`
- `kling_gupta_efficiency(actual, predicted) -> Dict[str, float]` (avec décomposition r, alpha, beta)

#### 8. `load_model_for_inference()` → `dashboard/utils/model_registry.py`
**Source** : `3_Forecasting.py` lignes 181-234 (~50 lignes)
Charge model + scalers + config + splits depuis MLflow. Reconstruit un duck-typed `config` object.
Remplacer `@st.cache_resource` par `functools.lru_cache(maxsize=10)`.

#### 9. SHAP compatibility patch → `dashboard/utils/timeshap_compat.py`
**Source** : `3_Forecasting.py` lignes 19-70 (~50 lignes)
Monkey-patch de `shap.explainers._kernel.Kernel` pour SHAP >= 0.43. Le fichier `timeshap_compat.py` existe déjà mais ne contient pas encore ce patch.

#### 10. Helpers data → `dashboard/utils/preprocessing.py`
**Source** : `3_Forecasting.py`
- `_merge_covariates(data_dict) -> Dict` (lignes 486-495)
- `generate_raw_from_processed(df, scalers, target_col) -> pd.DataFrame` (lignes 520-534)
- `compute_scale_stats(values) -> Dict` (lignes 562-574)

### Page 4 — Counterfactual Analysis (200 lignes, **mieux structurée**)

#### 11. Helpers CF → `dashboard/utils/counterfactual/`
**Source** : `4_Counterfactual_Analysis.py`
- `_detect_columns(config, data_dict) -> Tuple[str, List[str]]` (lignes 157-192)
- `_build_full_df(data_dict, target_col, cov_cols) -> pd.DataFrame` (lignes 195-221)
- `_build_physcf_scaler(mu, sigma, target_col, cov_params, cov_cols) -> Dict` (lignes 231-253)
- CF bounds computation (lignes 855-869) → `ips.py::compute_cf_bounds()`
- IPS transition info (lignes 780-793) → `ips.py::compute_transition_info()`
- LaTeX table generation (lignes 1254-1291) → `counterfactual/export.py`

### Home.py (24 lignes — trivial)

#### 12. `get_system_info()` → `dashboard/utils/model_factory.py`
**Source** : `Home.py` lignes 31-54
Déjà couvert par `get_device_info()` existant — étendre avec `python_version`, `torch_version`.

---

## Training Progress : SSE (pas WebSocket)

L'architecture actuelle utilise déjà un fichier JSON temporaire écrit par `MetricsFileCallback` (dans le thread de training) et lu par le main thread Streamlit. SSE est le mapping naturel :

```
Client <-- SSE ---- FastAPI ---- lit metrics.json ---- Thread Training
                                                          |
                                                    run_training_pipeline()
                                                    (dashboard/utils/training.py)
```

**Audit confirmé :** `MetricsState.to_dict()` retourne déjà un dict JSON-sérialisable avec `status`, `current_epoch`, `total_epochs`, `train_losses`, `val_losses`, `eta_seconds`. L'écriture est atomique (write-then-rename). Aucune transformation nécessaire côté SSE.

```python
# api/sse/training_stream.py
async def training_stream(task_id: str):
    metrics_file = task_manager.get_task(task_id)["metrics_file"]
    while True:
        data = json.loads(Path(metrics_file).read_text())
        yield f"data: {json.dumps(data)}\n\n"
        if data["status"] in {"completed", "error"}:
            break
        await asyncio.sleep(1.0)
```

- `POST /training/start` → crée thread + metrics_file + task_id
- `GET /training/{id}/stream` → async generator qui lit metrics.json toutes les 1s
- `POST /training/{id}/cancel` → set `threading.Event`

**Gap identifié :** `run_training_pipeline()` n'accepte pas de `stop_event`. L'annulation ne fonctionne qu'entre stations (multi-station), pas mid-epoch. Ajouter un paramètre `stop_event` + un callback PL qui check `stop_event.is_set()` à chaque epoch.

---

## TaskManager (remplace st.session_state)

```python
class TaskManager:
    """Thread-safe task registry pour utilisateurs concurrents."""
    _tasks: Dict[str, dict]    # task_id -> {status, thread, metrics_file, results, stop_event, ...}
    _lock: threading.Lock

    def create_task(task_type, config) -> str       # UUID
    def get_task(task_id) -> Optional[dict]
    def update_task(task_id, **kwargs)              # thread-safe
    def cancel_task(task_id) -> bool                # set stop_event
    def get_result(task_id) -> Optional[dict]       # poll or wait
    def cleanup_old_tasks(max_age_hours=24)
```

**Clés `st.session_state` à migrer :**

| Clé actuelle | Page | Destination TaskManager |
|-------------|------|------------------------|
| `training_state` (phase, results, logs) | Page 2 | `task.status`, `task.results`, `task.logs` |
| `_training_stop_event` | Page 2 | `task.stop_event` |
| `_training_thread` | Page 2 | `task.thread` |
| `cf_results_latest` | Page 4 | `task.results` (type=counterfactual) |
| `cf_context_latest` | Page 4 | `task.config` (type=counterfactual) |
| `cf_pastas_validation` | Page 4 | `task.results.pastas` |
| `window_pred_*` (avec éviction) | Page 3 | `task.cache` (LRU, maxsize=10) |

**Clés qui restent dans `st.session_state`** (UI-only, pas dans TaskManager) :
- Navigation : `forecasting_tab`, `explain_tab_index`
- Confirmations UI : `confirm_delete_*`
- Connexion DB : `db_engine`, `db_connected`, `db_connection_info`
- Configuration dataset : `training_data_configured`, `training_target_var`, etc.

---

## Sérialisation transversale

Un seul module `api/services/serializers.py` gère toutes les conversions :

```python
import math, numpy as np

def serialize_result(result: dict) -> dict:
    """Convertit récursivement tensors/arrays/NaN pour JSON."""
    out = {}
    for k, v in result.items():
        if hasattr(v, "detach"):          # torch.Tensor
            out[k] = v.detach().cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, dict):
            out[k] = serialize_result(v)
        elif isinstance(v, list):
            out[k] = [serialize_result(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out
```

**Cas spécifiques :**
- `CounterfactualResult` : champs `y_cf`, `s_cf_phys`, `s_cf_norm` sont `torch.Tensor`
- `ExplainabilityResult.to_dict()` : gère déjà les arrays numpy → `.tolist()`
- `attention.py` : supprimer la clé `raw_result` (objet Darts non sérialisable)
- `build_paper_metrics()` : peut retourner `float('nan')` → convertir en `None`
- `go.Figure` : retourner `fig.to_json()` ou `fig.to_dict()` (Plotly JSON)

---

## Tâches longues (background tasks)

Ces fonctions bloquent trop longtemps pour une réponse HTTP synchrone :

| Fonction | Temps typique | Pattern |
|----------|:------------:|---------|
| `run_training_pipeline()` | minutes → heures | `threading.Thread` + SSE |
| `run_optuna_study()` | minutes → heures | `threading.Thread` + SSE |
| `generate_counterfactual()` (PhysCF) | 10-60s | `asyncio.run_in_executor()` + poll |
| `generate_counterfactual_optuna()` | 30-120s | idem |
| `generate_counterfactual_comte()` (CoMTE) | 1-30s | idem (recherche exhaustive 2^3 × k) |
| `run_dual_validation_for_results()` (Pastas) | 10-120s | idem |
| `compute_shap_importance()` | 5-60s | idem |
| `compute_permutation_importance()` | 5-30s | idem |
| `compute_integrated_gradients()` | 1-30s (GPU: 1-5s) | idem ou sync si GPU |

**Pattern retenu :** `POST` retourne `202 Accepted` + `task_id`. `GET /status` poll le résultat. Pas de Celery nécessaire (serveur unique, `uvicorn --workers 1`).

---

## Nettoyage utils (6 fichiers, ~2h)

| Fichier | Action | Lignes impactées |
|---------|--------|:----------------:|
| `plots.py` | Supprimer `import streamlit as st` (L11) + 16× `@st.cache_data(ttl=3600)` | 17 |
| `statistics.py` | Supprimer `import streamlit as st` (L10) + 7× `@st.cache_data(ttl=3600)` | 8 |
| `data_loader.py` | Supprimer `import streamlit as st` (L9) + `@st.cache_data(ttl=3600)` (L279) | 2 |
| `export.py` | Déplacer `add_download_button()` (L38-67) vers `dashboard/components/sidebar/export_section.py`. Garder `create_model_archive()` (L10-35). | 5 |
| `training_monitor.py` | Extraire `display_progress()` et `create_training_monitor_fragment()` vers page 2. Garder la classe `TrainingMonitor` avec `read_metrics()`. | 16 |
| `state.py` | **Supprimer le fichier.** Seul import : `2_Train_Models.py:30` — remplacer par accès direct `st.session_state`. | 10 |
| `counterfactual/viz.py` | Fixer L265 : `.detach().numpy()` → `.detach().cpu().numpy()` (bug CUDA) | 1 |

---

## Phases d'implémentation (révisées post-audit)

| Phase | Scope | Jours estimés | Avec Claude Code | Dépendances |
|-------|-------|:------------:|:----------------:|-------------|
| **0** | Nettoyage utils (6 fichiers) + extractions critiques (1324 lignes) | 4 | **1.5** | — |
| **1** | Foundation API : `main.py`, `settings.py`, health, middleware, Docker | 2 | **0.5** | Phase 0 |
| **2** | Dataset Management : 7 endpoints + `data_profiler.py` | 3 | **1.5** | Phase 1 |
| **3** | Training Pipeline : TaskManager, SSE, start/cancel/status + Optuna | 4 | **2** | Phases 1+2 |
| **4** | Model Registry : 4 endpoints + download ZIP | 2 | **0.5** | Phase 1 |
| **5** | Forecasting : 4 endpoints + model cache | 3 | **1** | Phases 1+4 |
| **6** | Explainability : 4 endpoints + background tasks | 3 | **1.5** | Phases 1+4+5 |
| **7** | Counterfactual : 5 endpoints + background tasks + serializers | 3 | **1.5** | Phases 1+4+5 |
| **8** | Tests intégration + polish + Docker compose | 1 | **1** | Toutes |
| | **TOTAL** | **25 jours** | **~10 jours** | |

**Chemin critique** : Phase 0 → 1 → 2 → 3 → 8 = **5.5 jours minimum** (avec Claude Code)
**Phases parallélisables** : 4+5+6+7 (routeurs indépendants) via `superpowers:subagent-driven-development`

### Stratégie d'exécution optimale

```
Jour 1     : Phase 0 (nettoyage) + Phase 1 (foundation)
Jour 2-3   : Phase 2 (datasets) + Phase 4 (models) en parallèle
Jour 4-5   : Phase 3 (training + SSE) — chemin critique
Jour 6-7   : Phase 5 (forecasting) + Phase 6 (explainability) en parallèle
Jour 8-9   : Phase 7 (counterfactual)
Jour 10    : Phase 8 (tests + Docker + intégration)
```

---

## Nouvelles dépendances (pyproject.toml)

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "sse-starlette>=1.8.0",
    "pydantic-settings>=2.0.0",
]
```

---

## Docker (ajout service API)

Nouveau service dans `docker-compose.yml` :
```yaml
api:
  build:
    context: .
    dockerfile: docker/Dockerfile.api
  ports:
    - "49511:8000"
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
    - JUNON_API_KEY=${API_KEY:-}
    - POSTGRES_HOST=postgres
    - POSTGRES_PORT=5432
    - POSTGRES_USER=${POSTGRES_USER}
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    - POSTGRES_DB=${POSTGRES_DB}
  volumes:  # mêmes bind mounts que streamlit
    - ./data:/app/data:ro
    - ./checkpoints:/app/checkpoints
    - ./results:/app/results
    - ./figs:/app/figs
    - ./logs:/app/logs
  depends_on:
    postgres:
      condition: service_healthy
    mlflow:
      condition: service_started
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  networks:
    - default
    - hubeau_data_integration_default
```

Port `49511` choisi (suite de 49509 Streamlit, 49510 MLflow).

---

## Risques identifiés

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|:-----------:|------------|
| Régression pages Streamlit après extraction | Élevé | Moyenne | Les pages continuent d'appeler les mêmes fonctions depuis `utils/` au lieu d'inline. Tests manuels du workflow complet. |
| Import lourd de `config.py` (torch + GPU) | Moyen | Certaine | `api/settings.py` (Pydantic) pour les configs légères. Lazy-import `config` uniquement pour les endpoints ML. |
| Credentials hardcodées `db_presets.json` | Sécurité | Existant | Migrer vers env vars AVANT de créer l'API. Ne jamais lire `db_presets.json` côté API. |
| `MLflowManager` singleton en multi-worker | Faible | Faible | `uvicorn --workers 1` suffit pour un serveur unique. Chaque worker a sa propre instance. |
| Pastas optionnel | Faible | Faible | Endpoints CF retournent `503 {"detail": "pastas not installed"}` si `PASTAS_AVAILABLE=False`. |
| Cache modèles en mémoire (multi-Go) | Moyen | Moyenne | `lru_cache(maxsize=10)` avec TTL. Monitor la RAM via `/health`. |
| Annulation mid-epoch impossible | Faible | Existant | Phase 3 : ajouter `stop_event` à `run_training_pipeline()` + callback PL epoch-level. |
| `optuna_utils.py` duplique `optuna_training.py` | Faible | Existant | Consolider : `optuna_training.py` = orchestration, `optuna_utils.py` = visualisation/analyse. |

---

## Vérification

1. `pytest api/tests/` — tests unitaires par endpoint (`httpx.AsyncClient` + `TestClient`)
2. `docker compose up -d --build` — vérifier les 5 services (postgres, mlflow, streamlit, api)
3. `curl http://localhost:49511/api/v1/health` — healthcheck
4. `curl http://localhost:49511/api/v1/config` — config complète
5. Upload CSV → prepare → train → forecast → explain → CF : workflow complet via API
6. SSE training stream : `curl -N http://localhost:49511/api/v1/training/{id}/stream`
7. Cancellation : `curl -X POST http://localhost:49511/api/v1/training/{id}/cancel` pendant un entraînement actif
8. Background task CF : `POST /counterfactual/generate` → `202` → poll `GET /status`

---

## Fichiers critiques à modifier/créer

**Existants à modifier (Phase 0) :**
- `dashboard/utils/plots.py` — supprimer 16× `@st.cache_data`
- `dashboard/utils/statistics.py` — supprimer 7× `@st.cache_data`
- `dashboard/utils/data_loader.py` — supprimer 1× `@st.cache_data`
- `dashboard/utils/export.py` — séparer pure/UI
- `dashboard/utils/training_monitor.py` — séparer pure/UI
- `dashboard/utils/state.py` — supprimer
- `dashboard/utils/counterfactual/viz.py` — fix bug CUDA L265
- `dashboard/utils/preprocessing.py` — ajouter `prepare_station_data()`, `detect_date_column()`, helpers
- `dashboard/utils/training.py` — ajouter `nash_sutcliffe_efficiency()`, `kling_gupta_efficiency()`
- `dashboard/utils/timeshap_compat.py` — ajouter SHAP >= 0.43 kernel patch
- `dashboard/utils/model_registry.py` — ajouter `load_model_for_inference()`
- `pyproject.toml` — ajouter extra `api`
- `docker-compose.yml` — ajouter service `api`
- `.env` / `.env.example` — ajouter `API_KEY`, migrer DB credentials

**Nouveaux à créer :**
- `dashboard/utils/data_profiler.py` — fonctions de profiling extraites de page 1 (~200 lignes)
- `dashboard/utils/training_orchestrator.py` — `TrainingSession` extraite de page 2 (~250 lignes)
- `api/` — tout le package (~25 fichiers)
- `docker/Dockerfile.api` — image pour le service API
- `api/tests/` — tests d'intégration (~7 fichiers)
