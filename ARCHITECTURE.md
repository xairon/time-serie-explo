# Architecture PyTorch + Streamlit - Guide des Bonnes Pratiques

## Problème Résolu

Ce projet a été refactorisé pour éliminer les workarounds et patchs nécessaires pour charger les modèles PyTorch entraînés avec Streamlit. Le problème venait du fait que les callbacks Streamlit étaient sérialisés dans les fichiers de modèles, causant des erreurs au chargement.

## Architecture Standard

### Séparation des Responsabilités

L'architecture suit le principe de **séparation stricte** entre :
1. **Code d'entraînement** : Indépendant de toute interface graphique
2. **Code d'interface** : Lit les métriques depuis des fichiers partagés

### Composants Principaux

#### 1. Callbacks Standards (`core/callbacks.py`)

Les callbacks PyTorch Lightning sont **complètement indépendants** de Streamlit :

- **`MetricsFileCallback`** : Écrit les métriques dans un fichier JSON
- **`create_training_callbacks()`** : Factory pour créer des callbacks standards

**Caractéristiques** :
- ✅ Aucune dépendance à Streamlit
- ✅ Peut être utilisé dans n'importe quel contexte (CLI, backend, notebooks)
- ✅ Métriques écrites dans un format JSON standard
- ✅ Compatible avec tous les environnements Python

#### 2. Moniteur Streamlit (`dashboard/utils/training_monitor.py`)

Le moniteur lit les métriques depuis le fichier JSON et les affiche dans Streamlit :

- **`TrainingMonitor`** : Lit et parse le fichier JSON
- **`monitor_training_in_streamlit()`** : Helper pour l'intégration Streamlit

**Caractéristiques** :
- ✅ Séparé du processus d'entraînement
- ✅ Peut être utilisé pour monitorer n'importe quel entraînement
- ✅ Mise à jour en temps réel via lecture du fichier

#### 3. Pipeline d'Entraînement (`core/training.py`)

La fonction `run_training_pipeline()` utilise uniquement des callbacks standards :

```python
# NOUVEAU: Utiliser metrics_file au lieu de callbacks Streamlit
training_results = run_training_pipeline(
    model_name=selected_model,
    hyperparams=hyperparams,
    # ...
    metrics_file=metrics_file,  # Fichier JSON pour les métriques
    n_epochs=n_epochs,
    early_stopping_patience=early_stopping_patience,
    # ...
)
```

**Caractéristiques** :
- ✅ N'accepte plus de callbacks Streamlit via `pl_trainer_kwargs`
- ✅ Utilise uniquement des callbacks standards
- ✅ Nettoyage automatique des callbacks avant sauvegarde

#### 4. Sauvegarde Propre (`core/model_config.py`)

La fonction `save_model_with_data()` nettoie le modèle avant sauvegarde :

```python
# Nettoyer le modèle avant sauvegarde
cleaned_model = _clean_model_before_save(model)
cleaned_model.save(str(model_path))
```

**Caractéristiques** :
- ✅ Retire les callbacks non-standard avant sauvegarde
- ✅ Garantit qu'aucune référence Streamlit n'est sérialisée
- ✅ Modèles sauvegardés sont portables et robustes

## Flux de Données

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSUS D'ENTRAÎNEMENT                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PyTorch Lightning Trainer                                  │
│  ├── MetricsFileCallback → Écrit dans metrics.json         │
│  ├── EarlyStopping (standard)                              │
│  └── Autres callbacks standards                             │
│                                                              │
│  Modèle entraîné → Sauvegarde (sans références Streamlit)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (fichier JSON)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE STREAMLIT                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TrainingMonitor                                             │
│  ├── Lit metrics.json                                        │
│  ├── Parse les métriques                                     │
│  └── Affiche dans Streamlit                                 │
│                                                              │
│  Affichage en temps réel (barres, graphiques, métriques)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Format du Fichier JSON

Le fichier `metrics.json` contient :

```json
{
  "status": "training",
  "start_time": 1234567890.123,
  "current_epoch": 5,
  "total_epochs": 50,
  "train_losses": [0.5, 0.4, 0.35, 0.32, 0.30],
  "val_losses": [0.6, 0.5, 0.45, 0.42, 0.40],
  "epochs": [1, 2, 3, 4, 5],
  "elapsed_seconds": 120.5,
  "eta_seconds": 1080.0,
  "last_update": 1234568010.123
}
```

## Migration des Modèles Existants

Pour nettoyer les modèles existants qui contiennent des références Streamlit :

```bash
# Dry run (vérifier ce qui sera fait)
python scripts/migrate_models.py --checkpoints-dir checkpoints --dry-run

# Migration réelle
python scripts/migrate_models.py --checkpoints-dir checkpoints

# Migrer seulement un type de modèle
python scripts/migrate_models.py --checkpoints-dir checkpoints --model-type TFT
```

**Note** : La migration peut nécessiter une réentraînement si les callbacks sont profondément intégrés. Le script tente de nettoyer ce qui est possible.

## Bonnes Pratiques

### ✅ À FAIRE

1. **Utiliser uniquement des callbacks standards** dans `run_training_pipeline()`
2. **Passer `metrics_file`** au lieu de callbacks Streamlit
3. **Utiliser `TrainingMonitor`** pour afficher la progression dans Streamlit
4. **Nettoyer les modèles** avant sauvegarde (fait automatiquement)

### ❌ À ÉVITER

1. **Ne jamais passer de callbacks Streamlit** directement au Trainer
2. **Ne pas référencer Streamlit** dans le code d'entraînement
3. **Ne pas sérialiser des objets Streamlit** dans les modèles
4. **Ne pas utiliser `pl_trainer_kwargs` avec des callbacks personnalisés** (utiliser `metrics_file` à la place)

## Compatibilité

### Anciens Modèles

Les modèles existants qui contiennent des références Streamlit peuvent toujours être chargés via `robust_loader.py`, mais cette solution est **dépréciée** et devrait être remplacée par :

1. Migration des modèles (script fourni)
2. Réentraînement avec la nouvelle architecture

### Nouveaux Modèles

Tous les nouveaux modèles entraînés avec cette architecture sont :
- ✅ Portables (pas de dépendances Streamlit)
- ✅ Chargement standard (pas besoin de patchs)
- ✅ Compatibles avec tous les environnements Python
- ✅ Robustes et maintenables

## Exemple d'Utilisation

### Dans Streamlit

```python
import tempfile
from pathlib import Path
from dashboard.utils.training_monitor import TrainingMonitor

# Créer un fichier temporaire pour les métriques
metrics_file = Path(tempfile.gettempdir()) / "training_metrics.json"

# Lancer l'entraînement
training_results = run_training_pipeline(
    model_name="TFT",
    hyperparams={"n_epochs": 50},
    # ...
    metrics_file=metrics_file,
    n_epochs=50,
    early_stopping_patience=10
)

# Monitorer la progression
monitor = TrainingMonitor(metrics_file)
monitor.display_progress(
    progress_bar=st.progress(0),
    status_text=st.empty(),
    metrics_placeholder=st.container(),
    chart_placeholder=st.empty()
)
```

### Dans un Script CLI

```python
from core.callbacks import create_training_callbacks
from core.training import run_training_pipeline

# Créer les callbacks standards
metrics_file = Path("training_metrics.json")
callbacks = create_training_callbacks(
    metrics_file=metrics_file,
    total_epochs=50,
    early_stopping_patience=10
)

# Lancer l'entraînement (sans Streamlit)
results = run_training_pipeline(
    model_name="TFT",
    hyperparams={"n_epochs": 50},
    # ...
    metrics_file=metrics_file,
    n_epochs=50
)
```

## Conclusion

Cette architecture suit les **pratiques standards** de l'industrie pour PyTorch + Streamlit :

- ✅ Séparation claire des responsabilités
- ✅ Pas de dépendances circulaires
- ✅ Code d'entraînement réutilisable
- ✅ Modèles portables et robustes
- ✅ Monitoring flexible et extensible

Plus besoin de workarounds ou de patchs ! 🎉
