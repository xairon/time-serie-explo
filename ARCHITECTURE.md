# Architecture PyTorch + Streamlit

## Architecture Standard

### Separation des Responsabilites

L'architecture suit le principe de **separation stricte** entre :
1. **Code d'entrainement** : Independant de toute interface graphique
2. **Code d'interface** : Lit les metriques depuis des fichiers partages

### Composants Principaux

#### 1. Callbacks Standards (`dashboard/utils/callbacks.py`)

Les callbacks PyTorch Lightning sont **completement independants** de Streamlit :

- **`MetricsFileCallback`** : Ecrit les metriques dans un fichier JSON
- **`create_training_callbacks()`** : Factory pour creer des callbacks standards

**Caracteristiques** :
- Aucune dependance a Streamlit
- Peut etre utilise dans n'importe quel contexte (CLI, backend, notebooks)
- Metriques ecrites dans un format JSON standard
- Compatible avec tous les environnements Python

#### 2. Moniteur Streamlit (`dashboard/utils/training_monitor.py`)

Le moniteur lit les metriques depuis le fichier JSON et les affiche dans Streamlit :

- **`TrainingMonitor`** : Lit et parse le fichier JSON
- **`monitor_training_in_streamlit()`** : Helper pour l'integration Streamlit

**Caracteristiques** :
- Separe du processus d'entrainement
- Mise a jour en temps reel via lecture du fichier

#### 3. Pipeline d'Entrainement (`dashboard/utils/training.py`)

La fonction `run_training_pipeline()` utilise des callbacks standards et le parametre `metrics_file` pour le suivi.

#### 4. Factory de Modeles (`dashboard/utils/model_factory.py`)

La classe `ModelFactory` permet d'instancier dynamiquement les modeles Darts avec validation des hyperparametres.

## Structure du Projet

```
time-serie-explo/
├── dashboard/
│   ├── training/              # Application Streamlit principale
│   │   ├── Home.py           # Point d'entree
│   │   └── pages/
│   │       ├── 1_Dataset_Preparation.py
│   │       ├── 2_Train_Models.py
│   │       └── 3_Forecasting.py
│   │
│   ├── utils/                 # Modules utilitaires
│   │   ├── callbacks.py       # Callbacks PyTorch Lightning
│   │   ├── training.py        # Pipeline d'entrainement
│   │   ├── training_monitor.py # Monitoring Streamlit
│   │   ├── model_factory.py   # Factory de modeles
│   │   ├── preprocessing.py   # Preprocessing des donnees
│   │   ├── optuna_training.py # Integration Optuna
│   │   ├── mlflow_client.py   # Client MLflow
│   │   ├── model_registry.py  # Registre des modeles
│   │   ├── robust_loader.py   # Chargement robuste des modeles
│   │   ├── data_loader.py     # Chargement des donnees
│   │   ├── forecasting.py     # Fonctions de prediction
│   │   ├── explainability.py  # TimeSHAP et explications
│   │   └── ...
│   │
│   ├── components/            # Composants UI reutilisables
│   │   ├── cards/            # Cartes de metriques
│   │   ├── charts/           # Graphiques
│   │   └── sidebar/          # Elements de sidebar
│   │
│   └── config.py             # Configuration globale
│
├── requirements/             # Dependances par architecture
│   ├── base.txt
│   ├── cpu.txt
│   ├── cuda.txt
│   └── xpu.txt
│
├── docs/                     # Documentation
│   └── RAPPORT_PIPELINE_ENTRAINEMENT.md
│
├── scripts/                  # Scripts utilitaires
│   └── reset_mlflow_db.ps1
│
├── run_app.py               # Lanceur de l'application
├── setup_env.py             # Setup automatique de l'environnement
├── verify_installation.py   # Verification de l'installation
├── pyproject.toml           # Configuration du projet (uv)
├── docker-compose.yml       # Deploiement Docker
└── Dockerfile
```

## Flux de Donnees

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSUS D'ENTRAINEMENT                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PyTorch Lightning Trainer                                  │
│  ├── MetricsFileCallback → Ecrit dans metrics.json         │
│  ├── EarlyStopping (standard)                              │
│  └── Autres callbacks standards                             │
│                                                              │
│  Modele entraine → Sauvegarde (sans references Streamlit)  │
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
│  ├── Parse les metriques                                     │
│  └── Affiche dans Streamlit                                 │
│                                                              │
│  Affichage en temps reel (barres, graphiques, metriques)    │
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

## Bonnes Pratiques

### A FAIRE

1. **Utiliser uniquement des callbacks standards** dans `run_training_pipeline()`
2. **Passer `metrics_file`** au lieu de callbacks Streamlit
3. **Utiliser `TrainingMonitor`** pour afficher la progression dans Streamlit
4. **Nettoyer les modeles** avant sauvegarde (fait automatiquement)

### A EVITER

1. **Ne jamais passer de callbacks Streamlit** directement au Trainer
2. **Ne pas referencer Streamlit** dans le code d'entrainement
3. **Ne pas serialiser des objets Streamlit** dans les modeles
