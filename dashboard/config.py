"""Configuration centralisée pour le dashboard Junon."""

from pathlib import Path
import torch

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'piezos'
# MLflow : base SQLite au même niveau que le projet (run_app.py et mlflow ui utilisent ce chemin)
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
RESULTS_DIR = BASE_DIR / 'results'
FIGS_DIR = BASE_DIR / 'figs'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints' / 'darts'
LOGS_DIR = BASE_DIR / 'logs' / 'darts'

# Créer les répertoires s'ils n'existent pas
for dir_path in [RESULTS_DIR, FIGS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    try:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        elif not dir_path.is_dir():
            # Path exists but is not a directory (e.g., symlink or file)
            pass  # Skip, let it be handled elsewhere
    except (FileExistsError, OSError):
        # Directory already exists or permission issue, skip
        pass

# Liste des stations (dynamique)
if DATA_DIR.exists():
    # Récupérer tous les fichiers piezo*.csv
    files = list(DATA_DIR.glob("piezo*.csv"))
    # Extraire les noms de stations
    stations_list = [f.stem for f in files]
    
    # Essayer de trier numériquement
    try:
        stations_list.sort(key=lambda x: int(x.replace('piezo', '')))
    except ValueError:
        stations_list.sort()
        
    STATIONS = stations_list
else:
    STATIONS = []

# Variables disponibles
VARIABLES = ['level', 'PRELIQ_Q', 'T_Q', 'ETP_Q']
VARIABLE_NAMES = {
    'level': 'Niveau Piézométrique (m)',
    'PRELIQ_Q': 'Précipitation (mm)',
    'T_Q': 'Température (°C)',
    'ETP_Q': 'Évapotranspiration (mm)'
}

# Modèles disponibles
MODELS = ['NBEATS', 'TFT', 'TCN', 'LSTM']

# Couleurs par modèle
MODEL_COLORS = {
    'NBEATS': '#1f77b4',  # Bleu
    'TFT': '#ff7f0e',      # Orange
    'TCN': '#2ca02c',      # Vert
    'LSTM': '#d62728'      # Rouge
}

# Hyperparamètres par défaut
DEFAULT_HYPERPARAMS = {
    'input_chunk': 30,
    'output_chunk': 7,
    'batch_size': 32,
    'n_epochs': 50,
    'learning_rate': 1e-3
}

# Configuration Optuna
OPTUNA_CONFIG = {
    'n_trials': 50,
    'timeout': 3600,  # 1 heure
    'direction': 'minimize',  # Minimiser MAE
    'metric': 'MAE'
}

# Métriques
METRICS = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'WAPE', 'NRMSE', 'Dir_Acc', 'NSE', 'KGE']
METRICS_INFO = {
    'MAE': {'name': 'Mean Absolute Error', 'unit': 'm', 'lower_is_better': True},
    'RMSE': {'name': 'Root Mean Square Error', 'unit': 'm', 'lower_is_better': True},
    'MAPE': {'name': 'Mean Absolute Percentage Error', 'unit': '%', 'lower_is_better': True},
    'sMAPE': {'name': 'Symmetric MAPE', 'unit': '%', 'lower_is_better': True},
    'WAPE': {'name': 'Weighted Absolute Percentage Error', 'unit': '%', 'lower_is_better': True},
    'NRMSE': {'name': 'Normalized RMSE', 'unit': '0-1', 'lower_is_better': True},
    'Dir_Acc': {'name': 'Directional Accuracy', 'unit': '%', 'lower_is_better': False},
    'NSE': {'name': 'Nash-Sutcliffe Efficiency', 'unit': '-', 'lower_is_better': False},
    'KGE': {'name': 'Kling-Gupta Efficiency', 'unit': '-', 'lower_is_better': False}
}

# Device (CPU/GPU)
import os

# Configuration GPU
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

if torch.cuda.is_available():
    # Force utilisation GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    DEVICE = 'cuda'
    try:
        # Test simple pour vérifier que le GPU fonctionne
        test_tensor = torch.tensor([1.0]).cuda()
        del test_tensor
    except:
        DEVICE = 'cpu'
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = 'xpu'
    try:
        # Test simple pour vérifier que le XPU fonctionne
        test_tensor = torch.tensor([1.0]).to(DEVICE)
        del test_tensor
    except:
        DEVICE = 'cpu'
else:
    DEVICE = 'cpu'

# Random seed
RANDOM_SEED = 42

# Performance Configuration
PERFORMANCE_CONFIG = {
    'CACHE_TTL': 3600,  # 1 hour
    'MAX_PLOT_POINTS': 10000,
    'ENABLE_LAZY_LOADING': True,
    'ENABLE_DATA_DOWNSAMPLING': True
}
