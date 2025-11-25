"""Configuration des modèles disponibles pour le forecasting."""

# Liste complète des modèles Darts supportés
# Divisés par catégorie pour faciliter la sélection

# =============================================================================
# MODÈLES DEEP LEARNING (Require PyTorch)
# =============================================================================
DL_MODELS = {
    # Transformer-based
    'TFT': {
        'name': 'Temporal Fusion Transformer',
        'class': 'TFTModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Transformer avec attention, interprétable, excellent pour séries complexes',
        'hyperparams': {
            'hidden_size': (16, 128, 64),
            'lstm_layers': (1, 3, 1),
            'num_attention_heads': (1, 8, 4),
            'dropout': (0.0, 0.3, 0.1),
            'hidden_continuous_size': (8, 32, 16)
        }
    },
    'Transformer': {
        'name': 'Transformer',
        'class': 'TransformerModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Architecture Transformer classique pour séries temporelles',
        'hyperparams': {
            'd_model': (32, 256, 128),
            'nhead': (2, 8, 4),
            'num_encoder_layers': (1, 4, 2),
            'num_decoder_layers': (1, 4, 2),
            'dim_feedforward': (128, 512, 256),
            'dropout': (0.0, 0.3, 0.1)
        }
    },

    # Specialized architectures
    'NBEATS': {
        'name': 'N-BEATS',
        'class': 'NBEATSModel',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Décomposition automatique trend/seasonality, très performant',
        'hyperparams': {
            'num_stacks': (10, 50, 30),
            'num_blocks': (1, 3, 1),
            'num_layers': (2, 5, 4),
            'layer_widths': (128, 512, 256)
        }
    },
    'NHiTS': {
        'name': 'N-HiTS',
        'class': 'NHiTSModel',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Evolution de N-BEATS, plus rapide et efficace',
        'hyperparams': {
            'num_stacks': (2, 5, 3),
            'num_blocks': (1, 3, 1),
            'num_layers': (2, 5, 2),
            'layer_widths': (256, 1024, 512),
            'pooling_kernel_sizes': None  # Tuple complexe
        }
    },

    # RNN-based
    'LSTM': {
        'name': 'LSTM',
        'class': 'RNNModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Long Short-Term Memory, baseline classique pour séries temporelles',
        'hyperparams': {
            'model': ['LSTM'],  # Fixed
            'hidden_dim': (32, 256, 128),
            'n_rnn_layers': (1, 3, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
    'GRU': {
        'name': 'GRU',
        'class': 'RNNModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Gated Recurrent Unit, plus rapide que LSTM',
        'hyperparams': {
            'model': ['GRU'],  # Fixed
            'hidden_dim': (32, 256, 128),
            'n_rnn_layers': (1, 3, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
    'BlockRNN': {
        'name': 'Block RNN',
        'class': 'BlockRNNModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'RNN optimisé pour prédictions multi-horizon',
        'hyperparams': {
            'model': ['LSTM', 'GRU'],
            'hidden_dim': (32, 256, 128),
            'n_rnn_layers': (1, 3, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },

    # CNN-based
    'TCN': {
        'name': 'Temporal Convolutional Network',
        'class': 'TCNModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Convolutions causales, rapide et performant',
        'hyperparams': {
            'num_filters': (16, 128, 64),
            'kernel_size': (2, 7, 3),
            'num_layers': (2, 6, 3),
            'dilation_base': (2, 4, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
    'TiDE': {
        'name': 'TiDE',
        'class': 'TiDEModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Time-series Dense Encoder, architecture MLP efficace',
        'hyperparams': {
            'num_encoder_layers': (1, 3, 1),
            'num_decoder_layers': (1, 3, 1),
            'decoder_output_dim': (8, 32, 16),
            'hidden_size': (128, 512, 256),
            'temporal_width_past': (2, 8, 4),
            'temporal_width_future': (2, 8, 4),
            'dropout': (0.0, 0.3, 0.1)
        }
    },

    # Specialized for long sequences
    'DLinear': {
        'name': 'DLinear',
        'class': 'DLinearModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Décomposition + Linear, très simple mais efficace',
        'hyperparams': {
            'kernel_size': (5, 51, 25),
            'shared_weights': [True, False]
        }
    },
    'NLinear': {
        'name': 'NLinear',
        'class': 'NLinearModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Normalization + Linear, baseline pour long-term forecasting',
        'hyperparams': {
            'shared_weights': [True, False],
            'const_init': [True, False]
        }
    },
}

# =============================================================================
# MODÈLES STATISTIQUES (No PyTorch required)
# =============================================================================
STATISTICAL_MODELS = {
    'ARIMA': {
        'name': 'ARIMA',
        'class': 'ARIMA',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'AutoRegressive Integrated Moving Average, classique statistique',
        'hyperparams': {
            'p': (0, 5, 1),
            'd': (0, 2, 1),
            'q': (0, 5, 1)
        }
    },
    'AutoARIMA': {
        'name': 'Auto-ARIMA',
        'class': 'AutoARIMA',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Sélection automatique des paramètres ARIMA',
        'hyperparams': {}  # Auto
    },
    'VARIMA': {
        'name': 'VARIMA',
        'class': 'VARIMA',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Vector ARIMA pour séries multivariées',
        'hyperparams': {
            'p': (1, 5, 1),
            'd': (0, 2, 0),
            'q': (1, 5, 1)
        }
    },
    'ExponentialSmoothing': {
        'name': 'Exponential Smoothing',
        'class': 'ExponentialSmoothing',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Lissage exponentiel (Holt-Winters)',
        'hyperparams': {
            'trend': [None, 'add', 'mul'],
            'seasonal': [None, 'add', 'mul'],
            'seasonal_periods': (7, 365, 30)
        }
    },
    'Theta': {
        'name': 'Theta',
        'class': 'Theta',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Méthode Theta, gagnante de M3 Competition',
        'hyperparams': {
            'theta': (1, 3, 2),
            'season_mode': ['multiplicative', 'additive']
        }
    },
    'FourTheta': {
        'name': 'FourTheta',
        'class': 'FourTheta',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Extension de Theta avec 4 composantes',
        'hyperparams': {}
    },
    'Prophet': {
        'name': 'Prophet (Facebook)',
        'class': 'Prophet',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': True,  # Regressors
        'multivariate': False,
        'description': 'Modèle additif de Facebook, gère bien la saisonnalité',
        'hyperparams': {
            'growth': ['linear', 'logistic'],
            'changepoint_prior_scale': (0.001, 0.5, 0.05),
            'seasonality_prior_scale': (0.01, 10.0, 1.0),
            'seasonality_mode': ['additive', 'multiplicative']
        }
    },
}

# =============================================================================
# MODÈLES ML (Sklearn-based)
# =============================================================================
ML_MODELS = {
    'RandomForest': {
        'name': 'Random Forest',
        'class': 'RandomForest',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': False,
        'description': 'Forêt aléatoire avec lags automatiques',
        'hyperparams': {
            'n_estimators': (50, 500, 200),
            'max_depth': (5, 30, 15),
            'lags': (7, 90, 30)
        }
    },
    'LightGBM': {
        'name': 'LightGBM',
        'class': 'LightGBMModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': False,
        'description': 'Gradient Boosting rapide et performant',
        'hyperparams': {
            'lags': (7, 90, 30),
            'lags_past_covariates': (7, 60, 14),
            'num_leaves': (20, 100, 31),
            'learning_rate': (0.01, 0.3, 0.1)
        }
    },
    'XGBoost': {
        'name': 'XGBoost',
        'class': 'XGBModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': False,
        'description': 'Gradient Boosting classique, très performant',
        'hyperparams': {
            'lags': (7, 90, 30),
            'lags_past_covariates': (7, 60, 14),
            'max_depth': (3, 10, 6),
            'learning_rate': (0.01, 0.3, 0.1),
            'n_estimators': (50, 500, 100)
        }
    },
    'CatBoost': {
        'name': 'CatBoost',
        'class': 'CatBoostModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': False,
        'description': 'Gradient Boosting de Yandex, robuste',
        'hyperparams': {
            'lags': (7, 90, 30),
            'iterations': (50, 500, 100),
            'depth': (3, 10, 6),
            'learning_rate': (0.01, 0.3, 0.03)
        }
    },
    'LinearRegression': {
        'name': 'Linear Regression',
        'class': 'LinearRegressionModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': False,
        'description': 'Régression linéaire simple, baseline rapide',
        'hyperparams': {
            'lags': (7, 90, 30),
            'lags_past_covariates': (7, 60, 14)
        }
    },
}

# =============================================================================
# ENSEMBLES
# =============================================================================
ENSEMBLE_MODELS = {
    'NaiveEnsemble': {
        'name': 'Naive Ensemble',
        'class': 'NaiveEnsembleModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'Moyenne simple de plusieurs modèles',
        'hyperparams': {}
    },
}

# =============================================================================
# MODÈLES SIMPLES (BASELINES)
# =============================================================================
BASELINE_MODELS = {
    'NaiveSeasonal': {
        'name': 'Naive Seasonal',
        'class': 'NaiveSeasonal',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Répète la même période (ex: même jour l\'an dernier)',
        'hyperparams': {
            'K': (1, 365, 7)
        }
    },
    'NaiveDrift': {
        'name': 'Naive Drift',
        'class': 'NaiveDrift',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Extrapolation linéaire du dernier trend',
        'hyperparams': {}
    },
    'NaiveMean': {
        'name': 'Naive Mean',
        'class': 'NaiveMean',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': False,
        'description': 'Prédit la moyenne historique',
        'hyperparams': {}
    },
}

# =============================================================================
# AGRÉGATION FINALE
# =============================================================================
ALL_MODELS = {
    **DL_MODELS,
    **STATISTICAL_MODELS,
    **ML_MODELS,
    **ENSEMBLE_MODELS,
    **BASELINE_MODELS
}

# Catégories pour UI
MODEL_CATEGORIES = {
    'Deep Learning': list(DL_MODELS.keys()),
    'Statistical': list(STATISTICAL_MODELS.keys()),
    'Machine Learning': list(ML_MODELS.keys()),
    'Ensemble': list(ENSEMBLE_MODELS.keys()),
    'Baselines': list(BASELINE_MODELS.keys())
}

# Modèles recommandés pour démarrer
RECOMMENDED_MODELS = ['TFT', 'NBEATS', 'LightGBM', 'Prophet', 'LSTM']

def get_model_info(model_name):
    """Retourne les informations d'un modèle."""
    return ALL_MODELS.get(model_name, None)

def get_models_by_capability(capability):
    """Filtre les modèles par capacité (covariates, multivariate, etc.)."""
    if capability == 'covariates':
        return [name for name, info in ALL_MODELS.items() if info['supports_covariates']]
    elif capability == 'multivariate':
        return [name for name, info in ALL_MODELS.items() if info['multivariate']]
    elif capability == 'past_covariates':
        return [name for name, info in ALL_MODELS.items() if info['supports_past_covariates']]
    elif capability == 'future_covariates':
        return [name for name, info in ALL_MODELS.items() if info['supports_future_covariates']]
    return []

def get_hyperparams_space(model_name):
    """Retourne l'espace de recherche des hyperparamètres."""
    model_info = get_model_info(model_name)
    if model_info:
        return model_info['hyperparams']
    return {}


# =============================================================================
# DESCRIPTIONS DES HYPERPARAMÈTRES
# =============================================================================
HYPERPARAM_DESCRIPTIONS = {
    # Hyperparamètres communs
    'input_chunk_length': 'Nombre de pas de temps en entrée (historique utilisé pour la prédiction). Plus grand = plus de contexte mais plus lent.',
    'output_chunk_length': 'Nombre de pas de temps à prédire. Définit l\'horizon de prévision.',
    'batch_size': 'Nombre d\'échantillons traités simultanément. Plus grand = entraînement plus rapide mais plus de mémoire GPU.',
    'n_epochs': 'Nombre de passes complètes sur l\'ensemble d\'entraînement. Plus élevé = meilleur apprentissage mais risque de surapprentissage.',
    'learning_rate': 'Taux d\'apprentissage de l\'optimiseur. Typiquement 0.001-0.01. Plus petit = convergence plus lente mais plus stable.',

    # Transformer / Attention-based
    'hidden_size': 'Taille des représentations internes du modèle. Plus grand = plus de capacité mais plus lent et risque de surapprentissage.',
    'lstm_layers': 'Nombre de couches LSTM empilées. Plus de couches = modèle plus profond capable de capturer des patterns complexes.',
    'num_attention_heads': 'Nombre de têtes d\'attention parallèles. Doit diviser hidden_size. Plus = capture de patterns plus variés.',
    'hidden_continuous_size': 'Taille des embeddings pour variables continues dans TFT. Généralement hidden_size/4.',
    'd_model': 'Dimension du modèle Transformer. Équivalent à hidden_size. Doit être divisible par nhead.',
    'nhead': 'Nombre de têtes d\'attention dans le Transformer standard.',
    'num_encoder_layers': 'Nombre de couches dans l\'encodeur. Plus = modèle plus profond.',
    'num_decoder_layers': 'Nombre de couches dans le décodeur. Plus = modèle plus profond.',
    'dim_feedforward': 'Dimension des couches feed-forward dans le Transformer. Généralement 2-4x d_model.',

    # RNN-based
    'hidden_dim': 'Dimension de l\'état caché du RNN/LSTM/GRU. Plus grand = plus de mémoire à long terme.',
    'n_rnn_layers': 'Nombre de couches RNN empilées. 2-3 couches généralement suffisant.',
    'model': 'Type de cellule RNN à utiliser (LSTM, GRU, etc.). LSTM = plus stable, GRU = plus rapide.',

    # N-BEATS / N-HiTS
    'num_stacks': 'Nombre de stacks dans N-BEATS. Chaque stack capture différentes échelles temporelles.',
    'num_blocks': 'Nombre de blocs par stack. Plus = modèle plus expressif.',
    'num_layers': 'Nombre de couches fully-connected par bloc.',
    'layer_widths': 'Largeur des couches fully-connected. Plus grand = plus de paramètres.',

    # TCN
    'num_filters': 'Nombre de filtres convolutionnels. Plus = plus de capacité de représentation.',
    'kernel_size': 'Taille du noyau de convolution. Plus grand = réceptive field plus large.',
    'dilation_base': 'Base pour la dilatation exponentielle. 2 est standard pour doubler à chaque couche.',

    # TiDE
    'decoder_output_dim': 'Dimension de sortie du décodeur TiDE.',
    'temporal_width_past': 'Largeur temporelle pour l\'encodage du passé.',
    'temporal_width_future': 'Largeur temporelle pour l\'encodage du futur.',

    # Linear models (DLinear, NLinear)
    'kernel_size': 'Taille du noyau pour la décomposition. Plus grand = saisonnalité plus longue.',
    'shared_weights': 'Partager les poids entre séries (True) ou poids séparés (False). True = moins de paramètres.',
    'const_init': 'Initialiser avec constantes (True) ou aléatoire (False).',

    # Regularization
    'dropout': 'Taux de dropout pour régularisation (0-0.5). 0.1-0.3 typique. Plus élevé = plus de régularisation.',

    # ARIMA
    'p': 'Ordre autorégressif (AR). Nombre de lags de la série utilisés.',
    'd': 'Ordre de différenciation. 1 pour séries non-stationnaires, 0 pour séries stationnaires.',
    'q': 'Ordre moyenne mobile (MA). Nombre de lags des erreurs de prédiction.',

    # Exponential Smoothing
    'trend': 'Type de trend: None, "add" (additif), "mul" (multiplicatif).',
    'seasonal': 'Type de saisonnalité: None, "add", "mul".',
    'seasonal_periods': 'Période de saisonnalité (ex: 7 pour hebdomadaire, 365 pour annuel).',

    # Theta
    'theta': 'Paramètre theta. 2 = classique, <2 = plus de trend, >2 = moins de trend.',
    'season_mode': 'Mode de saisonnalité: "additive" ou "multiplicative".',

    # Prophet
    'growth': 'Type de croissance: "linear" ou "logistic" (avec saturation).',
    'changepoint_prior_scale': 'Flexibilité du trend. Plus grand = trend plus flexible.',
    'seasonality_prior_scale': 'Force de la saisonnalité. Plus grand = saisonnalité plus forte.',
    'seasonality_mode': 'Mode de saisonnalité: "additive" ou "multiplicative".',

    # ML Models (Tree-based)
    'lags': 'Nombre de lags passés à utiliser comme features. Plus = plus de contexte temporel.',
    'lags_past_covariates': 'Nombre de lags des covariables passées à utiliser.',
    'n_estimators': 'Nombre d\'arbres dans la forêt/boosting. Plus = meilleure performance mais plus lent.',
    'max_depth': 'Profondeur maximale des arbres. Plus profond = modèle plus complexe, risque de surapprentissage.',
    'num_leaves': 'Nombre maximum de feuilles (LightGBM). Plus = modèle plus complexe.',
    'iterations': 'Nombre d\'itérations de boosting (CatBoost).',
    'depth': 'Profondeur des arbres (CatBoost).',

    # Baseline models
    'K': 'Période de répétition (NaiveSeasonal). Ex: 7 pour répéter même jour de la semaine.',
}


def get_hyperparam_description(param_name: str) -> str:
    """
    Retourne la description d'un hyperparamètre.

    Args:
        param_name: Nom de l'hyperparamètre

    Returns:
        Description textuelle de l'hyperparamètre
    """
    return HYPERPARAM_DESCRIPTIONS.get(param_name, "")
