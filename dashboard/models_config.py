"""Configuration des modèles disponibles pour le forecasting.

IMPORTANT: Seuls les modèles Deep Learning sont supportés car ils partagent
la même interface (input_chunk_length, output_chunk_length).
"""

# =============================================================================
# MODÈLES DEEP LEARNING (Seuls modèles supportés)
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
        'description': 'Transformer with attention, interpretable, excellent for complex time series',
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

    # Linear models (surprisingly effective)
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
    
    # MLP-Mixer based (State-of-the-art 2023)
    'TSMixer': {
        'name': 'TSMixer',
        'class': 'TSMixerModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': True,
        'multivariate': True,
        'description': 'MLP-Mixer pour time series, état de l\'art 2023',
        'hyperparams': {
            'hidden_size': (32, 256, 64),
            'ff_size': (64, 512, 128),
            'num_blocks': (1, 4, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
}

# =============================================================================
# GLOBAL BASELINES (Modèles simples mais compatibles avec l'interface)
# =============================================================================
BASELINE_MODELS = {
    'GlobalNaiveAggregate': {
        'name': 'Global Naive Aggregate',
        'class': 'GlobalNaiveAggregate',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Moyenne/médiane des valeurs passées. Baseline rapide.',
        'hyperparams': {}
    },
    'GlobalNaiveDrift': {
        'name': 'Global Naive Drift',
        'class': 'GlobalNaiveDrift',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Extrapolation linéaire du trend. Baseline.',
        'hyperparams': {}
    },
    'GlobalNaiveSeasonal': {
        'name': 'Global Naive Seasonal',
        'class': 'GlobalNaiveSeasonal',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Répète la saisonnalité passée. Baseline.',
        'hyperparams': {}
    },
}

# =============================================================================
# AGRÉGATION FINALE
# =============================================================================
ALL_MODELS = {**DL_MODELS, **BASELINE_MODELS}

# Catégories pour UI
MODEL_CATEGORIES = {
    'Deep Learning': list(DL_MODELS.keys()),
    'Baselines': list(BASELINE_MODELS.keys()),
}

# Modèles recommandés pour démarrer
RECOMMENDED_MODELS = ['TFT', 'NBEATS', 'NHiTS', 'TSMixer', 'LSTM']


def get_model_info(model_name):
    """Retourne les informations d'un modèle."""
    return ALL_MODELS.get(model_name, None)


def get_available_models() -> dict:
    """Retourne les modèles disponibles par catégorie."""
    return MODEL_CATEGORIES.copy()


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
    'input_chunk_length': 'Nombre de pas de temps en entrée (historique utilisé pour la prédiction).',
    'output_chunk_length': 'Nombre de pas de temps à prédire. Définit l\'horizon de prévision.',
    'batch_size': 'Nombre d\'échantillons traités simultanément.',
    'n_epochs': 'Nombre de passes complètes sur l\'ensemble d\'entraînement.',
    'learning_rate': 'Taux d\'apprentissage de l\'optimiseur. Typiquement 0.001-0.01.',

    # Transformer / Attention-based
    'hidden_size': 'Taille des représentations internes du modèle.',
    'lstm_layers': 'Nombre de couches LSTM empilées.',
    'num_attention_heads': 'Nombre de têtes d\'attention parallèles.',
    'hidden_continuous_size': 'Taille des embeddings pour variables continues dans TFT.',
    'd_model': 'Dimension du modèle Transformer.',
    'nhead': 'Nombre de têtes d\'attention dans le Transformer.',
    'num_encoder_layers': 'Nombre de couches dans l\'encodeur.',
    'num_decoder_layers': 'Nombre de couches dans le décodeur.',
    'dim_feedforward': 'Dimension des couches feed-forward.',

    # RNN-based
    'hidden_dim': 'Dimension de l\'état caché du RNN/LSTM/GRU.',
    'n_rnn_layers': 'Nombre de couches RNN empilées.',
    'model': 'Type de cellule RNN (LSTM ou GRU).',

    # N-BEATS / N-HiTS
    'num_stacks': 'Nombre de stacks (échelles temporelles différentes).',
    'num_blocks': 'Nombre de blocs par stack.',
    'num_layers': 'Nombre de couches fully-connected par bloc.',
    'layer_widths': 'Largeur des couches fully-connected.',

    # TCN
    'num_filters': 'Nombre de filtres convolutionnels.',
    'kernel_size': 'Taille du noyau de convolution.',
    'dilation_base': 'Base pour la dilatation exponentielle.',

    # TiDE
    'decoder_output_dim': 'Dimension de sortie du décodeur TiDE.',
    'temporal_width_past': 'Largeur temporelle pour l\'encodage du passé.',
    'temporal_width_future': 'Largeur temporelle pour l\'encodage du futur.',

    # Linear models
    'shared_weights': 'Partager les poids entre séries (True) ou non (False).',
    'const_init': 'Initialiser avec constantes (True) ou aléatoire (False).',

    # Regularization
    'dropout': 'Taux de dropout pour régularisation (0-0.5).',
}


def get_hyperparam_description(param_name: str) -> str:
    """Retourne la description d'un hyperparamètre."""
    return HYPERPARAM_DESCRIPTIONS.get(param_name, "")
