"""Configuration of available models for forecasting.

IMPORTANT: Only Deep Learning models are supported as they share
the same interface (input_chunk_length, output_chunk_length).
"""

# =============================================================================
# DEEP LEARNING MODELS (Only supported models)
# =============================================================================
DL_MODELS = {
    # Transformer-based (defaults tuned for better out-of-box performance)
    'TFT': {
        'name': 'Temporal Fusion Transformer',
        'class': 'TFTModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Transformer with attention, interpretable, excellent for complex time series',
        'hyperparams': {
            'hidden_size': (16, 256, 128),
            'lstm_layers': (1, 4, 2),
            'num_attention_heads': (1, 8, 4),
            'dropout': (0.0, 0.3, 0.1),
            'hidden_continuous_size': (8, 64, 32)
        }
    },
    'Transformer': {
        'name': 'Transformer',
        'class': 'TransformerModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Classic Transformer architecture for time series',
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
        'description': 'Automatic trend/seasonality decomposition, highly performant',
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
        'description': 'N-BEATS evolution, faster and more efficient',
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
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Long Short-Term Memory, classic baseline for time series',
        'hyperparams': {
            'hidden_dim': (32, 256, 128),
            'n_rnn_layers': (1, 4, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
    'GRU': {
        'name': 'GRU',
        'class': 'RNNModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Gated Recurrent Unit, faster than LSTM',
        'hyperparams': {
            'hidden_dim': (32, 256, 128),
            'n_rnn_layers': (1, 4, 2),
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
        'description': 'Optimized RNN for multi-horizon predictions',
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
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Causal convolutions, fast and performant',
        'hyperparams': {
            'num_filters': (16, 128, 64),
            'kernel_size': (2, 7, 5),
            'num_layers': (2, 6, 4),
            'dilation_base': (2, 4, 2),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
    'TiDE': {
        'name': 'TiDE',
        'class': 'TiDEModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Time-series Dense Encoder, efficient MLP architecture',
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
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Decomposition + Linear, very simple but effective',
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
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'Normalization + Linear, baseline for long-term forecasting',
        'hyperparams': {
            'shared_weights': [True, False],
            'const_init': [True, False]
        }
    },
    
    # MLP-Mixer based (State-of-the-art 2023; defaults tuned for better OOB performance)
    'TSMixer': {
        'name': 'TSMixer',
        'class': 'TSMixerModel',
        'supports_covariates': True,
        'supports_past_covariates': True,
        'supports_future_covariates': False,  # Disabled to avoid prediction bias
        'multivariate': True,
        'description': 'MLP-Mixer for time series, State-of-the-art 2023',
        'hyperparams': {
            'hidden_size': (32, 256, 128),
            'ff_size': (64, 512, 256),
            'num_blocks': (1, 4, 3),
            'dropout': (0.0, 0.3, 0.1)
        }
    },
}

# =============================================================================
# GLOBAL BASELINES (Simple models but compatible with interface)
# =============================================================================
BASELINE_MODELS = {
    'GlobalNaiveAggregate': {
        'name': 'Global Naive Aggregate',
        'class': 'GlobalNaiveAggregate',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Mean/Median of past values. Fast baseline.',
        'hyperparams': {}
    },
    'GlobalNaiveDrift': {
        'name': 'Global Naive Drift',
        'class': 'GlobalNaiveDrift',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Linear extrapolation of trend. Baseline.',
        'hyperparams': {}
    },
    'GlobalNaiveSeasonal': {
        'name': 'Global Naive Seasonal',
        'class': 'GlobalNaiveSeasonal',
        'supports_covariates': False,
        'supports_past_covariates': False,
        'supports_future_covariates': False,
        'multivariate': True,
        'description': 'Repeats past seasonality. Baseline.',
        'hyperparams': {}
    },
}

# =============================================================================
# FINAL AGGREGATION
# =============================================================================
ALL_MODELS = {**DL_MODELS, **BASELINE_MODELS}

# Categories for UI
MODEL_CATEGORIES = {
    'Deep Learning': list(DL_MODELS.keys()),
    'Baselines': list(BASELINE_MODELS.keys()),
}

# Recommended models to start
RECOMMENDED_MODELS = ['TFT', 'NBEATS', 'NHiTS', 'TSMixer', 'LSTM']


def get_model_info(model_name):
    """Returns information about a model."""
    return ALL_MODELS.get(model_name, None)


def get_available_models() -> dict:
    """Returns available models by category."""
    return MODEL_CATEGORIES.copy()


def get_models_by_capability(capability):
    """Filter models by capability (covariates, multivariate, etc.)."""
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
    """Returns the hyperparameter search space."""
    model_info = get_model_info(model_name)
    if model_info:
        return model_info['hyperparams']
    return {}


# =============================================================================
# HYPERPARAMETER DESCRIPTIONS
# =============================================================================
HYPERPARAM_DESCRIPTIONS = {
    # Common hyperparameters
    'input_chunk_length': 'Number of input time steps (history used for prediction).',
    'output_chunk_length': 'Number of time steps to predict. Defines forecast horizon.',
    'batch_size': 'Number of samples processed simultaneously.',
    'n_epochs': 'Number of complete passes through the training set.',
    'learning_rate': 'Optimizer learning rate. Typically 0.001-0.01.',

    # Transformer / Attention-based
    'hidden_size': 'Size of internal model representations.',
    'lstm_layers': 'Number of stacked LSTM layers.',
    'num_attention_heads': 'Number of parallel attention heads.',
    'hidden_continuous_size': 'Embedding size for continuous variables in TFT.',
    'd_model': 'Transformer model dimension.',
    'nhead': 'Number of attention heads in Transformer.',
    'num_encoder_layers': 'Number of layers in encoder.',
    'num_decoder_layers': 'Number of layers in decoder.',
    'dim_feedforward': 'Dimension of feed-forward layers.',

    # RNN-based
    'hidden_dim': 'Hidden state dimension for RNN/LSTM/GRU.',
    'n_rnn_layers': 'Number of stacked RNN layers.',
    'model': 'RNN cell type (LSTM or GRU).',

    # N-BEATS / N-HiTS
    'num_stacks': 'Number of stacks (different time scales).',
    'num_blocks': 'Number of blocks per stack.',
    'num_layers': 'Number of fully-connected layers per block.',
    'layer_widths': 'Width of fully-connected layers.',

    # TCN
    'num_filters': 'Number of convolutional filters.',
    'kernel_size': 'Size of convolution kernel.',
    'dilation_base': 'Base for exponential dilation.',

    # TiDE
    'decoder_output_dim': 'Output dimension of TiDE decoder.',
    'temporal_width_past': 'Temporal width for encoding the past.',
    'temporal_width_future': 'Temporal width for encoding the future.',

    # Linear models
    'shared_weights': 'Share weights between series (True) or not (False).',
    'const_init': 'Initialize with constants (True) or random (False).',

    # Regularization
    'dropout': 'Dropout rate for regularization (0-0.5).',
}


def get_hyperparam_description(param_name: str) -> str:
    """Returns the description of a hyperparameter."""
    return HYPERPARAM_DESCRIPTIONS.get(param_name, "")
