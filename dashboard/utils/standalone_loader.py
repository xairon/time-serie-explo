#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script autonome pour charger les modèles Darts.

Ce script est conçu pour être exécuté de manière complètement indépendante,
sans aucune interférence de Streamlit.

Usage:
    python standalone_loader.py <model_path> <model_type> <output_path>
"""

import sys
import os
import pickle
import json
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Usage: standalone_loader.py <model_path> <model_type> <output_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].upper()
    output_path = sys.argv[3]

    try:
        # Désactiver les warnings
        import warnings
        warnings.filterwarnings("ignore")

        # Import des modèles Darts
        from darts.models import (
            TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
            RNNModel, BlockRNNModel, TCNModel, TiDEModel,
            DLinearModel, NLinearModel
        )

        # Mapping des modèles
        model_classes = {
            'TFT': TFTModel,
            'NBEATS': NBEATSModel,
            'NHITS': NHiTSModel,
            'TRANSFORMER': TransformerModel,
            'LSTM': RNNModel,
            'GRU': RNNModel,
            'BLOCKRNN': BlockRNNModel,
            'TCN': TCNModel,
            'TIDE': TiDEModel,
            'DLINEAR': DLinearModel,
            'NLINEAR': NLinearModel,
        }

        model_class = model_classes.get(model_type)
        if not model_class:
            print(json.dumps({"error": f"Unknown model type: {model_type}"}))
            sys.exit(1)

        # Charger le modèle
        model = model_class.load(model_path)

        # Sauvegarder dans le fichier de sortie
        with open(output_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(json.dumps({"success": True}))

    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()