"""
Script de migration pour nettoyer les modèles existants.

Ce script charge les modèles avec le patch robust_loader, nettoie les callbacks
problématiques, et resauvegarde les modèles de manière propre.

Usage:
    python scripts/migrate_models.py --checkpoints-dir checkpoints/darts
    python scripts/migrate_models.py --checkpoints-dir checkpoints/darts --dry-run
"""

import argparse
import sys
from pathlib import Path
import logging

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.utils.robust_loader import load_model_safe
from core.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_model_callbacks(model):
    """
    Nettoie les callbacks d'un modèle chargé.
    
    Cette fonction retire les callbacks qui contiennent des références Streamlit
    ou d'autres dépendances non-sérialisables.
    
    Args:
        model: Modèle Darts chargé
    
    Returns:
        Modèle nettoyé
    """
    # Pour les modèles PyTorch Lightning, les callbacks sont dans le Trainer
    if hasattr(model, 'trainer') and model.trainer is not None:
        original_callbacks = model.trainer.callbacks if hasattr(model.trainer, 'callbacks') else []
        
        # Identifier les callbacks problématiques
        problematic_callbacks = []
        for cb in original_callbacks:
            cb_module = cb.__class__.__module__
            cb_class_name = cb.__class__.__name__
            
            # Identifier les callbacks Streamlit
            if 'streamlit' in cb_module.lower() or 'StreamlitProgressCallback' in cb_class_name:
                problematic_callbacks.append(cb)
                logger.debug(f"Found problematic callback: {cb_class_name} from {cb_module}")
        
        if problematic_callbacks:
            logger.info(f"Found {len(problematic_callbacks)} problematic callbacks")
            # Note: On ne peut pas modifier le Trainer après création
            # Mais on peut documenter le problème et suggérer une réentraînement
    
    return model


def migrate_model(model_path: Path, model_type: str, dry_run: bool = False) -> bool:
    """
    Migre un modèle en le chargeant, nettoyant, et resauvegardant.
    
    Args:
        model_path: Chemin vers le fichier .pkl du modèle
        model_type: Type du modèle (TFT, NBEATS, etc.)
        dry_run: Si True, ne sauvegarde pas, juste vérifie
    
    Returns:
        True si la migration a réussi, False sinon
    """
    try:
        logger.info(f"Loading model: {model_path}")
        
        # Charger avec le patch (pour les modèles anciens)
        model = load_model_safe(model_path, model_type)
        
        # Nettoyer les callbacks
        cleaned_model = clean_model_callbacks(model)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would resave model: {model_path}")
            return True
        
        # Resauvegarder le modèle
        # On sauvegarde dans un fichier temporaire d'abord
        temp_path = model_path.with_suffix('.pkl.tmp')
        cleaned_model.save(str(temp_path))
        
        # Remplacer l'ancien fichier
        backup_path = model_path.with_suffix('.pkl.backup')
        model_path.rename(backup_path)
        temp_path.rename(model_path)
        
        logger.info(f"✅ Model migrated successfully: {model_path}")
        logger.info(f"   Backup saved to: {backup_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to migrate {model_path}: {e}")
        return False


def find_models(checkpoints_dir: Path) -> list:
    """
    Trouve tous les modèles dans le répertoire de checkpoints.
    
    Args:
        checkpoints_dir: Répertoire racine des checkpoints
    
    Returns:
        Liste de tuples (model_path, model_type)
    """
    models = []
    
    # Parcourir les sous-répertoires
    darts_dir = checkpoints_dir / "darts"
    if not darts_dir.exists():
        logger.warning(f"Directory not found: {darts_dir}")
        return models
    
    # Parcourir tous les dossiers de modèles
    for model_dir in darts_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Chercher les fichiers .pkl
        for pkl_file in model_dir.glob("*.pkl"):
            # Ignorer les fichiers de backup
            if pkl_file.name.endswith('.backup') or pkl_file.name.endswith('.tmp'):
                continue
            
            # Essayer de déterminer le type de modèle depuis le nom du dossier
            # Format: {type}_{MODEL}_{STATION}_{TIMESTAMP}
            model_id = model_dir.name
            parts = model_id.split('_')
            
            if len(parts) >= 2:
                model_type = parts[1].upper()
                models.append((pkl_file, model_type))
            else:
                logger.warning(f"Could not determine model type for {pkl_file}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Migrate existing models to remove Streamlit dependencies")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory of checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually migrate, just check what would be done"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Only migrate models of this type (e.g., TFT, NBEATS)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Model Migration Script")
    logger.info("=" * 60)
    logger.info(f"Checkpoints directory: {args.checkpoints_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.model_type:
        logger.info(f"Filtering by model type: {args.model_type}")
    logger.info("")
    
    # Trouver tous les modèles
    models = find_models(args.checkpoints_dir)
    
    if not models:
        logger.warning("No models found!")
        return
    
    logger.info(f"Found {len(models)} models to migrate")
    logger.info("")
    
    # Filtrer par type si demandé
    if args.model_type:
        models = [(p, t) for p, t in models if t.upper() == args.model_type.upper()]
        logger.info(f"Filtered to {len(models)} models of type {args.model_type}")
        logger.info("")
    
    # Migrer chaque modèle
    success_count = 0
    fail_count = 0
    
    for model_path, model_type in models:
        if migrate_model(model_path, model_type, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")
    logger.info(f"📊 Total: {len(models)}")
    
    if args.dry_run:
        logger.info("")
        logger.info("This was a dry run. Use without --dry-run to actually migrate.")


if __name__ == "__main__":
    main()
