"""
Script de test pour vérifier le chargement des modèles.
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from dashboard.utils.model_config import load_model_with_config
from dashboard.config import CHECKPOINTS_DIR

def test_model_loading():
    """Test le chargement d'un modèle existant."""

    print("Recherche des modèles dans:", CHECKPOINTS_DIR)

    checkpoints_dir = Path(CHECKPOINTS_DIR)
    if not checkpoints_dir.exists():
        print("Le répertoire checkpoints n'existe pas")
        return

    # Lister les modèles disponibles
    models_found = []
    for model_dir in checkpoints_dir.iterdir():
        if model_dir.is_dir():
            config_file = model_dir / "model_config.yaml"
            if config_file.exists():
                models_found.append(model_dir)
                print(f"Modèle trouvé: {model_dir.name}")

    if not models_found:
        print("[WARNING] Aucun modèle trouvé")
        return

    # Tester le chargement du premier modèle
    test_model_dir = models_found[0]
    print(f"\n[LOADING] Test de chargement de: {test_model_dir.name}")

    try:
        # Charger le modèle
        model, config, data_dict = load_model_with_config(test_model_dir)

        print("[OK] Modèle chargé avec succès!")
        print(f"   - Type: {config.model_name}")
        print(f"   - Station: {config.station}")
        print(f"   - Station originale: {config.original_station_id}")
        print(f"   - Target: {config.columns.get('target')}")
        print(f"   - Métriques: MAE={config.metrics.get('MAE', 'N/A')}")

        # Vérifier que le modèle a les bonnes méthodes
        if hasattr(model, 'predict'):
            print("[OK] Le modèle a une méthode predict()")
        else:
            print("[ERROR] Le modèle n'a pas de méthode predict()")

        # Vérifier les données
        print(f"\n[DATA] Données chargées:")
        for key, df in data_dict.items():
            if df is not None:
                print(f"   - {key}: {len(df)} lignes")

        print("\n[SUCCESS] Test réussi!")

    except Exception as e:
        print(f"\n[ERROR] Erreur lors du chargement: {e}")

        # Afficher plus de détails sur l'erreur
        import traceback
        print("\nDétails de l'erreur:")
        traceback.print_exc()

        # Conseils de résolution
        if "__setstate__" in str(e) or "StreamlitAPIException" in str(e):
            print("\n[TIP] Erreur Streamlit détectée!")
            print("   Cette erreur survient quand Streamlit interfère avec le chargement PyTorch.")
            print("   La solution de contournement par subprocess devrait fonctionner.")

        print(f"\n[SOLUTIONS] Solutions possibles:")
        print(f"   1. Vérifier que tous les packages sont installés (darts, torch, etc.)")
        print(f"   2. Supprimer et ré-entraîner le modèle")
        print(f"   3. Utiliser le script en dehors de Streamlit")

if __name__ == "__main__":
    print("=" * 50)
    print("TEST DE CHARGEMENT DE MODÈLE")
    print("=" * 50)
    test_model_loading()