"""
Script Python de vérification avant les tests.

Usage:
    python scripts/verify_before_test.py
"""

import sys
import inspect
from pathlib import Path
import ast
import tempfile
import json
import os

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Couleurs pour l'affichage (compatibles Windows)
try:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
except:
    GREEN = RED = YELLOW = RESET = ''

def print_ok(message):
    print(f"{GREEN}[OK] {message}{RESET}")

def print_error(message):
    print(f"{RED}[ERROR] {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}[WARNING] {message}{RESET}")

def check_imports():
    """Vérifie que tous les imports fonctionnent."""
    print("\n=== 1. Vérification des Imports ===")
    
    try:
        from core.callbacks import MetricsFileCallback, create_training_callbacks
        print_ok("core.callbacks")
    except Exception as e:
        print_error(f"core.callbacks: {e}")
        return False
    
    try:
        from dashboard.utils.training_monitor import TrainingMonitor
        print_ok("dashboard.utils.training_monitor")
    except Exception as e:
        print_error(f"dashboard.utils.training_monitor: {e}")
        return False
    
    try:
        from core.training import run_training_pipeline
        print_ok("core.training")
    except Exception as e:
        print_error(f"core.training: {e}")
        return False
    
    try:
        from dashboard.utils.training import run_training_pipeline
        print_ok("dashboard.utils.training")
    except Exception as e:
        print_error(f"dashboard.utils.training: {e}")
        return False
    
    return True

def check_syntax():
    """Vérifie la syntaxe des fichiers Python."""
    print("\n=== 2. Vérification de la Syntaxe ===")
    
    files = [
        "core/callbacks.py",
        "dashboard/utils/training_monitor.py",
        "core/training.py",
        "dashboard/utils/training.py",
        "core/model_config.py"
    ]
    
    all_ok = True
    for file in files:
        if not Path(file).exists():
            print_warning(f"{file} n'existe pas")
            continue
        
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                ast.parse(f.read())
            print_ok(f"{file}")
        except SyntaxError as e:
            print_error(f"{file}: {e}")
            all_ok = False
        except UnicodeDecodeError as e:
            print_warning(f"{file}: Problème d'encodage (non-bloquant): {e}")
            # Essayer avec latin-1
            try:
                with open(file, 'r', encoding='latin-1') as f:
                    ast.parse(f.read())
                print_ok(f"{file} (latin-1)")
            except:
                print_error(f"{file}: Impossible de parser")
                all_ok = False
    
    return all_ok

def check_dependencies():
    """Vérifie qu'il n'y a pas de dépendances Streamlit dans core/."""
    print("\n=== 3. Vérification des Dépendances ===")
    
    core_dir = Path("core")
    if not core_dir.exists():
        print_warning("Répertoire core/ n'existe pas")
        return True
    
    violations = []
    for py_file in core_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'import streamlit' in content or 'from streamlit' in content:
                    violations.append(str(py_file))
        except Exception:
            pass
    
    if violations:
        print_error(f"Streamlit trouvé dans core/: {violations}")
        return False
    else:
        print_ok("Pas de Streamlit dans core/")
        return True

def check_signatures():
    """Vérifie que les signatures sont cohérentes."""
    print("\n=== 4. Vérification des Signatures ===")
    
    try:
        from core.training import run_training_pipeline
        from dashboard.utils.training import run_training_pipeline as dashboard_run_training_pipeline
        
        core_sig = inspect.signature(run_training_pipeline)
        dashboard_sig = inspect.signature(dashboard_run_training_pipeline)
        
        required_params = ['metrics_file', 'n_epochs']
        all_ok = True
        
        for param in required_params:
            if param not in core_sig.parameters:
                print_error(f"Paramètre {param} manquant dans core/training.py")
                all_ok = False
            elif param not in dashboard_sig.parameters:
                print_error(f"Paramètre {param} manquant dans dashboard/utils/training.py")
                all_ok = False
        
        if all_ok:
            print_ok("Signatures cohérentes")
        
        return all_ok
    except Exception as e:
        print_error(f"Erreur lors de la vérification des signatures: {e}")
        return False

def check_callbacks():
    """Vérifie que les callbacks sont correctement implémentés."""
    print("\n=== 5. Vérification des Callbacks ===")
    
    try:
        from core.callbacks import MetricsFileCallback, create_training_callbacks
        from pytorch_lightning.callbacks import Callback
        
        # Vérifier l'héritage
        if not issubclass(MetricsFileCallback, Callback):
            print_error("MetricsFileCallback n'hérite pas de Callback")
            return False
        print_ok("Héritage correct")
        
        # Vérifier les méthodes requises
        required_methods = ['on_train_start', 'on_train_epoch_end', 'on_train_end']
        for method in required_methods:
            if not hasattr(MetricsFileCallback, method):
                print_error(f"Méthode {method} manquante")
                return False
        print_ok("Méthodes requises présentes")
        
        # Vérifier create_training_callbacks
        metrics_file = Path(tempfile.gettempdir()) / "test_metrics.json"
        callbacks = create_training_callbacks(
            metrics_file=metrics_file,
            total_epochs=10,
            early_stopping_patience=5
        )
        
        if not isinstance(callbacks, list):
            print_error("create_training_callbacks ne retourne pas une liste")
            return False
        
        if len(callbacks) != 2:
            print_warning(f"Devrait avoir 2 callbacks, a {len(callbacks)}")
        
        print_ok("Callbacks créés correctement")
        return True
    except Exception as e:
        print_error(f"Erreur lors de la vérification des callbacks: {e}")
        return False

def check_documentation():
    """Vérifie que la documentation existe."""
    print("\n=== 6. Vérification de la Documentation ===")
    
    docs = [
        "ARCHITECTURE.md",
        "PLAN_MIGRATION.md",
        "PLAN_VERIFICATION.md"
    ]
    
    all_ok = True
    for doc in docs:
        if Path(doc).exists():
            print_ok(f"{doc}")
        else:
            print_error(f"{doc} manquant")
            all_ok = False
    
    return all_ok

def check_training_monitor():
    """Vérifie que TrainingMonitor fonctionne correctement."""
    print("\n=== 7. Vérification de TrainingMonitor ===")
    
    try:
        from dashboard.utils.training_monitor import TrainingMonitor
        from pathlib import Path
        
        # Test avec fichier inexistant
        monitor = TrainingMonitor(Path("/tmp/nonexistent_test.json"))
        metrics = monitor.read_metrics()
        if metrics is not None:
            print_warning("Devrait retourner None pour fichier inexistant")
        else:
            print_ok("Gestion fichier inexistant")
        
        # Test avec fichier JSON valide
        test_file = Path(tempfile.gettempdir()) / "test_metrics.json"
        test_data = {
            "status": "training",
            "current_epoch": 1,
            "total_epochs": 10,
            "train_losses": [0.5],
            "val_losses": [0.6],
            "epochs": [1]
        }
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        monitor = TrainingMonitor(test_file)
        metrics = monitor.read_metrics()
        if metrics and metrics.get('status') == 'training':
            print_ok("Lecture métriques OK")
        else:
            print_error("Lecture métriques échouée")
            return False
        
        # Nettoyer
        test_file.unlink()
        
        return True
    except Exception as e:
        print_error(f"Erreur lors de la vérification de TrainingMonitor: {e}")
        return False

def main():
    """Fonction principale."""
    print("=" * 50)
    print("Vérification Avant Tests")
    print("=" * 50)
    
    checks = [
        ("Imports", check_imports),
        ("Syntaxe", check_syntax),
        ("Dépendances", check_dependencies),
        ("Signatures", check_signatures),
        ("Callbacks", check_callbacks),
        ("Documentation", check_documentation),
        ("TrainingMonitor", check_training_monitor),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Erreur dans {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Résumé")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        if result:
            print_ok(f"{name}")
        else:
            print_error(f"{name}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print_ok("Toutes les vérifications sont OK !")
        print("\nVous pouvez maintenant procéder aux tests fonctionnels.")
        return 0
    else:
        print_error("Certaines vérifications ont échoué.")
        print("\nVeuillez corriger les erreurs avant de procéder aux tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
