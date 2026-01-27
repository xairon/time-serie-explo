#!/bin/bash
# Script de vérification rapide avant les tests

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "Vérification Avant Tests"
echo "=========================================="
echo ""

echo "=== 1. Vérification des Imports ==="
python -c "from core.callbacks import MetricsFileCallback, create_training_callbacks; print('✅ core.callbacks OK')" || { echo "❌ Erreur core.callbacks"; exit 1; }
python -c "from dashboard.utils.training_monitor import TrainingMonitor; print('✅ training_monitor OK')" || { echo "❌ Erreur training_monitor"; exit 1; }
python -c "from core.training import run_training_pipeline; print('✅ core.training OK')" || { echo "❌ Erreur core.training"; exit 1; }
python -c "from dashboard.utils.training import run_training_pipeline; print('✅ dashboard.training OK')" || { echo "❌ Erreur dashboard.training"; exit 1; }
echo ""

echo "=== 2. Vérification de la Syntaxe ==="
python -m py_compile core/callbacks.py || { echo "❌ Erreur syntaxe core/callbacks.py"; exit 1; }
python -m py_compile dashboard/utils/training_monitor.py || { echo "❌ Erreur syntaxe training_monitor.py"; exit 1; }
python -m py_compile core/training.py || { echo "❌ Erreur syntaxe core/training.py"; exit 1; }
python -m py_compile dashboard/utils/training.py || { echo "❌ Erreur syntaxe dashboard/utils/training.py"; exit 1; }
python -m py_compile core/model_config.py || { echo "❌ Erreur syntaxe core/model_config.py"; exit 1; }
echo "✅ Syntaxe OK"
echo ""

echo "=== 3. Vérification des Dépendances ==="
if grep -r "import streamlit\|from streamlit" core/ 2>/dev/null; then
    echo "❌ Streamlit trouvé dans core/ - PROBLÈME!"
    exit 1
else
    echo "✅ Pas de Streamlit dans core/"
fi
echo ""

echo "=== 4. Vérification de la Documentation ==="
if [ ! -f ARCHITECTURE.md ]; then
    echo "❌ ARCHITECTURE.md manquant"
    exit 1
fi
if [ ! -f PLAN_MIGRATION.md ]; then
    echo "❌ PLAN_MIGRATION.md manquant"
    exit 1
fi
if [ ! -f PLAN_VERIFICATION.md ]; then
    echo "❌ PLAN_VERIFICATION.md manquant"
    exit 1
fi
echo "✅ Documentation complète"
echo ""

echo "=== 5. Vérification des Signatures ==="
python << 'EOF'
import inspect
from core.training import run_training_pipeline
from dashboard.utils.training import run_training_pipeline as dashboard_run_training_pipeline

core_sig = inspect.signature(run_training_pipeline)
dashboard_sig = inspect.signature(dashboard_run_training_pipeline)

required_params = ['metrics_file', 'n_epochs']
for param in required_params:
    if param not in core_sig.parameters:
        print(f"❌ Paramètre {param} manquant dans core/training.py")
        exit(1)
    if param not in dashboard_sig.parameters:
        print(f"❌ Paramètre {param} manquant dans dashboard/utils/training.py")
        exit(1)

print("✅ Signatures cohérentes")
EOF
echo ""

echo "=== 6. Vérification des Callbacks ==="
python << 'EOF'
from core.callbacks import MetricsFileCallback, create_training_callbacks
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import tempfile

# Vérifier l'héritage
if not issubclass(MetricsFileCallback, Callback):
    print("❌ MetricsFileCallback n'hérite pas de Callback")
    exit(1)

# Vérifier les méthodes requises
required_methods = ['on_train_start', 'on_train_epoch_end', 'on_train_end']
for method in required_methods:
    if not hasattr(MetricsFileCallback, method):
        print(f"❌ Méthode {method} manquante")
        exit(1)

# Vérifier create_training_callbacks
metrics_file = Path(tempfile.gettempdir()) / "test_metrics.json"
callbacks = create_training_callbacks(
    metrics_file=metrics_file,
    total_epochs=10,
    early_stopping_patience=5
)

if not isinstance(callbacks, list):
    print("❌ create_training_callbacks ne retourne pas une liste")
    exit(1)

if len(callbacks) != 2:
    print(f"❌ Devrait avoir 2 callbacks, a {len(callbacks)}")
    exit(1)

print("✅ Callbacks OK")
EOF
echo ""

echo "=========================================="
echo "✅ Toutes les vérifications sont OK !"
echo "=========================================="
echo ""
echo "Vous pouvez maintenant procéder aux tests fonctionnels."
echo ""
