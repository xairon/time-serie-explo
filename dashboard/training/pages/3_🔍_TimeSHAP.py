"""Page d'explainability avec TimeSHAP pour interpréter les prédictions."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.model_registry import get_registry
from dashboard.utils.data_loader import load_station_data
from dashboard.utils.preprocessing import prepare_dataframe_for_darts

st.set_page_config(page_title="TimeSHAP Explainability", page_icon="🔍", layout="wide")

st.title("🔍 TimeSHAP - Explainability des Prédictions")
st.markdown("""
Analysez **pourquoi** votre modèle fait certaines prédictions.

TimeSHAP (Time Series SHAP) permet de comprendre l'importance de chaque pas de temps passé
dans la prédiction, révélant quelles périodes historiques influencent le plus les prévisions.
""")
st.markdown("---")

# Vérifier si timeshap est installé
try:
    import shap
    import timeshap
    TIMESHAP_AVAILABLE = True
except ImportError:
    TIMESHAP_AVAILABLE = False

if not TIMESHAP_AVAILABLE:
    st.error("""
    ❌ **TimeSHAP n'est pas installé**

    Pour utiliser cette fonctionnalité, installez les dépendances nécessaires:

    ```bash
    pip install shap timeshap
    ```

    TimeSHAP permet de calculer les valeurs SHAP pour les modèles de séries temporelles,
    révélant l'importance de chaque pas de temps dans les prédictions.
    """)

    st.markdown("---")
    st.markdown("### 📚 En savoir plus sur TimeSHAP")

    st.markdown("""
    **TimeSHAP** est une extension de SHAP (SHapley Additive exPlanations) pour les séries temporelles.

    **Avantages:**
    - 🎯 Identifie les pas de temps importants pour chaque prédiction
    - 📊 Visualise l'importance temporelle
    - 🔍 Détecte les patterns et dépendances temporelles
    - 💡 Aide à comprendre et déboguer les modèles

    **Cas d'usage:**
    - Comprendre pourquoi le modèle prédit une hausse/baisse
    - Identifier les événements passés influents
    - Valider que le modèle utilise les bonnes informations
    - Détecter les biais temporels

    **Documentation:**
    - [SHAP Documentation](https://shap.readthedocs.io/)
    - [TimeSHAP Paper](https://arxiv.org/abs/2012.15539)
    - [GitHub timeshap](https://github.com/feedzai/timeshap)
    """)

    st.stop()

# Si TimeSHAP est disponible, continuer
registry = get_registry()
models = registry.scan_models()

if not models:
    st.warning("⚠️ Aucun modèle entraîné trouvé. Entraînez d'abord un modèle dans **Train Models**.")
    st.stop()

# Grouper par station
models_by_station = {}
for model in models:
    if model.station not in models_by_station:
        models_by_station[model.station] = []
    models_by_station[model.station].append(model)

st.success(f"✅ {len(models)} modèle(s) disponible(s)")

# Section de sélection
st.subheader("1️⃣ Sélection du modèle")

col1, col2 = st.columns(2)

with col1:
    available_stations = sorted(models_by_station.keys())
    selected_station = st.selectbox(
        "Station",
        options=available_stations,
        help="Station à analyser"
    )

with col2:
    station_models = models_by_station[selected_station]

    model_labels = [f"{m.model_type} - {m.model_name}" for m in station_models]

    selected_model_idx = st.selectbox(
        "Modèle",
        options=range(len(station_models)),
        format_func=lambda i: model_labels[i],
        help="Modèle à expliquer"
    )

selected_model = station_models[selected_model_idx]

st.markdown("---")

st.subheader("2️⃣ Configuration de l'analyse")

col1, col2 = st.columns(2)

with col1:
    n_samples = st.slider(
        "Nombre d'échantillons à analyser",
        min_value=1,
        max_value=20,
        value=5,
        help="Plus d'échantillons = analyse plus précise mais plus lente"
    )

with col2:
    background_size = st.slider(
        "Taille du background dataset",
        min_value=10,
        max_value=100,
        value=50,
        help="Dataset de référence pour le calcul SHAP. Plus grand = plus précis mais plus lent"
    )

st.markdown("---")

# Note importante
st.info("""
⚠️ **Note importante:**

L'implémentation complète de TimeSHAP nécessite:
1. Un wrapper autour du modèle Darts pour l'interface SHAP
2. Une fonction de prédiction compatible avec le calcul des valeurs SHAP
3. La gestion des séquences temporelles dans le format attendu par TimeSHAP

Cette page fournit le cadre et l'interface. L'intégration complète nécessite:
- Adaptation du modèle au format SHAP explainer
- Implémentation du calcul des valeurs SHAP temporelles
- Génération des visualisations interactives
""")

if st.button("🔍 Analyser avec TimeSHAP", type="primary"):
    st.warning("""
    🚧 **Fonctionnalité en développement**

    L'intégration complète de TimeSHAP avec les modèles Darts est en cours.

    **Prochaines étapes:**
    1. Créer un wrapper PyTorch pour exposer les modèles Darts à SHAP
    2. Implémenter le calcul des valeurs SHAP pour séquences temporelles
    3. Générer les visualisations d'importance temporelle
    4. Ajouter l'analyse d'événements spécifiques

    **En attendant, vous pouvez:**
    - Utiliser les graphiques d'erreurs dans la page Forecasting
    - Analyser les prédictions vs réalité pour identifier les patterns
    - Examiner les hyperparamètres du modèle pour comprendre son comportement
    """)

# Section éducative: Comment TimeSHAP fonctionne
st.markdown("---")
st.subheader("📚 Comment fonctionne TimeSHAP?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Principe de base:**

    TimeSHAP calcule la contribution de chaque pas de temps passé
    à la prédiction en utilisant les valeurs de Shapley issues de la
    théorie des jeux coopératifs.

    **Étapes:**
    1. Sélection d'une prédiction à expliquer
    2. Création d'un background dataset
    3. Calcul des contributions marginales
    4. Agrégation en valeurs SHAP
    5. Visualisation de l'importance temporelle
    """)

with col2:
    st.markdown("""
    **Interprétation:**

    - **Valeur SHAP positive** → Ce pas de temps pousse la prédiction vers le haut
    - **Valeur SHAP négative** → Ce pas de temps pousse la prédiction vers le bas
    - **Magnitude** → Importance de la contribution

    **Applications:**
    - Identifier les événements déclencheurs
    - Comprendre les dépendances temporelles
    - Valider la logique du modèle
    - Détecter les anomalies d'apprentissage
    """)

# Exemple de visualisation (mockup)
st.markdown("---")
st.subheader("📊 Exemple de visualisation TimeSHAP")

# Créer des données exemple pour montrer à quoi ça ressemblerait
st.markdown("*Illustration conceptuelle - données fictives*")

# Générer des valeurs SHAP fictives
n_timesteps = 30
dates = pd.date_range(end=pd.Timestamp.now(), periods=n_timesteps, freq='D')
shap_values = np.random.randn(n_timesteps) * 0.5

# Graphique d'importance temporelle
fig = go.Figure()

# Barres colorées selon le signe
colors = ['red' if v < 0 else 'green' for v in shap_values]

fig.add_trace(go.Bar(
    x=dates,
    y=shap_values,
    marker_color=colors,
    name='Contribution SHAP',
    hovertemplate='<b>Date:</b> %{x}<br><b>Contribution:</b> %{y:.3f}<extra></extra>'
))

fig.update_layout(
    title="Importance temporelle - Contribution de chaque jour à la prédiction",
    xaxis_title="Date",
    yaxis_title="Valeur SHAP (contribution)",
    hovermode='x',
    height=400
)

fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
Dans cet exemple fictif:
- Les **barres vertes** (valeurs positives) indiquent les jours qui poussent la prédiction vers le haut
- Les **barres rouges** (valeurs négatives) indiquent les jours qui poussent la prédiction vers le bas
- La **hauteur** des barres indique l'importance de chaque jour

Avec TimeSHAP réel, vous verriez l'importance exacte de chaque pas de temps dans vos prédictions!
""")

# Waterfall plot exemple
st.markdown("---")
st.markdown("### Waterfall Plot - Décomposition de la prédiction")

# Créer un waterfall plot fictif
categories = ['Base', 'J-30 à J-21', 'J-20 à J-11', 'J-10 à J-1', 'Prédiction finale']
values = [2.5, 0.3, -0.2, 0.8, 3.4]

fig_waterfall = go.Figure(go.Waterfall(
    name = "Contribution",
    orientation = "v",
    measure = ["absolute", "relative", "relative", "relative", "total"],
    x = categories,
    textposition = "outside",
    y = values,
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
))

fig_waterfall.update_layout(
    title = "Décomposition de la prédiction (exemple fictif)",
    height=400
)

st.plotly_chart(fig_waterfall, use_container_width=True)

st.markdown("""
Le waterfall plot montre comment la prédiction est construite étape par étape:
1. On part d'une **valeur de base** (prédiction moyenne)
2. Chaque **groupe de pas de temps** ajoute/retire de la valeur
3. On arrive à la **prédiction finale**

Cela permet de comprendre quelles périodes ont eu le plus d'impact!
""")

# Footer avec ressources
st.markdown("---")
st.markdown("### 🔗 Ressources supplémentaires")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Documentation SHAP**
    - [SHAP Official Docs](https://shap.readthedocs.io/)
    - [SHAP Python API](https://shap.readthedocs.io/en/latest/api.html)
    - [Tutorials](https://shap.readthedocs.io/en/latest/example_notebooks.html)
    """)

with col2:
    st.markdown("""
    **TimeSHAP Resources**
    - [TimeSHAP Paper](https://arxiv.org/abs/2012.15539)
    - [GitHub Repository](https://github.com/feedzai/timeshap)
    - [Blog Post](https://blog.feedzai.com/time-series-explainability/)
    """)

with col3:
    st.markdown("""
    **Interpretability**
    - [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
    - [Distill.pub Articles](https://distill.pub/)
    - [Google's PAIR](https://pair.withgoogle.com/)
    """)
