"""Page d'entraînement de modèles avec Optuna."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import (
    STATIONS, MODELS, DEFAULT_HYPERPARAMS, CHECKPOINTS_DIR, METRICS, METRICS_INFO
)
from dashboard.utils.data_loader import prepare_data_for_training
from dashboard.utils.forecasting import (
    run_training_pipeline, update_leaderboard
)
from dashboard.utils.optuna_utils import (
    create_optuna_objective, run_optuna_study, get_best_params, get_best_value,
    plot_optuna_optimization_history, plot_optuna_param_importances, get_trials_dataframe
)
from dashboard.utils.plots import plot_predictions
from dashboard.utils.state import save_context

def get_model_specific_space(model_type):
    """Retourne la description de l'espace de recherche spécifique."""
    if model_type == 'NBEATS':
        return """- num_stacks: [10, 20, 30]
- num_blocks: [1, 2, 3]
- num_layers: [2, 3, 4, 5]
- layer_widths: [128, 256, 512]"""
    elif model_type == 'TFT':
        return """- hidden_size: [32, 64, 128]
- lstm_layers: [1, 2, 3]
- num_attention_heads: [2, 4, 8]
- dropout: [0.0, 0.3]"""
    elif model_type == 'TCN':
        return """- num_filters: [32, 64, 128]
- kernel_size: [3, 5, 7]
- num_layers: [2, 3, 4, 5]
- dilation_base: [2, 3, 4]
- dropout: [0.0, 0.3]"""
    elif model_type == 'LSTM':
        return """- hidden_dim: [64, 128, 256]
- n_rnn_layers: [1, 2, 3]
- dropout: [0.0, 0.3]"""

st.set_page_config(page_title="Train Models", page_icon="🎯", layout="wide")

st.title("🎯 Train Models")
st.markdown("Entraînez des modèles deep learning avec ou sans optimisation Optuna.")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🚀 Quick Train", "🔬 Optuna Optimization"])

# ============================================================================
# TAB 1: QUICK TRAIN
# ============================================================================
with tab1:
    st.subheader("Entraînement Rapide")

    col1, col2 = st.columns(2)

    with col1:
        station = st.selectbox("Station", STATIONS, key='quick_station')
        model_type = st.selectbox("Modèle", MODELS, key='quick_model')

    with col2:
        use_covariates = st.checkbox(
            "Utiliser les covariates",
            value=True,
            help="TFT, TCN et LSTM supportent les covariates. N-BEATS ne les utilise pas.",
            key='quick_cov'
        )
        fill_missing = st.checkbox(
            "Interpoler les données manquantes",
            value=False,
            help="Si coché, remplit les trous par interpolation temporelle. Sinon, supprime les lignes incomplètes.",
            key='quick_fill'
        )
        n_epochs = st.slider("Epochs", 10, 100, 50, key='quick_epochs')

    # Hyperparamètres
    with st.expander("⚙️ Hyperparamètres"):
        input_chunk = st.number_input("Input Chunk", 10, 90, DEFAULT_HYPERPARAMS['input_chunk'])
        output_chunk = st.number_input("Output Chunk", 1, 30, DEFAULT_HYPERPARAMS['output_chunk'])
        batch_size = st.number_input("Batch Size", 8, 128, DEFAULT_HYPERPARAMS['batch_size'])
        learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, DEFAULT_HYPERPARAMS['learning_rate'], format="%.5f")

        # Modèle-spécifique
        if model_type == 'NBEATS':
            num_stacks = st.slider("Num Stacks", 10, 50, 30)
            layer_widths = st.slider("Layer Widths", 128, 512, 256, step=64)
            hyperparams = {
                'input_chunk': input_chunk, 'output_chunk': output_chunk,
                'batch_size': batch_size, 'n_epochs': n_epochs, 'learning_rate': learning_rate,
                'num_stacks': num_stacks, 'layer_widths': layer_widths
            }

        elif model_type == 'TFT':
            hidden_size = st.slider("Hidden Size", 32, 128, 64)
            num_attention_heads = st.slider("Attention Heads", 2, 8, 4)
            hyperparams = {
                'input_chunk': input_chunk, 'output_chunk': output_chunk,
                'batch_size': batch_size, 'n_epochs': n_epochs, 'learning_rate': learning_rate,
                'hidden_size': hidden_size, 'num_attention_heads': num_attention_heads
            }

        elif model_type == 'TCN':
            num_filters = st.slider("Num Filters", 32, 128, 64)
            kernel_size = st.slider("Kernel Size", 3, 7, 3)
            hyperparams = {
                'input_chunk': input_chunk, 'output_chunk': output_chunk,
                'batch_size': batch_size, 'n_epochs': n_epochs, 'learning_rate': learning_rate,
                'num_filters': num_filters, 'kernel_size': kernel_size
            }

        elif model_type == 'LSTM':
            hidden_dim = st.slider("Hidden Dim", 64, 256, 128, step=32)
            n_rnn_layers = st.slider("RNN Layers", 1, 3, 2)
            hyperparams = {
                'input_chunk': input_chunk, 'output_chunk': output_chunk,
                'batch_size': batch_size, 'n_epochs': n_epochs, 'learning_rate': learning_rate,
                'hidden_dim': hidden_dim, 'n_rnn_layers': n_rnn_layers
            }

    # Bouton d'entraînement
    if st.button("🚀 Lancer l'entraînement", type="primary", key='quick_train_btn'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_ui(prog, msg):
            progress_bar.progress(prog)
            status_text.text(msg)

        try:
            results = run_training_pipeline(
                station=station,
                model_type=model_type,
                hyperparams=hyperparams,
                fill_missing=fill_missing,
                use_covariates=use_covariates,
                progress_callback=update_ui
            )

            st.success("🎉 Modèle entraîné et ajouté au classement !")

            # Afficher métriques
            st.markdown("### 📈 Métriques sur le Test Set")
            metrics = results['metrics']
            
            cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics.items()):
                if i < 4:
                    cols[i].metric(metric, f"{value:.4f}")

            cols2 = st.columns(3)
            for i, (metric, value) in enumerate(list(metrics.items())[4:]):
                cols2[i].metric(metric, f"{value:.4f}")

            # Graphique
            st.markdown("### 📊 Prédictions vs Réalité")
            
            pred_denorm = results['scalers']['target'].inverse_transform(results['pred'])
            test_denorm = results['scalers']['target'].inverse_transform(results['test_data']['target'])

            pred_df = pred_denorm.pd_dataframe()
            test_df = test_denorm.pd_dataframe()

            fig = plot_predictions(test_df, {model_type: pred_df})
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"💾 Modèle sauvegardé : `{results['save_path'].name}`")
            
            # Navigation Buttons
            st.markdown("---")
            st.subheader("🔗 Prochaines étapes")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔮 Aller aux Prédictions"):
                    st.switch_page("pages/5_🔮_Forecasting.py")
            with c2:
                if st.button("🔄 Aller au Backtesting"):
                    st.switch_page("pages/7_🔄_Backtesting.py")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            import traceback
            st.exception(traceback.format_exc())

# ============================================================================
# TAB 2: OPTUNA OPTIMIZATION
# ============================================================================
with tab2:
    st.subheader("🔬 Optimisation Optuna")

    st.info("""
    Optuna va tester automatiquement plusieurs combinaisons d'hyperparamètres
    pour trouver la meilleure configuration.
    """)

    col1, col2 = st.columns(2)

    with col1:
        station_optuna = st.selectbox("Station", STATIONS, key='optuna_station')
        model_optuna = st.selectbox("Modèle", MODELS, key='optuna_model')

    with col2:
        n_trials = st.number_input("Nombre d'essais", 10, 200, 30, key='optuna_trials')
        timeout = st.number_input("Timeout (secondes)", 600, 7200, 3600, key='optuna_timeout')
        fill_missing_optuna = st.checkbox(
            "Interpoler les données manquantes",
            value=False,
            help="Si coché, remplit les trous par interpolation temporelle.",
            key='optuna_fill'
        )

    metric_optuna = st.selectbox(
        "Métrique à optimiser",
        options=METRICS,
        index=0,  # MAE par défaut
        key='optuna_metric'
    )

    direction = 'minimize' if METRICS_INFO[metric_optuna]['lower_is_better'] else 'maximize'

    # Afficher l'espace de recherche
    with st.expander("🔍 Espace de recherche"):
        st.code(f"""
Hyperparamètres optimisés pour {model_optuna}:

Communs:
- input_chunk: [20, 30, 60, 90]
- output_chunk: [7, 14, 30]
- batch_size: [16, 32, 64]
- learning_rate: log-uniform [1e-5, 1e-2]

Spécifiques à {model_optuna}:
{get_model_specific_space(model_optuna)}
        """)

    if st.button("🔬 Lancer l'optimisation", type="primary", key='optuna_btn'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        trial_container = st.empty()

        try:
            # Préparer les données
            status_text.text("📥 Préparation des données...")
            data = prepare_data_for_training(station_optuna, fill_missing=fill_missing_optuna)
            train_data = data['train_scaled']
            val_data = data['val_scaled']
            progress_bar.progress(10)

            # Créer l'objective
            status_text.text("🔧 Configuration d'Optuna...")
            work_dir = CHECKPOINTS_DIR / f"optuna_{model_optuna.lower()}"
            work_dir.mkdir(exist_ok=True)

            objective = create_optuna_objective(
                model_optuna, train_data, val_data, work_dir, metric=metric_optuna
            )

            progress_bar.progress(15)

            # Lancer l'étude
            status_text.text(f"🔬 Optimisation en cours ({n_trials} essais)...")

            study_name = f"{model_optuna}_{station_optuna}_{metric_optuna}"

            import optuna

            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Callback pour mettre à jour la progress bar
            def callback(study, trial):
                progress = 15 + int((trial.number / n_trials) * 80)
                progress_bar.progress(min(progress, 95))
                trial_container.text(
                    f"Trial {trial.number + 1}/{n_trials} | "
                    f"Best {metric_optuna}: {study.best_value:.4f}"
                )

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[callback],
                show_progress_bar=False
            )

            progress_bar.progress(100)
            status_text.text("✅ Optimisation terminée !")

            st.success("🎉 Optimisation terminée avec succès !")

            # Résultats
            st.markdown("### 🏆 Meilleurs Hyperparamètres")

            best_params = get_best_params(study)
            st.json(best_params)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Meilleur {metric_optuna}", f"{get_best_value(study):.4f}")
            with col2:
                st.metric("Nombre de trials complétés", len(study.trials))

            # Graphiques
            st.markdown("### 📊 Historique d'Optimisation")
            fig_history = plot_optuna_optimization_history(study)
            st.plotly_chart(fig_history, use_container_width=True)

            st.markdown("### 📊 Importance des Hyperparamètres")
            fig_importance = plot_optuna_param_importances(study)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Pas assez de trials pour calculer l'importance.")

            # Tableau des trials
            with st.expander("📋 Détails de tous les trials"):
                df_trials = get_trials_dataframe(study)
                st.dataframe(df_trials, use_container_width=True)

            # Bouton pour entraîner le modèle final
            if st.button("🚀 Entraîner le modèle final avec les meilleurs params", key='optuna_final'):
                progress_bar_final = st.progress(0)
                status_final = st.empty()
                
                def update_ui_final(prog, msg):
                    progress_bar_final.progress(prog)
                    status_final.text(msg)
                
                try:
                    # Récupérer les meilleurs hyperparamètres
                    best_params = get_best_params(study)
                    
                    # Ajouter les paramètres fixes/communs s'ils ne sont pas dans best_params
                    final_hyperparams = best_params.copy()
                    defaults = {
                        'input_chunk': 30, 'output_chunk': 7,
                        'batch_size': 32, 'n_epochs': 50, 'learning_rate': 1e-3
                    }
                    for k, v in defaults.items():
                        if k not in final_hyperparams:
                            final_hyperparams[k] = v
                    
                    results_final = run_training_pipeline(
                        station=station_optuna,
                        model_type=model_optuna,
                        hyperparams=final_hyperparams,
                        fill_missing=fill_missing_optuna,
                        use_covariates=(model_optuna in ['TFT', 'TCN', 'LSTM']),
                        progress_callback=update_ui_final
                    )
                    
                    st.success(f"Modèle sauvegardé : {results_final['save_path'].name}")
                    
                    # Afficher métriques
                    st.markdown("### 📈 Performance du Modèle Final")
                    metrics_final = results_final['metrics']
                    cols = st.columns(4)
                    for i, (metric, value) in enumerate(metrics_final.items()):
                        if i < 4:
                            cols[i].metric(metric, f"{value:.4f}")
                            
                    # Navigation Buttons
                    st.markdown("---")
                    st.subheader("🔗 Prochaines étapes")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔮 Aller aux Prédictions", key="nav_pred_optuna"):
                            st.switch_page("pages/5_🔮_Forecasting.py")
                    with c2:
                        if st.button("🔄 Aller au Backtesting", key="nav_back_optuna"):
                            st.switch_page("pages/7_🔄_Backtesting.py")
                            
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement final : {e}")
                    st.exception(e)

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            import traceback
            st.exception(traceback.format_exc())
