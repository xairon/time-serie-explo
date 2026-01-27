"""
Moniteur de progression d'entraînement pour Streamlit.

Ce module lit les métriques écrites par MetricsFileCallback et les affiche
dans Streamlit de manière réactive en utilisant des fragments Streamlit.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
import plotly.graph_objects as go


class TrainingMonitor:
    """
    Moniteur qui lit un fichier JSON de métriques et affiche la progression dans Streamlit.
    
    Cette classe est complètement séparée du processus d'entraînement et peut être
    utilisée pour monitorer n'importe quel entraînement qui écrit dans le format JSON.
    """
    
    def __init__(self, metrics_file: Path):
        """
        Args:
            metrics_file: Chemin vers le fichier JSON de métriques
        """
        self.metrics_file = Path(metrics_file)
    
    def read_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Lit les métriques depuis le fichier JSON.
        
        Returns:
            Dictionnaire de métriques ou None si le fichier n'existe pas ou est invalide
        """
        if not self.metrics_file.exists():
            return None
        
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except (json.JSONDecodeError, IOError):
            # Fichier en cours d'écriture ou invalide
            return None
    
    def display_progress(
        self,
        progress_bar,
        status_text,
        metrics_placeholder,
        chart_placeholder,
        *,
        rerun_on_complete: bool = False
    ):
        """
        Affiche la progression dans les éléments Streamlit fournis.
        
        Args:
            progress_bar: Streamlit progress bar
            status_text: Streamlit text/empty pour le statut
            metrics_placeholder: Streamlit container pour les métriques
            chart_placeholder: Streamlit container pour le graphique
            rerun_on_complete: Si True, appelle st.rerun() quand status=='completed'
                pour permettre à la page d'afficher le tableau récapitulatif.
        """
        metrics = self.read_metrics()
        
        if metrics is None:
            if status_text:
                status_text.text("⏳ En attente du démarrage de l'entraînement...")
            if progress_bar:
                progress_bar.progress(0.0, text="Waiting for training to start...")
            return
        
        status = metrics.get('status', 'unknown')
        current_epoch = metrics.get('current_epoch', 0)
        total_epochs = metrics.get('total_epochs')
        
        # Mise à jour de la barre de progression
        if progress_bar and total_epochs:
            progress = current_epoch / total_epochs if total_epochs > 0 else 0.0
            progress_text = f"Epoch {current_epoch}/{total_epochs}"
            progress_bar.progress(min(progress, 1.0), text=progress_text)
        
        # Mise à jour du texte de statut
        if status_text:
            if status == 'training':
                elapsed = metrics.get('elapsed_seconds', 0)
                eta = metrics.get('eta_seconds')
                
                train_losses = metrics.get('train_losses', [])
                val_losses = metrics.get('val_losses', [])
                
                status_msg = f"🔄 Epoch {current_epoch}"
                if total_epochs:
                    status_msg += f"/{total_epochs}"
                
                if train_losses and len(train_losses) > 0 and train_losses[-1] is not None:
                    status_msg += f" | Train Loss: {train_losses[-1]:.4f}"
                if val_losses and len(val_losses) > 0 and val_losses[-1] is not None:
                    status_msg += f" | Val Loss: {val_losses[-1]:.4f}"
                
                if eta:
                    status_msg += f" | ETA: {int(eta)}s"
                elif elapsed:
                    status_msg += f" | Elapsed: {int(elapsed)}s"
                
                status_text.text(status_msg)
            
            elif status == 'completed':
                total_time = metrics.get('total_time_seconds', 0)
                status_text.text(f"✅ Entraînement terminé en {int(total_time)}s")
                if rerun_on_complete:
                    st.rerun()
            
            elif status == 'error':
                error = metrics.get('error', 'Unknown error')
                status_text.text(f"❌ Erreur: {error}")
                if rerun_on_complete:
                    st.rerun()
        
        # Mise à jour des métriques
        if metrics_placeholder:
            train_losses = metrics.get('train_losses', [])
            val_losses = metrics.get('val_losses', [])
            
            with metrics_placeholder.container():
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Epoch", f"{current_epoch}/{total_epochs if total_epochs else '?'}")
                with cols[1]:
                    if train_losses and len(train_losses) > 0 and train_losses[-1] is not None:
                        st.metric("Train Loss", f"{train_losses[-1]:.4f}")
                    else:
                        st.metric("Train Loss", "N/A")
                with cols[2]:
                    if val_losses and len(val_losses) > 0 and val_losses[-1] is not None:
                        st.metric("Val Loss", f"{val_losses[-1]:.4f}")
                    else:
                        st.metric("Val Loss", "N/A")
        
        # Mise à jour du graphique
        if chart_placeholder:
            epochs = metrics.get('epochs', [])
            train_losses = metrics.get('train_losses', [])
            val_losses = metrics.get('val_losses', [])
            
            if epochs and (train_losses or val_losses):
                fig = go.Figure()
                
                # Train loss
                if train_losses:
                    valid_train = [(e, l) for e, l in zip(epochs, train_losses) if l is not None]
                    if valid_train:
                        epochs_train, losses_train = zip(*valid_train)
                        fig.add_trace(go.Scatter(
                            x=list(epochs_train),
                            y=list(losses_train),
                            mode='lines+markers',
                            name='Train Loss',
                            line=dict(color='#FF4B4B', width=2),
                            marker=dict(size=6)
                        ))
                
                # Val loss
                if val_losses:
                    valid_val = [(e, l) for e, l in zip(epochs, val_losses) if l is not None]
                    if valid_val:
                        epochs_val, losses_val = zip(*valid_val)
                        fig.add_trace(go.Scatter(
                            x=list(epochs_val),
                            y=list(losses_val),
                            mode='lines+markers',
                            name='Val Loss',
                            line=dict(color='#0068C9', width=2),
                            marker=dict(size=6)
                        ))
                
                fig.update_layout(
                    title="Loss Evolution",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    template="plotly_white",
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=50),
                    hovermode='x unified'
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)


def create_training_monitor_fragment(
    metrics_file: Path,
    progress_bar,
    status_text,
    metrics_placeholder,
    chart_placeholder,
    *,
    rerun_on_complete: bool = True
):
    """
    Crée un fragment Streamlit qui met à jour automatiquement le monitoring.
    
    Cette fonction doit être appelée une seule fois pour créer le fragment.
    Le fragment se mettra à jour automatiquement toutes les secondes.
    Quand l'entraînement est terminé (status=completed), st.rerun() est appelé
    pour que la page affiche le tableau récapitulatif.
    
    Args:
        metrics_file: Chemin vers le fichier JSON de métriques
        progress_bar: Streamlit progress bar
        status_text: Streamlit text/empty pour le statut
        metrics_placeholder: Streamlit container pour les métriques
        chart_placeholder: Streamlit container pour le graphique
        rerun_on_complete: Si True (défaut), rerun la page quand terminé/erreur
    
    Returns:
        La fonction du fragment (à appeler pour activer le monitoring)
    """
    @st.fragment(run_every=1.0)
    def update_monitoring():
        """Fragment qui se met à jour automatiquement toutes les secondes."""
        monitor = TrainingMonitor(metrics_file)
        monitor.display_progress(
            progress_bar=progress_bar,
            status_text=status_text,
            metrics_placeholder=metrics_placeholder,
            chart_placeholder=chart_placeholder,
            rerun_on_complete=rerun_on_complete
        )
    
    return update_monitoring
