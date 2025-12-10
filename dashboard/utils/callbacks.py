"""Callbacks personnalisés pour l'entraînement avec Streamlit."""

import streamlit as st
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any
import time


class StreamlitProgressCallback(Callback):
    """
    Callback PyTorch Lightning pour afficher la progression dans Streamlit.

    Affiche:
    - Numéro d'epoch actuel
    - Loss d'entraînement
    - Loss de validation
    - Barre de progression
    - Courbe de loss en temps réel
    """

    def __init__(
        self,
        total_epochs: int,
        progress_bar=None,
        status_text=None,
        metrics_placeholder=None,
        chart_placeholder=None
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.metrics_placeholder = metrics_placeholder
        self.chart_placeholder = chart_placeholder

        # Historique des losses
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

        self.current_epoch = 0
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        """Appelé au début de l'entraînement."""
        self.start_time = time.time()
        if self.status_text:
            self.status_text.text(f" Starting training ({self.total_epochs} epochs)...")

    def on_train_epoch_start(self, trainer, pl_module):
        """Appelé au début de chaque epoch."""
        self.current_epoch = trainer.current_epoch + 1

    def on_train_epoch_end(self, trainer, pl_module):
        """Appelé à la fin de chaque epoch."""
        # Récupérer les losses
        train_loss = None
        val_loss = None

        # Train loss
        if 'train_loss' in trainer.callback_metrics:
            train_loss = float(trainer.callback_metrics['train_loss'])
        elif 'loss' in trainer.callback_metrics:
            train_loss = float(trainer.callback_metrics['loss'])

        # Val loss
        if 'val_loss' in trainer.callback_metrics:
            val_loss = float(trainer.callback_metrics['val_loss'])

        # Sauvegarder l'historique
        self.epochs.append(self.current_epoch)
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        # Calculer le temps écoulé et estimé
        elapsed = time.time() - self.start_time
        avg_epoch_time = elapsed / self.current_epoch
        remaining_epochs = self.total_epochs - self.current_epoch
        eta = avg_epoch_time * remaining_epochs

        # Mise à jour de la barre de progression
        progress = self.current_epoch / self.total_epochs
        if self.progress_bar:
            self.progress_bar.progress(progress)

        # Mise à jour du texte de statut
        if self.status_text:
            status_msg = f" Epoch {self.current_epoch}/{self.total_epochs}"
            if train_loss is not None:
                status_msg += f" | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                status_msg += f" | Val Loss: {val_loss:.4f}"
            status_msg += f" | ETA: {int(eta)}s"
            self.status_text.text(status_msg)

        # Mise à jour des métriques
        if self.metrics_placeholder:
            cols = self.metrics_placeholder.columns(3)
            with cols[0]:
                st.metric("Epoch", f"{self.current_epoch}/{self.total_epochs}")
            with cols[1]:
                if train_loss is not None:
                    st.metric("Train Loss", f"{train_loss:.4f}")
            with cols[2]:
                if val_loss is not None:
                    st.metric("Val Loss", f"{val_loss:.4f}")

        # Mise à jour de la courbe de loss
        if self.chart_placeholder and len(self.train_losses) > 0:
            import pandas as pd
            import plotly.graph_objects as go

            fig = go.Figure()

            # Train loss
            if len(self.train_losses) > 0:
                fig.add_trace(go.Scatter(
                    x=self.epochs[:len(self.train_losses)],
                    y=self.train_losses,
                    mode='lines+markers',
                    name='Train Loss',
                    line=dict(color='#FF4B4B', width=2),
                    marker=dict(size=6)
                ))

            # Val loss
            if len(self.val_losses) > 0:
                fig.add_trace(go.Scatter(
                    x=self.epochs[:len(self.val_losses)],
                    y=self.val_losses,
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

            self.chart_placeholder.plotly_chart(fig, use_container_width=True)

    def on_train_end(self, trainer, pl_module):
        """Appelé à la fin de l'entraînement."""
        total_time = time.time() - self.start_time

        if self.status_text:
            self.status_text.text(
                f" Entraînement terminé en {int(total_time)}s "
                f"({total_time/self.total_epochs:.1f}s/epoch)"
            )

        if self.progress_bar:
            self.progress_bar.progress(1.0)


class TrainingLogger:
    """
    Logger pour capturer les logs d'entraînement en mode non-Streamlit.
    Utilisé pour les modèles non-PyTorch Lightning.
    """

    def __init__(self):
        self.logs = []

    def log(self, message: str):
        """Ajoute un message au log."""
        self.logs.append(message)

    def get_logs(self) -> list:
        """Retourne tous les logs."""
        return self.logs

    def clear(self):
        """Efface les logs."""
        self.logs = []

    def print_logs(self):
        """Affiche tous les logs."""
        for log in self.logs:
            print(log)

