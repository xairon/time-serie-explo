"""
Template pour intégrer des modèles externes (non-Darts) dans le dashboard.

Ce fichier montre comment créer un wrapper compatible avec l'interface Darts
pour pouvoir utiliser n'importe quelle bibliothèque de forecasting.

USAGE:
1. Copier ce template
2. Implémenter les méthodes abstraites
3. Ajouter le modèle dans models_config.py et model_factory.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import pickle
import io
import pandas as pd
import numpy as np
from darts import TimeSeries


class _RestrictedUnpickler(pickle.Unpickler):
    """Restrict unpickling to safe classes only."""
    _SAFE_MODULES = {
        'numpy', 'numpy.core', 'numpy.core.multiarray',
        'sklearn', 'sklearn.preprocessing',
        'collections', 'builtins', 'copy',
        'pandas', 'pandas.core',
    }

    def find_class(self, module, name):
        if module.split('.')[0] in self._SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden unpickle: {module}.{name}"
        )


class BaseModelWrapper(ABC):
    """
    Classe de base pour wrapper des modèles externes.
    
    Implémente l'interface minimale compatible avec le dashboard:
    - fit(series, past_covariates, val_series, val_past_covariates)
    - predict(n, series, past_covariates)
    - save(path)
    - load(path) [classmethod]
    
    Les attributs requis:
    - input_chunk_length: int
    - output_chunk_length: int
    """
    
    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        **kwargs
    ):
        """
        Initialise le wrapper.
        
        Args:
            input_chunk_length: Nombre de pas de temps en entrée
            output_chunk_length: Horizon de prédiction
            **kwargs: Paramètres spécifiques au modèle
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model = None  # Le modèle externe
        self.kwargs = kwargs
        self._fitted = False
    
    @property
    def supports_past_covariates(self) -> bool:
        """Le modèle supporte-t-il les past_covariates ?"""
        return False  # Override si oui
    
    @property
    def supports_future_covariates(self) -> bool:
        """Le modèle supporte-t-il les future_covariates ?"""
        return False  # Override si oui
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Crée l'instance du modèle externe.
        
        Returns:
            Instance du modèle externe
        """
        pass
    
    @abstractmethod
    def _fit_model(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        val_series: Optional[TimeSeries] = None,
        val_past_covariates: Optional[TimeSeries] = None
    ) -> None:
        """
        Entraîne le modèle externe.
        
        Args:
            series: Série cible (TimeSeries Darts)
            past_covariates: Covariables passées
            val_series: Série de validation
            val_past_covariates: Covariables de validation
        """
        pass
    
    @abstractmethod
    def _predict_model(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None
    ) -> np.ndarray:
        """
        Génère des prédictions.
        
        Args:
            n: Nombre de pas de temps à prédire
            series: Série historique
            past_covariates: Covariables
        
        Returns:
            Array numpy de shape (n,) avec les prédictions
        """
        pass
    
    def fit(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        val_series: Optional[TimeSeries] = None,
        val_past_covariates: Optional[TimeSeries] = None,
        **kwargs
    ) -> 'BaseModelWrapper':
        """
        Entraîne le modèle (interface Darts).
        
        Args:
            series: Série cible
            past_covariates: Covariables passées
            val_series: Série de validation
            val_past_covariates: Covariables de validation
        
        Returns:
            self
        """
        self.model = self._create_model()
        self._fit_model(series, past_covariates, val_series, val_past_covariates)
        self._fitted = True
        return self
    
    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        **kwargs
    ) -> TimeSeries:
        """
        Génère des prédictions (interface Darts).
        
        Args:
            n: Horizon de prédiction
            series: Série historique (optionnel si déjà fitté)
            past_covariates: Covariables
        
        Returns:
            TimeSeries avec les prédictions
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self._predict_model(n, series, past_covariates)
        
        # Convertir en TimeSeries Darts
        if series is not None:
            last_date = series.end_time()
            freq = series.freq
        else:
            raise ValueError("series is required for prediction")
        
        # Créer l'index temporel pour les prédictions
        pred_index = pd.date_range(
            start=last_date + freq,
            periods=n,
            freq=freq
        )
        
        pred_df = pd.DataFrame(
            {'prediction': predictions},
            index=pred_index
        )
        
        return TimeSeries.from_dataframe(pred_df, freq=freq)
    
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin de sauvegarde
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'input_chunk_length': self.input_chunk_length,
                'output_chunk_length': self.output_chunk_length,
                'kwargs': self.kwargs,
                '_fitted': self._fitted
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModelWrapper':
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin du modèle
        
        Returns:
            Instance du wrapper avec le modèle chargé
        """
        # WARNING: pickle.load can execute arbitrary code; use restricted unpickler
        with open(path, 'rb') as f:
            data = _RestrictedUnpickler(f).load()
        
        instance = cls(
            input_chunk_length=data['input_chunk_length'],
            output_chunk_length=data['output_chunk_length'],
            **data['kwargs']
        )
        instance.model = data['model']
        instance._fitted = data['_fitted']
        return instance


# =============================================================================
# EXEMPLE: Wrapper pour scikit-learn
# =============================================================================

class SklearnModelWrapper(BaseModelWrapper):
    """
    Exemple de wrapper pour un modèle scikit-learn.
    
    Convertit les TimeSeries en features X (lags) et y pour sklearn.
    """
    
    def __init__(
        self,
        sklearn_model_class,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        **sklearn_kwargs
    ):
        """
        Args:
            sklearn_model_class: Classe sklearn (ex: RandomForestRegressor)
            input_chunk_length: Nombre de lags à utiliser
            output_chunk_length: Horizon
            **sklearn_kwargs: Paramètres pour le modèle sklearn
        """
        super().__init__(input_chunk_length, output_chunk_length, **sklearn_kwargs)
        self.sklearn_model_class = sklearn_model_class
        self.sklearn_kwargs = sklearn_kwargs
    
    def _create_model(self):
        return self.sklearn_model_class(**self.sklearn_kwargs)
    
    def _series_to_features(self, series: TimeSeries) -> tuple:
        """Convertit une TimeSeries en features X, y pour sklearn."""
        values = series.values().flatten()
        
        X, y = [], []
        for i in range(self.input_chunk_length, len(values)):
            X.append(values[i - self.input_chunk_length:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def _fit_model(self, series, past_covariates=None, val_series=None, val_past_covariates=None):
        X, y = self._series_to_features(series)
        self.model.fit(X, y)
    
    def _predict_model(self, n, series, past_covariates=None):
        values = series.values().flatten()
        predictions = []
        
        # Prédiction autorégressive
        current_input = values[-self.input_chunk_length:].copy()
        
        for _ in range(n):
            pred = self.model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            # Shift et ajouter la prédiction
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred
        
        return np.array(predictions)


# =============================================================================
# EXEMPLE: Wrapper pour statsmodels
# =============================================================================

class StatsmodelsARIMAWrapper(BaseModelWrapper):
    """
    Exemple de wrapper pour ARIMA de statsmodels.
    """
    
    def __init__(
        self,
        order: tuple = (1, 1, 1),
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        **kwargs
    ):
        super().__init__(input_chunk_length, output_chunk_length, **kwargs)
        self.order = order
    
    def _create_model(self):
        # Le modèle est créé pendant le fit
        return None
    
    def _fit_model(self, series, past_covariates=None, val_series=None, val_past_covariates=None):
        from statsmodels.tsa.arima.model import ARIMA
        
        values = series.values().flatten()
        self.model = ARIMA(values, order=self.order)
        self.fitted_model = self.model.fit()
    
    def _predict_model(self, n, series, past_covariates=None):
        forecast = self.fitted_model.forecast(steps=n)
        return np.array(forecast)


# =============================================================================
# COMMENT INTÉGRER UN NOUVEAU WRAPPER
# =============================================================================
"""
1. Créer votre wrapper en héritant de BaseModelWrapper

2. Dans models_config.py, ajouter:
   
   'MonModele': {
       'name': 'Mon Modèle Custom',
       'class': 'MonModeleWrapper',
       'supports_covariates': False,
       ...
   }

3. Dans model_factory.py, ajouter une logique spéciale:

   @classmethod
   def create_model(cls, model_name, hyperparams, ...):
       ...
       # Avant le try/except existant:
       if model_name == 'MonModele':
           from dashboard.utils.custom_wrappers import MonModeleWrapper
           return MonModeleWrapper(**hyperparams)
       ...

4. Le wrapper sera automatiquement compatible avec:
   - L'entraînement
   - La sauvegarde/chargement
   - Le forecasting
   - (Partiellement) l'explicabilité
"""
