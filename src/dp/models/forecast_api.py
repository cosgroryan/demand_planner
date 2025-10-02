"""
Unified forecasting API that automatically selects the best model
based on demand pattern characteristics.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .baselines import (
    get_baseline_forecast, 
    evaluate_baseline_models,
    calculate_mape, 
    calculate_smape, 
    calculate_rmse
)
from .stat_models import (
    get_statistical_forecast,
    detect_demand_pattern
)
from .ml_models import get_ml_forecast

logger = logging.getLogger(__name__)


class DemandForecaster:
    """Unified demand forecasting system."""
    
    def __init__(self):
        self.model_cache = {}
        self.pattern_cache = {}
    
    def forecast(
        self,
        series_id: str,
        series_data: pl.DataFrame,
        horizon: int = 13,
        model: str = "auto",
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate forecast for a time series.
        
        Args:
            series_id: Unique identifier for the series
            series_data: DataFrame with time series data
            horizon: Forecast horizon in periods
            model: Model type ("auto", "naive", "seasonal_naive", "arima", "sarimax", "random_forest", etc.)
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional model parameters
        
        Returns:
            Dictionary with forecast results
        """
        # Extract time series
        if "units" not in series_data.columns:
            raise ValueError("'units' column is required in series_data")
        
        series = series_data.select("units").to_series()
        
        # Detect demand pattern if using auto model
        if model == "auto":
            pattern_info = self._detect_pattern(series_id, series)
            model = self._select_best_model(pattern_info, series)
        
        # Generate forecast based on model type
        if model in ["naive", "seasonal_naive", "moving_average", "croston", "exponential_smoothing"]:
            result = get_baseline_forecast(series, model_type=model, horizon=horizon, **kwargs)
        elif model in ["arima", "sarimax", "auto_arima"]:
            result = get_statistical_forecast(series, model_type=model, horizon=horizon, **kwargs)
        elif model in ["random_forest", "gradient_boosting"]:
            result = get_ml_forecast(series_data, model_type=model, horizon=horizon, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model}")
        
        # Add metadata
        result.update({
            "series_id": series_id,
            "model_used": model,
            "confidence_level": confidence_level,
            "forecast_date": datetime.now().isoformat(),
            "pattern_info": self._detect_pattern(series_id, series) if model == "auto" else None
        })
        
        # Add prediction intervals (simplified)
        if "forecast" in result:
            forecast_values = result["forecast"]
            std_dev = np.std(forecast_values) if len(forecast_values) > 1 else 1.0
            
            # Simple prediction intervals based on historical volatility
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            margin = z_score * std_dev
            
            result["prediction_intervals"] = [
                (max(0, f - margin), f + margin) for f in forecast_values
            ]
        
        return result
    
    def _detect_pattern(self, series_id: str, series: pl.Series) -> Dict[str, Any]:
        """Detect demand pattern for a series."""
        if series_id in self.pattern_cache:
            return self.pattern_cache[series_id]
        
        pattern_info = detect_demand_pattern(series)
        self.pattern_cache[series_id] = pattern_info
        return pattern_info
    
    def _select_best_model(self, pattern_info: Dict[str, Any], series: pl.Series) -> str:
        """Select the best model based on demand pattern."""
        pattern = pattern_info.get("pattern", "regular")
        intermittency = pattern_info.get("intermittency", 0)
        cv = pattern_info.get("cv", 1.0)
        
        # Model selection logic based on demand characteristics
        if intermittency > 0.5:
            # High intermittency - use Croston's method
            return "croston"
        elif pattern == "smooth" and cv < 0.3:
            # Smooth demand - use simple methods
            return "exponential_smoothing"
        elif pattern == "erratic" or cv > 1.5:
            # Erratic demand - use robust methods
            return "moving_average"
        elif len(series) >= 52:
            # Enough data for seasonal models - use exponential smoothing with seasonality
            return "exponential_smoothing"
        else:
            # Default to naive for short series
            return "naive"
    
    def evaluate_models(
        self,
        series_data: pl.DataFrame,
        test_size: int = 13,
        models: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models on a time series.
        
        Args:
            series_data: DataFrame with time series data
            test_size: Number of periods to use for testing
            models: List of models to evaluate
        
        Returns:
            Dictionary with evaluation results for each model
        """
        if models is None:
            models = [
                "naive", "seasonal_naive", "moving_average", "croston",
                "exponential_smoothing", "auto_arima", "random_forest"
            ]
        
        series = series_data.select("units").to_series()
        
        if len(series) < test_size + 1:
            raise ValueError(f"Series must have at least {test_size + 1} observations")
        
        # Split data
        train_series = series[:-test_size]
        test_series = series[-test_size:]
        train_data = series_data[:-test_size]
        
        results = {}
        
        for model in models:
            try:
                if model in ["naive", "seasonal_naive", "moving_average", "croston", "exponential_smoothing"]:
                    # Baseline models
                    forecast_result = get_baseline_forecast(
                        train_series, model_type=model, horizon=test_size
                    )
                elif model in ["arima", "sarimax", "auto_arima"]:
                    # Statistical models
                    forecast_result = get_statistical_forecast(
                        train_series, model_type=model, horizon=test_size
                    )
                elif model in ["random_forest", "gradient_boosting"]:
                    # ML models
                    forecast_result = get_ml_forecast(
                        train_data, model_type=model, horizon=test_size
                    )
                else:
                    continue
                
                forecast_values = forecast_result["forecast"]
                actual_values = test_series.to_list()
                
                # Calculate metrics
                mape = calculate_mape(actual_values, forecast_values)
                smape = calculate_smape(actual_values, forecast_values)
                rmse = calculate_rmse(actual_values, forecast_values)
                
                results[model] = {
                    "mape": mape,
                    "smape": smape,
                    "rmse": rmse
                }
                
            except Exception as e:
                logger.warning(f"Error evaluating {model}: {e}")
                results[model] = {
                    "mape": float('inf'),
                    "smape": float('inf'),
                    "rmse": float('inf')
                }
        
        return results
    
    def get_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Get the best model based on evaluation results."""
        if not evaluation_results:
            return "naive"
        
        # Use MAPE as primary metric
        best_model = min(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x].get("mape", float('inf')))
        
        return best_model


def forecast(
    series_id: str,
    series_data: pl.DataFrame,
    horizon: int = 13,
    model: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for forecasting.
    
    Args:
        series_id: Unique identifier for the series
        series_data: DataFrame with time series data
        horizon: Forecast horizon in periods
        model: Model type ("auto" for automatic selection)
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary with forecast results
    """
    forecaster = DemandForecaster()
    return forecaster.forecast(series_id, series_data, horizon, model, **kwargs)


def batch_forecast(
    series_dict: Dict[str, pl.DataFrame],
    horizon: int = 13,
    model: str = "auto",
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Generate forecasts for multiple series.
    
    Args:
        series_dict: Dictionary mapping series_id to series_data
        horizon: Forecast horizon in periods
        model: Model type ("auto" for automatic selection)
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary mapping series_id to forecast results
    """
    forecaster = DemandForecaster()
    results = {}
    
    for series_id, series_data in series_dict.items():
        try:
            result = forecaster.forecast(series_id, series_data, horizon, model, **kwargs)
            results[series_id] = result
        except Exception as e:
            logger.error(f"Error forecasting {series_id}: {e}")
            results[series_id] = {
                "error": str(e),
                "series_id": series_id,
                "model_used": model
            }
    
    return results
