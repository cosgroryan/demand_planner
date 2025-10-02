"""
Baseline forecasting models for demand planning.

These models provide simple but effective baselines for comparison
with more sophisticated forecasting approaches.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NaiveForecast:
    """Naive forecast - uses the last observed value."""
    
    def __init__(self):
        self.last_value = None
        self.is_fitted = False
    
    def fit(self, series: pl.Series) -> 'NaiveForecast':
        """Fit the naive model."""
        if len(series) == 0:
            raise ValueError("Series cannot be empty")
        
        # Get the last non-null value
        non_null_values = series.drop_nulls()
        if len(non_null_values) == 0:
            self.last_value = 0.0
        else:
            self.last_value = non_null_values[-1]
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> List[float]:
        """Generate naive forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        return [self.last_value] * horizon


class SeasonalNaiveForecast:
    """Seasonal naive forecast - uses the value from the same season last year."""
    
    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self.seasonal_values = []
        self.is_fitted = False
    
    def fit(self, series: pl.Series) -> 'SeasonalNaiveForecast':
        """Fit the seasonal naive model."""
        if len(series) < self.season_length:
            raise ValueError(f"Series must have at least {self.season_length} observations")
        
        # Extract seasonal values
        self.seasonal_values = series.tail(self.season_length).to_list()
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> List[float]:
        """Generate seasonal naive forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = []
        for i in range(horizon):
            seasonal_index = i % self.season_length
            forecast.append(self.seasonal_values[seasonal_index])
        
        return forecast


class MovingAverageForecast:
    """Moving average forecast."""
    
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.average = None
        self.is_fitted = False
    
    def fit(self, series: pl.Series) -> 'MovingAverageForecast':
        """Fit the moving average model."""
        if len(series) < self.window_size:
            raise ValueError(f"Series must have at least {self.window_size} observations")
        
        # Calculate moving average from last window_size values
        self.average = series.tail(self.window_size).mean()
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> List[float]:
        """Generate moving average forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        return [self.average] * horizon


class CrostonForecast:
    """Croston's method for intermittent demand forecasting."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.demand_rate = None
        self.interval_rate = None
        self.is_fitted = False
    
    def fit(self, series: pl.Series) -> 'CrostonForecast':
        """Fit Croston's method."""
        values = series.to_list()
        
        if len(values) == 0:
            raise ValueError("Series cannot be empty")
        
        # Separate demand and intervals
        demand_periods = [v for v in values if v > 0]
        intervals = []
        
        current_interval = 0
        for v in values:
            if v > 0:
                if current_interval > 0:
                    intervals.append(current_interval)
                current_interval = 1
            else:
                current_interval += 1
        
        if len(demand_periods) == 0:
            self.demand_rate = 0.0
            self.interval_rate = len(values)
        else:
            # Initialize with simple averages
            self.demand_rate = np.mean(demand_periods)
            self.interval_rate = np.mean(intervals) if intervals else len(values)
            
            # Apply exponential smoothing
            for i, demand in enumerate(demand_periods[1:], 1):
                self.demand_rate = self.alpha * demand + (1 - self.alpha) * self.demand_rate
            
            for i, interval in enumerate(intervals[1:], 1):
                self.interval_rate = self.alpha * interval + (1 - self.alpha) * self.interval_rate
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> List[float]:
        """Generate Croston forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.interval_rate == 0:
            return [0.0] * horizon
        
        # Forecast demand rate per period
        demand_per_period = self.demand_rate / self.interval_rate
        return [demand_per_period] * horizon


class ExponentialSmoothingForecast:
    """Holt-Winters exponential smoothing with seasonality."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1, season_length: int = 52):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing  
        self.gamma = gamma  # Seasonal smoothing
        self.season_length = season_length
        self.level = None
        self.trend = None
        self.seasonal = None
        self.is_fitted = False
    
    def fit(self, series: pl.Series) -> 'ExponentialSmoothingForecast':
        """Fit Holt-Winters exponential smoothing model."""
        values = series.drop_nulls().to_list()
        
        if len(values) < self.season_length * 2:
            # Fallback to simple exponential smoothing for short series
            self.level = values[0] if values else 0
            for value in values[1:]:
                self.level = self.alpha * value + (1 - self.alpha) * self.level
            self.trend = 0
            self.seasonal = [0] * self.season_length
        else:
            # Initialize with simple averages
            self.level = np.mean(values[:self.season_length])
            self.trend = (np.mean(values[self.season_length:2*self.season_length]) - 
                         np.mean(values[:self.season_length])) / self.season_length
            
            # Initialize seasonal components
            self.seasonal = []
            for i in range(self.season_length):
                seasonal_values = [values[j] for j in range(i, len(values), self.season_length)]
                self.seasonal.append(np.mean(seasonal_values) - self.level)
            
            # Apply Holt-Winters smoothing
            for i in range(self.season_length, len(values)):
                prev_level = self.level
                prev_trend = self.trend
                prev_seasonal = self.seasonal[i % self.season_length]
                
                # Update level
                self.level = (self.alpha * (values[i] - prev_seasonal) + 
                             (1 - self.alpha) * (prev_level + prev_trend))
                
                # Update trend
                self.trend = (self.beta * (self.level - prev_level) + 
                             (1 - self.beta) * prev_trend)
                
                # Update seasonal
                self.seasonal[i % self.season_length] = (self.gamma * (values[i] - self.level) + 
                                                        (1 - self.gamma) * prev_seasonal)
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> List[float]:
        """Generate Holt-Winters forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.trend is None:  # Simple exponential smoothing fallback
            return [self.level] * horizon
        
        forecast_values = []
        for h in range(1, horizon + 1):
            seasonal_idx = (h - 1) % self.season_length
            forecast_value = self.level + h * self.trend + self.seasonal[seasonal_idx]
            forecast_values.append(max(0, forecast_value))  # Ensure non-negative
        
        return forecast_values


def get_baseline_forecast(
    series: pl.Series,
    model_type: str = "naive",
    horizon: int = 13,
    **kwargs
) -> Dict[str, Any]:
    """
    Get baseline forecast using specified model.
    
    Args:
        series: Time series data
        model_type: Type of baseline model ("naive", "seasonal_naive", "moving_average", "croston", "exponential_smoothing")
        horizon: Forecast horizon
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary with forecast results
    """
    if model_type == "naive":
        model = NaiveForecast()
    elif model_type == "seasonal_naive":
        season_length = kwargs.get("season_length", 52)
        model = SeasonalNaiveForecast(season_length=season_length)
    elif model_type == "moving_average":
        window_size = kwargs.get("window_size", 4)
        model = MovingAverageForecast(window_size=window_size)
    elif model_type == "croston":
        alpha = kwargs.get("alpha", 0.1)
        model = CrostonForecast(alpha=alpha)
    elif model_type == "exponential_smoothing":
        alpha = kwargs.get("alpha", 0.3)
        model = ExponentialSmoothingForecast(alpha=alpha)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit and forecast
    model.fit(series)
    forecast_values = model.forecast(horizon)
    
    return {
        "model_type": model_type,
        "forecast": forecast_values,
        "horizon": horizon,
        "model_params": kwargs
    }


def evaluate_baseline_models(
    series: pl.Series,
    test_size: int = 13,
    models: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple baseline models on a time series.
    
    Args:
        series: Time series data
        test_size: Number of observations to use for testing
        models: List of model types to evaluate
    
    Returns:
        Dictionary with evaluation metrics for each model
    """
    if models is None:
        models = ["naive", "seasonal_naive", "moving_average", "croston", "exponential_smoothing"]
    
    if len(series) < test_size + 1:
        raise ValueError(f"Series must have at least {test_size + 1} observations")
    
    # Split data
    train_series = series[:-test_size]
    test_series = series[-test_size:]
    
    results = {}
    
    for model_type in models:
        try:
            # Get forecast
            forecast_result = get_baseline_forecast(
                train_series,
                model_type=model_type,
                horizon=test_size
            )
            
            forecast_values = forecast_result["forecast"]
            actual_values = test_series.to_list()
            
            # Calculate metrics
            mape = calculate_mape(actual_values, forecast_values)
            smape = calculate_smape(actual_values, forecast_values)
            rmse = calculate_rmse(actual_values, forecast_values)
            
            results[model_type] = {
                "mape": mape,
                "smape": smape,
                "rmse": rmse
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating {model_type}: {e}")
            results[model_type] = {
                "mape": float('inf'),
                "smape": float('inf'),
                "rmse": float('inf')
            }
    
    return results


def calculate_mape(actual: List[float], forecast: List[float]) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have same length")
    
    errors = []
    for a, f in zip(actual, forecast):
        if a != 0:
            errors.append(abs((a - f) / a))
        else:
            errors.append(abs(f) if f != 0 else 0)
    
    return np.mean(errors) * 100


def calculate_smape(actual: List[float], forecast: List[float]) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have same length")
    
    errors = []
    for a, f in zip(actual, forecast):
        denominator = (abs(a) + abs(f)) / 2
        if denominator != 0:
            errors.append(abs(a - f) / denominator)
        else:
            errors.append(0)
    
    return np.mean(errors) * 100


def calculate_rmse(actual: List[float], forecast: List[float]) -> float:
    """Calculate Root Mean Square Error."""
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have same length")
    
    squared_errors = [(a - f) ** 2 for a, f in zip(actual, forecast)]
    return np.sqrt(np.mean(squared_errors))
