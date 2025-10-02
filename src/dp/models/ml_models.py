"""
Machine learning forecasting models for demand planning.

This module implements tree-based models for short-term forecasting
using engineered features.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML models will be disabled.")

logger = logging.getLogger(__name__)


class RandomForestForecast:
    """Random Forest forecasting model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, target_col: str = "units") -> 'RandomForestForecast':
        """Fit Random Forest model with change detection features."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest forecasting")
        
        logger.info(f"Input DataFrame columns: {df.columns}")
        logger.info(f"Input DataFrame shape: {df.shape}")
        
        # Add change detection features
        df_with_features = self._add_change_features(df, target_col)
        
        logger.info(f"Features DataFrame columns: {df_with_features.columns}")
        logger.info(f"Features DataFrame shape: {df_with_features.shape}")
        
        # Prepare features and target
        feature_cols = [col for col in df_with_features.columns if col not in [target_col, "period", "SKU"]]
        self.feature_columns = feature_cols
        
        logger.info(f"Feature columns: {feature_cols}")
        
        # Convert to pandas for sklearn
        X = df_with_features.select(feature_cols).to_pandas()
        y = df_with_features.select(target_col).to_pandas().iloc[:, 0]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data for training")
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def _add_change_features(self, df: pl.DataFrame, target_col: str) -> pl.DataFrame:
        """Add change detection and seasonal features."""
        # Sort by period to ensure proper ordering
        df_sorted = df.sort("period")
        
        # Add basic features (period is already a date)
        try:
            df_features = df_sorted.with_columns([
                # Week of year (seasonal feature)
                pl.col("period").dt.week().alias("week_of_year"),
                # Month (seasonal feature)  
                pl.col("period").dt.month().alias("month"),
                # Quarter (seasonal feature)
                pl.col("period").dt.quarter().alias("quarter"),
            ])
        except Exception as e:
            # Fallback: if period is not a date, create simple numeric features
            logger.warning(f"Could not extract date features: {e}")
            df_features = df_sorted.with_columns([
                pl.lit(1).alias("week_of_year"),
                pl.lit(1).alias("month"),
                pl.lit(1).alias("quarter"),
            ])
        
        # Add lag features (previous week values)
        df_features = df_features.with_columns([
            pl.col(target_col).shift(1).alias("lag_1"),
            pl.col(target_col).shift(2).alias("lag_2"),
            pl.col(target_col).shift(4).alias("lag_4"),  # 4 weeks ago
            pl.col(target_col).shift(52).alias("lag_52"),  # Same week last year
        ])
        
        # Add rolling statistics (change detection features)
        df_features = df_features.with_columns([
            # Rolling mean (baseline comparison)
            pl.col(target_col).rolling_mean(window_size=4).alias("rolling_mean_4"),
            pl.col(target_col).rolling_mean(window_size=8).alias("rolling_mean_8"),
            pl.col(target_col).rolling_mean(window_size=12).alias("rolling_mean_12"),
            
            # Rolling std (volatility detection)
            pl.col(target_col).rolling_std(window_size=4).alias("rolling_std_4"),
            pl.col(target_col).rolling_std(window_size=8).alias("rolling_std_8"),
        ])
        
        # Add change detection features (after rolling stats are created)
        df_features = df_features.with_columns([
            # Change from baseline (rate change detection)
            (pl.col(target_col) - pl.col("rolling_mean_8")).alias("change_from_baseline"),
            (pl.col(target_col) / pl.col("rolling_mean_8")).alias("rate_vs_baseline"),
        ])
        
        # Add trend features
        df_features = df_features.with_columns([
            # Recent trend (last 4 weeks vs previous 4 weeks)
            (pl.col("rolling_mean_4") - pl.col("rolling_mean_4").shift(4)).alias("trend_4w"),
            # Year-over-year change
            (pl.col(target_col) - pl.col("lag_52")).alias("yoy_change"),
            (pl.col(target_col) / pl.col("lag_52")).alias("yoy_rate"),
        ])
        
        # Fill null values
        df_features = df_features.fill_null(0)
        
        return df_features
    
    def forecast(self, df: pl.DataFrame, horizon: int) -> List[float]:
        """Generate Random Forest forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.model is None:
            return [0.0] * horizon
        
        # Use the last row for prediction
        if len(df) == 0:
            return [0.0] * horizon
        
        last_row = df.tail(1)
        X = last_row.select(self.feature_columns).to_pandas()
        
        # Remove missing values
        X = X.fillna(0)
        
        # Generate forecast
        forecast_values = []
        current_features = X.iloc[0].copy()
        
        for _ in range(horizon):
            # Predict next value
            pred = self.model.predict([current_features])[0]
            forecast_values.append(max(0, pred))  # Ensure non-negative
            
            # Update features for next prediction (simplified)
            # In practice, you'd update lag features, rolling stats, etc.
            current_features = current_features.copy()
        
        return forecast_values


class GradientBoostingForecast:
    """Gradient Boosting forecasting model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, target_col: str = "units") -> 'GradientBoostingForecast':
        """Fit Gradient Boosting model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Gradient Boosting forecasting")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [target_col, "period", "SKU"]]
        self.feature_columns = feature_cols
        
        # Convert to pandas for sklearn
        X = df.select(feature_cols).to_pandas()
        y = df.select(target_col).to_pandas().iloc[:, 0]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data for training")
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def forecast(self, df: pl.DataFrame, horizon: int) -> List[float]:
        """Generate Gradient Boosting forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.model is None:
            return [0.0] * horizon
        
        # Use the last row for prediction
        if len(df) == 0:
            return [0.0] * horizon
        
        last_row = df.tail(1)
        X = last_row.select(self.feature_columns).to_pandas()
        
        # Remove missing values
        X = X.fillna(0)
        
        # Generate forecast
        forecast_values = []
        current_features = X.iloc[0].copy()
        
        for _ in range(horizon):
            # Predict next value
            pred = self.model.predict([current_features])[0]
            forecast_values.append(max(0, pred))  # Ensure non-negative
            
            # Update features for next prediction (simplified)
            current_features = current_features.copy()
        
        return forecast_values


def get_ml_forecast(
    df: pl.DataFrame,
    model_type: str = "random_forest",
    target_col: str = "units",
    horizon: int = 13,
    **kwargs
) -> Dict[str, Any]:
    """
    Get machine learning forecast using specified model.
    
    Args:
        df: DataFrame with features and target
        model_type: Type of ML model ("random_forest", "gradient_boosting")
        target_col: Name of target column
        horizon: Forecast horizon
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary with forecast results
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for ML forecasting")
    
    if model_type == "random_forest":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 10)
        random_state = kwargs.get("random_state", 42)
        model = RandomForestForecast(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    elif model_type == "gradient_boosting":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 6)
        learning_rate = kwargs.get("learning_rate", 0.1)
        random_state = kwargs.get("random_state", 42)
        model = GradientBoostingForecast(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit and forecast
    model.fit(df, target_col)
    forecast_values = model.forecast(df, horizon)
    
    return {
        "model_type": model_type,
        "forecast": forecast_values,
        "horizon": horizon,
        "model_params": kwargs
    }


def evaluate_ml_model(
    df: pl.DataFrame,
    model_type: str = "random_forest",
    target_col: str = "units",
    test_size: float = 0.2,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate ML model using train-test split.
    
    Args:
        df: DataFrame with features and target
        model_type: Type of ML model
        target_col: Name of target column
        test_size: Proportion of data to use for testing
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for ML model evaluation")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in [target_col, "period", "SKU"]]
    X = df.select(feature_cols).to_pandas()
    y = df.select(target_col).to_pandas().iloc[:, 0]
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return {"mae": float('inf'), "rmse": float('inf'), "r2": -float('inf')}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
