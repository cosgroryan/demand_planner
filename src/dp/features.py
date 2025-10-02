"""
Feature engineering for demand planning.

This module creates calendar features, rolling statistics, seasonality indicators,
and product attributes for forecasting models.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


def create_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add calendar-based features to the time series data.
    
    Args:
        df: DataFrame with 'period' column containing dates
    
    Returns:
        DataFrame with additional calendar features
    """
    return df.with_columns([
        # Basic calendar features
        pl.col("period").dt.year().alias("year"),
        pl.col("period").dt.month().alias("month"),
        pl.col("period").dt.quarter().alias("quarter"),
        pl.col("period").dt.week().alias("weekofyear"),
        pl.col("period").dt.weekday().alias("dayofweek"),
        pl.col("period").dt.ordinal_day().alias("dayofyear"),
        
        # Cyclical encoding for seasonality
        (2 * pl.col("period").dt.month() * 3.14159 / 12).sin().alias("month_sin"),
        (2 * pl.col("period").dt.month() * 3.14159 / 12).cos().alias("month_cos"),
        (2 * pl.col("period").dt.week() * 3.14159 / 52).sin().alias("week_sin"),
        (2 * pl.col("period").dt.week() * 3.14159 / 52).cos().alias("week_cos"),
        
        # Season indicators
        pl.when(pl.col("period").dt.month().is_in([12, 1, 2]))
        .then(1).otherwise(0).alias("is_winter"),
        
        pl.when(pl.col("period").dt.month().is_in([3, 4, 5]))
        .then(1).otherwise(0).alias("is_spring"),
        
        pl.when(pl.col("period").dt.month().is_in([6, 7, 8]))
        .then(1).otherwise(0).alias("is_summer"),
        
        pl.when(pl.col("period").dt.month().is_in([9, 10, 11]))
        .then(1).otherwise(0).alias("is_fall"),
        
        # Holiday periods (simplified)
        pl.when(pl.col("period").dt.month() == 12)
        .then(1).otherwise(0).alias("is_holiday_season"),
        
        # End of month/quarter/year
        pl.when(pl.col("period").dt.day() >= 25)
        .then(1).otherwise(0).alias("is_month_end"),
        
        pl.when(pl.col("period").dt.month().is_in([3, 6, 9, 12]))
        .then(1).otherwise(0).alias("is_quarter_end"),
        
        pl.when(pl.col("period").dt.month() == 12)
        .then(1).otherwise(0).alias("is_year_end"),
    ])


def create_rolling_features(
    df: pl.DataFrame, 
    value_col: str = "units",
    periods: List[int] = [4, 8, 13, 26, 52]
) -> pl.DataFrame:
    """
    Create rolling statistics features for time series.
    
    Args:
        df: DataFrame with time series data
        value_col: Column to calculate rolling statistics on
        periods: List of periods for rolling calculations
    
    Returns:
        DataFrame with rolling features added
    """
    result_df = df.clone()
    
    # Sort by period to ensure proper rolling calculations
    result_df = result_df.sort("period")
    
    for period in periods:
        # Rolling mean
        result_df = result_df.with_columns([
            pl.col(value_col).rolling_mean(window_size=period).alias(f"{value_col}_ma_{period}"),
            pl.col(value_col).rolling_std(window_size=period).alias(f"{value_col}_std_{period}"),
        ])
        
        # Rolling momentum (current vs period ago)
        result_df = result_df.with_columns([
            (pl.col(value_col) / pl.col(f"{value_col}_ma_{period}") - 1).alias(f"{value_col}_momentum_{period}")
        ])
        
        # Rolling min/max
        result_df = result_df.with_columns([
            pl.col(value_col).rolling_min(window_size=period).alias(f"{value_col}_min_{period}"),
            pl.col(value_col).rolling_max(window_size=period).alias(f"{value_col}_max_{period}"),
        ])
        
        # Rolling percentiles
        result_df = result_df.with_columns([
            pl.col(value_col).rolling_quantile(0.25, window_size=period).alias(f"{value_col}_q25_{period}"),
            pl.col(value_col).rolling_quantile(0.75, window_size=period).alias(f"{value_col}_q75_{period}"),
        ])
    
    return result_df


def create_lag_features(
    df: pl.DataFrame,
    value_col: str = "units",
    lags: List[int] = [1, 2, 3, 4, 8, 13, 26, 52]
) -> pl.DataFrame:
    """
    Create lagged features for time series.
    
    Args:
        df: DataFrame with time series data
        value_col: Column to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features added
    """
    result_df = df.clone()
    
    # Sort by period
    result_df = result_df.sort("period")
    
    for lag in lags:
        result_df = result_df.with_columns([
            pl.col(value_col).shift(lag).alias(f"{value_col}_lag_{lag}")
        ])
    
    return result_df


def create_product_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create product attribute features from the original sales data.
    
    Args:
        df: DataFrame with product columns
    
    Returns:
        DataFrame with product features added
    """
    result_df = df.clone()
    
    # Product category features
    if "Style" in df.columns:
        result_df = result_df.with_columns([
            pl.col("Style").is_in(["Atmos", "Merino"]).alias("is_premium_style"),
            pl.col("Style").is_in(["Tech", "Classic"]).alias("is_standard_style"),
        ])
    
    if "Gender" in df.columns:
        result_df = result_df.with_columns([
            (pl.col("Gender") == "Mens").alias("is_mens"),
            (pl.col("Gender") == "Womens").alias("is_womens"),
            (pl.col("Gender") == "Unisex").alias("is_unisex"),
        ])
    
    if "Range Segment" in df.columns:
        result_df = result_df.with_columns([
            (pl.col("Range Segment") == "Premium").alias("is_premium_range"),
            (pl.col("Range Segment") == "Standard").alias("is_standard_range"),
            (pl.col("Range Segment") == "Budget").alias("is_budget_range"),
        ])
    
    # Color features
    if "Colour" in df.columns:
        result_df = result_df.with_columns([
            pl.col("Colour").is_in(["Black", "White"]).alias("is_neutral_color"),
            pl.col("Colour").is_in(["Blue", "Green", "Red"]).alias("is_colorful"),
        ])
    
    return result_df


def create_market_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create market and channel features.
    
    Args:
        df: DataFrame with market columns
    
    Returns:
        DataFrame with market features added
    """
    result_df = df.clone()
    
    # Market dummies
    if "Market" in df.columns:
        markets = df.select("Market").unique().to_series().to_list()
        for market in markets:
            result_df = result_df.with_columns([
                (pl.col("Market") == market).alias(f"market_{market.replace(' ', '_').replace('-', '_')}")
            ])
    
    # Channel features
    if "Market-Channel" in df.columns:
        result_df = result_df.with_columns([
            pl.col("Market-Channel").str.contains("Retail").alias("is_retail_channel"),
            pl.col("Market-Channel").str.contains("Online").alias("is_online_channel"),
            pl.col("Market-Channel").str.contains("Wholesale").alias("is_wholesale_channel"),
        ])
    
    # Customer category features
    if "Customer Category" in df.columns:
        result_df = result_df.with_columns([
            (pl.col("Customer Category") == "Own Retail").alias("is_own_retail"),
            (pl.col("Customer Category") == "Brand").alias("is_brand"),
            (pl.col("Customer Category") == "Wholesale").alias("is_wholesale"),
        ])
    
    return result_df


def create_event_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create event and promotion features.
    
    Args:
        df: DataFrame with time series data
    
    Returns:
        DataFrame with event features added
    """
    result_df = df.clone()
    
    # Discount features
    if "discount_rate" in df.columns:
        result_df = result_df.with_columns([
            (pl.col("discount_rate") > 0).alias("has_discount"),
            (pl.col("discount_rate") > 0.1).alias("has_high_discount"),
            (pl.col("discount_rate") > 0.2).alias("has_heavy_discount"),
        ])
    
    # Staff allowance features
    if "Is Staff Allowance" in df.columns:
        result_df = result_df.with_columns([
            pl.col("Is Staff Allowance").alias("is_staff_allowance")
        ])
    
    # Transaction volume features
    if "transactions" in df.columns:
        result_df = result_df.with_columns([
            (pl.col("transactions") > pl.col("transactions").mean()).alias("high_transaction_volume"),
            (pl.col("transactions") == 0).alias("no_transactions"),
        ])
    
    return result_df


def engineer_features(
    df: pl.DataFrame,
    value_col: str = "units",
    rolling_periods: List[int] = [4, 8, 13, 26, 52],
    lag_periods: List[int] = [1, 2, 3, 4, 8, 13, 26, 52],
    output_dir: Optional[str] = None
) -> pl.DataFrame:
    """
    Create all features for demand planning.
    
    Args:
        df: Input time series DataFrame
        value_col: Column to create features for
        rolling_periods: Periods for rolling statistics
        lag_periods: Periods for lag features
        output_dir: Directory to save feature data (optional)
    
    Returns:
        DataFrame with all engineered features
    """
    console.print("[blue]Engineering features for demand planning[/blue]")
    
    # Start with calendar features
    result_df = create_calendar_features(df)
    console.print("✓ Added calendar features")
    
    # Add rolling statistics
    result_df = create_rolling_features(result_df, value_col, rolling_periods)
    console.print("✓ Added rolling statistics")
    
    # Add lag features
    result_df = create_lag_features(result_df, value_col, lag_periods)
    console.print("✓ Added lag features")
    
    # Add product features
    result_df = create_product_features(result_df)
    console.print("✓ Added product features")
    
    # Add market features
    result_df = create_market_features(result_df)
    console.print("✓ Added market features")
    
    # Add event features
    result_df = create_event_features(result_df)
    console.print("✓ Added event features")
    
    # Save if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract grain information from column names
        grain_cols = [col for col in df.columns if col not in ["period", "units", "net_sales", "total_discount", "transactions", "avg_selling_price", "discount_rate", "year", "month"]]
        
        if grain_cols:
            grain_key = "_".join(grain_cols)
            filename = f"features_{grain_key}.parquet"
        else:
            filename = "features.parquet"
        
        file_path = output_path / filename
        result_df.write_parquet(file_path, compression="snappy", use_pyarrow=True)
        console.print(f"✓ Saved features to {file_path}")
    
    console.print(f"[green]✓ Feature engineering complete! {len(result_df.columns)} total columns[/green]")
    
    return result_df


def load_features(
    grain: List[str],
    data_dir: str = "data/processed/features"
) -> pl.DataFrame:
    """
    Load previously engineered features.
    
    Args:
        grain: List of columns used for grouping
        data_dir: Directory containing the feature files
    
    Returns:
        DataFrame with the engineered features
    """
    grain_key = "_".join(grain)
    filename = f"features_{grain_key}.parquet"
    file_path = Path(data_dir) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Features file not found: {file_path}")
    
    return pl.read_parquet(file_path)


def get_feature_summary(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for engineered features.
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        Dictionary with feature summary statistics
    """
    # Count different types of features
    calendar_features = [col for col in df.columns if any(x in col for x in ["year", "month", "quarter", "week", "day", "is_", "_sin", "_cos"])]
    rolling_features = [col for col in df.columns if any(x in col for x in ["_ma_", "_std_", "_momentum_", "_min_", "_max_", "_q25_", "_q75_"])]
    lag_features = [col for col in df.columns if "_lag_" in col]
    product_features = [col for col in df.columns if any(x in col for x in ["is_premium", "is_standard", "is_mens", "is_womens", "is_neutral", "is_colorful"])]
    market_features = [col for col in df.columns if any(x in col for x in ["market_", "is_retail", "is_online", "is_wholesale", "is_own_retail", "is_brand"])]
    event_features = [col for col in df.columns if any(x in col for x in ["has_discount", "is_staff", "high_transaction", "no_transactions"])]
    
    return {
        "total_features": len(df.columns),
        "calendar_features": len(calendar_features),
        "rolling_features": len(rolling_features),
        "lag_features": len(lag_features),
        "product_features": len(product_features),
        "market_features": len(market_features),
        "event_features": len(event_features),
        "total_rows": len(df),
        "feature_columns": {
            "calendar": calendar_features,
            "rolling": rolling_features,
            "lag": lag_features,
            "product": product_features,
            "market": market_features,
            "event": event_features
        }
    }
