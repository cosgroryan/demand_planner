"""
Time series aggregation for demand planning.

This module creates demand series at different grains (SKU, Market-Channel, etc.)
and frequencies (weekly, monthly) for forecasting.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .constants import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)
console = Console()


def aggregate_sales(
    df: pl.DataFrame,
    grain: List[str] = ["SKU"],
    freq: str = "W",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate sales data into time series at specified grain and frequency.
    
    Args:
        df: Input sales DataFrame
        grain: List of columns to group by (e.g., ["SKU"], ["SKU", "Market-Channel"])
        freq: Frequency for aggregation ("W" for weekly, "M" for monthly)
        output_dir: Directory to save aggregated series (optional)
    
    Returns:
        Dictionary with aggregation statistics and file paths
    """
    console.print(f"[blue]Aggregating sales data by {grain} at {freq} frequency[/blue]")
    
    # Validate grain columns exist
    missing_cols = [col for col in grain if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in grain: {missing_cols}")
    
    # Ensure Date column exists and is properly formatted
    if "Date" not in df.columns:
        raise ValueError("Date column is required for time series aggregation")
    
    # Convert Date to datetime if needed, handling invalid dates
    if df["Date"].dtype == pl.Utf8:
        # First, filter out rows with invalid dates
        df = df.filter(
            pl.col("Date").str.len_chars() > 8  # Must be at least 8 characters for a date
        )
        
        if len(df) == 0:
            raise ValueError("No valid dates found in the data")
        
        # Try to convert to date, setting invalid dates to null
        df = df.with_columns(
            pl.col("Date").str.to_date("%Y-%m-%d", strict=False).alias("Date")
        )
        
        # Remove rows where date conversion failed
        df = df.filter(pl.col("Date").is_not_null())
        
        if len(df) == 0:
            raise ValueError("No valid dates could be parsed from the data")
    elif df["Date"].dtype == pl.Date:
        # Date is already in proper format, no conversion needed
        pass
    else:
        raise ValueError(f"Unsupported Date column type: {df['Date'].dtype}")
    
    # Create aggregation key
    grain_key = "_".join(grain)
    
    # For SKU-level aggregation, use forecast SKU (without size) if available
    if grain == ["SKU"] and "SKU_Forecast" in df.columns:
        # Use forecast SKU for better data density
        df_agg = df.with_columns([
            pl.col("SKU_Forecast").alias("SKU")
        ])
        grain_key = "SKU_Forecast"
    else:
        df_agg = df
    
    # Aggregate by grain and time period
    agg_df = _perform_aggregation(df_agg, grain, freq)
    
    # Create time series with contiguous periods
    series_df = _create_contiguous_series(agg_df, grain, freq)
    
    # Calculate derived metrics
    series_df = _add_derived_metrics(series_df)
    
    # Save if output directory specified
    file_path = None
    if output_dir:
        file_path = _save_aggregated_series(series_df, grain_key, freq, output_dir)
    
    # Generate summary statistics
    stats = _generate_aggregation_stats(series_df, grain, freq)
    
    console.print(f"[green]✓ Aggregated {len(series_df)} periods for {grain_key} at {freq} frequency[/green]")
    
    return {
        "grain": grain,
        "frequency": freq,
        "periods": len(series_df),
        "file_path": file_path,
        "stats": stats
    }


def _perform_aggregation(df: pl.DataFrame, grain: List[str], freq: str) -> pl.DataFrame:
    """Perform the core aggregation by grain and time period."""
    
    # Create time period column
    if freq == "W":
        period_col = pl.col("Date").dt.truncate("1w").alias("period")
    elif freq == "M":
        period_col = pl.col("Date").dt.truncate("1mo").alias("period")
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    # Group by grain columns and period, then aggregate
    agg_df = (
        df.with_columns(period_col)
        .group_by(grain + ["period"])
        .agg([
            pl.col("units").sum().alias("units"),
            pl.col("Amount (Net)").sum().alias("net_sales"),
            pl.col("Amount Discount").sum().alias("total_discount"),
            pl.col("units").count().alias("transactions")
        ])
        .sort(grain + ["period"])
    )
    
    return agg_df


def _create_contiguous_series(agg_df: pl.DataFrame, grain: List[str], freq: str) -> pl.DataFrame:
    """Create contiguous time series by filling missing periods with zeros."""
    
    # Get date range
    min_date = agg_df.select(pl.col("period").min()).item()
    max_date = agg_df.select(pl.col("period").max()).item()
    
    # Create full date range
    if freq == "W":
        date_range = pl.date_range(min_date, max_date, interval="1w", eager=True)
    elif freq == "M":
        date_range = pl.date_range(min_date, max_date, interval="1mo", eager=True)
    
    # Get unique grain combinations
    grain_combinations = agg_df.select(grain).unique()
    
    # Create cartesian product of grain combinations and date range
    full_index = grain_combinations.join(
        pl.DataFrame({"period": date_range}),
        how="cross"
    )
    
    # Join with aggregated data and fill missing values with zeros
    series_df = (
        full_index
        .join(agg_df, on=grain + ["period"], how="left")
        .with_columns([
            pl.col("units").fill_null(0),
            pl.col("net_sales").fill_null(0),
            pl.col("total_discount").fill_null(0),
            pl.col("transactions").fill_null(0)
        ])
        .sort(grain + ["period"])
    )
    
    return series_df


def _add_derived_metrics(series_df: pl.DataFrame) -> pl.DataFrame:
    """Add derived metrics like average selling price and discount rate."""
    
    return series_df.with_columns([
        # Average selling price (avoid division by zero)
        pl.when(pl.col("units") > 0)
        .then(pl.col("net_sales") / pl.col("units"))
        .otherwise(0.0)
        .alias("avg_selling_price"),
        
        # Discount rate
        pl.when(pl.col("net_sales") > 0)
        .then(pl.col("total_discount") / (pl.col("net_sales") + pl.col("total_discount")))
        .otherwise(0.0)
        .alias("discount_rate"),
        
        # Add year and month for partitioning
        pl.col("period").dt.year().alias("year"),
        pl.col("period").dt.month().alias("month")
    ])


def _save_aggregated_series(
    series_df: pl.DataFrame, 
    grain_key: str, 
    freq: str, 
    output_dir: str
) -> str:
    """Save aggregated series to parquet file."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"series_{grain_key}_{freq}.parquet"
    file_path = output_path / filename
    
    series_df.write_parquet(
        file_path,
        compression="snappy",
        use_pyarrow=True
    )
    
    return str(file_path)


def _generate_aggregation_stats(series_df: pl.DataFrame, grain: List[str], freq: str) -> Dict[str, Any]:
    """Generate summary statistics for the aggregated series."""
    
    stats = {
        "total_periods": len(series_df),
        "date_range": (
            series_df.select(pl.col("period").min()).item(),
            series_df.select(pl.col("period").max()).item()
        ),
        "total_units": series_df.select(pl.col("units").sum()).item(),
        "total_sales": series_df.select(pl.col("net_sales").sum()).item(),
        "avg_transactions_per_period": series_df.select(pl.col("transactions").mean()).item(),
        "zero_periods": series_df.filter(pl.col("units") == 0).height,
        "non_zero_periods": series_df.filter(pl.col("units") > 0).height
    }
    
    # Add grain-specific stats
    for col in grain:
        unique_count = series_df.select(pl.col(col).n_unique()).item()
        stats[f"unique_{col.lower()}"] = unique_count
    
    return stats


def aggregate_all_grains(
    df: pl.DataFrame,
    output_dir: str = "data/processed/series",
    frequencies: List[str] = ["W", "M"]
) -> Dict[str, Any]:
    """
    Aggregate sales data for all common grain combinations.
    
    Args:
        df: Input sales DataFrame
        output_dir: Directory to save all aggregated series
        frequencies: List of frequencies to aggregate (e.g., ["W", "M"])
    
    Returns:
        Dictionary with results for all aggregations
    """
    console.print("[blue]Running aggregation for all grain combinations[/blue]")
    
    # Define common grain combinations
    grain_combinations = [
        ["SKU"],
        ["SKU", "Market-Channel"],
        ["Style", "Colour", "Gender"],
        ["Market-Channel"],
        ["Style"],
        ["Range Segment"]
    ]
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        total_tasks = len(grain_combinations) * len(frequencies)
        task = progress.add_task("Aggregating series...", total=total_tasks)
        
        for grain in grain_combinations:
            for freq in frequencies:
                try:
                    result = aggregate_sales(
                        df=df,
                        grain=grain,
                        freq=freq,
                        output_dir=output_dir
                    )
                    
                    key = f"{'_'.join(grain)}_{freq}"
                    results[key] = result
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]Error aggregating {grain} at {freq}: {e}[/red]")
                    progress.update(task, advance=1)
    
    console.print(f"[green]✓ Completed {len(results)} aggregations[/green]")
    
    return results


def load_aggregated_series(
    grain: List[str],
    freq: str,
    data_dir: str = "data/processed/series",
    use_forecast_sku: bool = False
) -> pl.DataFrame:
    """
    Load a previously aggregated series.
    
    Args:
        grain: List of columns used for grouping
        freq: Frequency of the series ("W" or "M")
        data_dir: Directory containing the series files
        use_forecast_sku: Whether to use forecast SKU data (without sizes)
    
    Returns:
        DataFrame with the aggregated series
    """
    grain_key = "_".join(grain)
    
    # For SKU-level data, use forecast SKU if requested
    if grain == ["SKU"] and use_forecast_sku:
        filename = f"series_SKU_Forecast_{freq}.parquet"
    else:
        filename = f"series_{grain_key}_{freq}.parquet"
    
    file_path = Path(data_dir) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Series file not found: {file_path}")
    
    return pl.read_parquet(file_path)


def get_series_summary(
    data_dir: str = "data/processed/series"
) -> pl.DataFrame:
    """
    Get a summary of all available aggregated series.
    
    Args:
        data_dir: Directory containing the series files
    
    Returns:
        DataFrame with summary information about available series
    """
    series_dir = Path(data_dir)
    
    if not series_dir.exists():
        return pl.DataFrame()
    
    series_files = list(series_dir.glob("series_*.parquet"))
    
    if not series_files:
        return pl.DataFrame()
    
    summaries = []
    
    for file_path in series_files:
        try:
            df = pl.read_parquet(file_path)
            
            # Extract grain and frequency from filename
            filename = file_path.stem
            parts = filename.split("_")
            grain = "_".join(parts[1:-1])  # Everything between "series" and frequency
            freq = parts[-1]
            
            summary = {
                "file": filename,
                "grain": grain,
                "frequency": freq,
                "periods": len(df),
                "date_range_start": df.select(pl.col("period").min()).item(),
                "date_range_end": df.select(pl.col("period").max()).item(),
                "total_units": df.select(pl.col("units").sum()).item(),
                "total_sales": df.select(pl.col("net_sales").sum()).item(),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
            summaries.append(summary)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
    
    return pl.DataFrame(summaries)
