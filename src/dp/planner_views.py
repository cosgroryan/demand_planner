"""
Planning workbook endpoints for demand planning.

This module provides planner-friendly views and exports for Excel integration
and business planning workflows.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date, timedelta
import logging
from io import StringIO, BytesIO

from .models.forecast_api import DemandForecaster, forecast
from .aggregate import load_aggregated_series
from .features import load_features

logger = logging.getLogger(__name__)


def make_planning_table(
    sku: str,
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13,
    history_periods: int = 52,
    model: str = "auto",
    include_confidence_bands: bool = True,
    include_revenue_projection: bool = True
) -> Dict[str, Any]:
    """
    Create a planner-friendly planning table with past actuals and future forecasts.
    
    Args:
        sku: SKU to create planning table for
        market_channel: Optional market channel filter
        freq: Frequency ("W" for weekly, "M" for monthly)
        horizon: Forecast horizon in periods
        history_periods: Number of historical periods to include
        model: Model type to use for forecasting
        include_confidence_bands: Whether to include confidence intervals
        include_revenue_projection: Whether to include revenue projections
    
    Returns:
        Dictionary with planning table data and metadata
    """
    logger.info(f"Creating planning table for SKU: {sku}, Market-Channel: {market_channel}")
    
    # Load historical data
    if market_channel:
        series_data = load_aggregated_series(["SKU", "Market-Channel"], freq)
        filtered_data = series_data.filter(
            (pl.col("SKU") == sku) & 
            (pl.col("Market-Channel") == market_channel)
        )
    else:
        series_data = load_aggregated_series(["SKU"], freq)
        filtered_data = series_data.filter(pl.col("SKU") == sku)
    
    if len(filtered_data) == 0:
        raise ValueError(f"No data found for SKU {sku}" + 
                        (f" and market channel {market_channel}" if market_channel else ""))
    
    # Sort by period and get recent history
    filtered_data = filtered_data.sort("period").tail(history_periods)
    
    # Generate forecast
    forecaster = DemandForecaster()
    series_id = f"{sku}_{market_channel}" if market_channel else sku
    
    forecast_result = forecaster.forecast(
        series_id=series_id,
        series_data=filtered_data,
        horizon=horizon,
        model=model
    )
    
    # Create planning table
    planning_table = _build_planning_table(
        historical_data=filtered_data,
        forecast_result=forecast_result,
        include_confidence_bands=include_confidence_bands,
        include_revenue_projection=include_revenue_projection
    )
    
    # Calculate summary statistics
    summary_stats = _calculate_planning_summary(
        historical_data=filtered_data,
        forecast_result=forecast_result
    )
    
    return {
        "sku": sku,
        "market_channel": market_channel,
        "frequency": freq,
        "planning_table": planning_table,
        "summary_stats": summary_stats,
        "forecast_metadata": {
            "model_used": forecast_result.get("model_used", "unknown"),
            "forecast_date": forecast_result.get("forecast_date", datetime.now().isoformat()),
            "confidence_level": forecast_result.get("confidence_level", 0.95),
            "pattern_info": forecast_result.get("pattern_info")
        }
    }


def _build_planning_table(
    historical_data: pl.DataFrame,
    forecast_result: Dict[str, Any],
    include_confidence_bands: bool = True,
    include_revenue_projection: bool = True
) -> pl.DataFrame:
    """Build the actual planning table DataFrame."""
    
    # Prepare historical data
    hist_df = historical_data.select([
        "period", "units", "net_sales", "avg_selling_price", "discount_rate"
    ]).with_columns([
        pl.lit("Historical").alias("data_type"),
        pl.lit(None).alias("forecast_units"),
        pl.lit(None).alias("forecast_sales"),
        pl.lit(None).alias("lower_bound"),
        pl.lit(None).alias("upper_bound"),
        pl.lit(None).alias("confidence_interval")
    ])
    
    # Prepare forecast data
    forecast_values = forecast_result["forecast"]
    prediction_intervals = forecast_result.get("prediction_intervals", [])
    
    # Create future periods
    last_period = historical_data.select("period").tail(1).item()
    freq = "1w" if "W" in str(last_period) else "1mo"  # Simple heuristic
    
    future_periods = pl.date_range(
        start=last_period + pl.duration(weeks=1 if freq == "1w" else 0, months=1 if freq == "1mo" else 0),
        interval=freq,
        length=len(forecast_values),
        eager=True
    )
    
    # Calculate forecast sales (using historical ASP)
    hist_asp = historical_data.select("avg_selling_price").mean().item()
    forecast_sales = [units * hist_asp for units in forecast_values]
    
    # Build forecast DataFrame
    forecast_data = []
    for i, (period, units, sales) in enumerate(zip(future_periods, forecast_values, forecast_sales)):
        row = {
            "period": period,
            "units": None,  # No historical units for forecast periods
            "net_sales": None,  # No historical sales for forecast periods
            "avg_selling_price": hist_asp,
            "discount_rate": None,
            "data_type": "Forecast",
            "forecast_units": units,
            "forecast_sales": sales,
        }
        
        if include_confidence_bands and i < len(prediction_intervals):
            lower, upper = prediction_intervals[i]
            row["lower_bound"] = lower
            row["upper_bound"] = upper
            row["confidence_interval"] = f"{lower:.1f} - {upper:.1f}"
        else:
            row["lower_bound"] = None
            row["upper_bound"] = None
            row["confidence_interval"] = None
        
        forecast_data.append(row)
    
    forecast_df = pl.DataFrame(forecast_data)
    
    # Combine historical and forecast data
    combined_df = pl.concat([hist_df, forecast_df])
    
    # Add derived columns
    combined_df = combined_df.with_columns([
        # Actual or forecast units
        pl.when(pl.col("data_type") == "Historical")
        .then(pl.col("units"))
        .otherwise(pl.col("forecast_units"))
        .alias("total_units"),
        
        # Actual or forecast sales
        pl.when(pl.col("data_type") == "Historical")
        .then(pl.col("net_sales"))
        .otherwise(pl.col("forecast_sales"))
        .alias("total_sales"),
        
        # Period number
        pl.int_range(1, len(combined_df) + 1).alias("period_number"),
        
        # Year and month for grouping
        pl.col("period").dt.year().alias("year"),
        pl.col("period").dt.month().alias("month"),
        pl.col("period").dt.quarter().alias("quarter")
    ])
    
    return combined_df


def _calculate_planning_summary(
    historical_data: pl.DataFrame,
    forecast_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate summary statistics for the planning table."""
    
    # Historical summary
    hist_units = historical_data.select("units").to_series().to_list()
    hist_sales = historical_data.select("net_sales").to_series().to_list()
    
    # Forecast summary
    forecast_units = forecast_result["forecast"]
    hist_asp = historical_data.select("avg_selling_price").mean().item()
    forecast_sales = [units * hist_asp for units in forecast_units]
    
    # Calculate statistics
    summary = {
        "historical": {
            "periods": len(historical_data),
            "total_units": sum(hist_units),
            "total_sales": sum(hist_sales),
            "avg_units_per_period": np.mean(hist_units),
            "avg_sales_per_period": np.mean(hist_sales),
            "std_units": np.std(hist_units),
            "std_sales": np.std(hist_sales),
            "cv_units": np.std(hist_units) / np.mean(hist_units) if np.mean(hist_units) > 0 else 0,
            "zero_periods": sum(1 for u in hist_units if u == 0),
            "non_zero_periods": sum(1 for u in hist_units if u > 0)
        },
        "forecast": {
            "periods": len(forecast_units),
            "total_units": sum(forecast_units),
            "total_sales": sum(forecast_sales),
            "avg_units_per_period": np.mean(forecast_units),
            "avg_sales_per_period": np.mean(forecast_sales),
            "std_units": np.std(forecast_units),
            "std_sales": np.std(forecast_sales)
        },
        "comparison": {
            "units_growth_rate": (np.mean(forecast_units) - np.mean(hist_units)) / np.mean(hist_units) if np.mean(hist_units) > 0 else 0,
            "sales_growth_rate": (np.mean(forecast_sales) - np.mean(hist_sales)) / np.mean(hist_sales) if np.mean(hist_sales) > 0 else 0,
            "total_units_growth": (sum(forecast_units) - sum(hist_units)) / sum(hist_units) if sum(hist_units) > 0 else 0,
            "total_sales_growth": (sum(forecast_sales) - sum(hist_sales)) / sum(hist_sales) if sum(hist_sales) > 0 else 0
        }
    }
    
    return summary


def export_planning_table_to_csv(
    planning_data: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Export planning table to CSV format."""
    
    planning_table = planning_data["planning_table"]
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sku = planning_data["sku"]
        market_channel = planning_data.get("market_channel", "all")
        filename = f"planning_table_{sku}_{market_channel}_{timestamp}.csv"
        output_path = f"reports/planning/{filename}"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas for better CSV formatting
    pandas_df = planning_table.to_pandas()
    
    # Format columns for Excel compatibility
    pandas_df["period"] = pandas_df["period"].dt.strftime("%Y-%m-%d")
    pandas_df["total_units"] = pandas_df["total_units"].round(1)
    pandas_df["total_sales"] = pandas_df["total_sales"].round(2)
    pandas_df["avg_selling_price"] = pandas_df["avg_selling_price"].round(2)
    
    # Save to CSV
    pandas_df.to_csv(output_path, index=False)
    
    logger.info(f"Planning table exported to: {output_path}")
    return output_path


def export_planning_table_to_excel(
    planning_data: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Export planning table to Excel format with multiple sheets."""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sku = planning_data["sku"]
        market_channel = planning_data.get("market_channel", "all")
        filename = f"planning_table_{sku}_{market_channel}_{timestamp}.xlsx"
        output_path = f"reports/planning/{filename}"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas
    planning_table = planning_data["planning_table"].to_pandas()
    summary_stats = planning_data["summary_stats"]
    
    # Format data for Excel
    planning_table["period"] = planning_table["period"].dt.strftime("%Y-%m-%d")
    planning_table["total_units"] = planning_table["total_units"].round(1)
    planning_table["total_sales"] = planning_table["total_sales"].round(2)
    planning_table["avg_selling_price"] = planning_table["avg_selling_price"].round(2)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main planning table
        planning_table.to_excel(writer, sheet_name='Planning Table', index=False)
        
        # Summary statistics
        summary_df = _create_summary_dataframe(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Forecast metadata
        metadata_df = _create_metadata_dataframe(planning_data["forecast_metadata"])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    logger.info(f"Planning table exported to Excel: {output_path}")
    return output_path


def _create_summary_dataframe(summary_stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary statistics DataFrame."""
    
    summary_data = []
    
    # Historical stats
    hist_stats = summary_stats["historical"]
    for key, value in hist_stats.items():
        summary_data.append({
            "Category": "Historical",
            "Metric": key.replace("_", " ").title(),
            "Value": value
        })
    
    # Forecast stats
    forecast_stats = summary_stats["forecast"]
    for key, value in forecast_stats.items():
        summary_data.append({
            "Category": "Forecast",
            "Metric": key.replace("_", " ").title(),
            "Value": value
        })
    
    # Comparison stats
    comparison_stats = summary_stats["comparison"]
    for key, value in comparison_stats.items():
        summary_data.append({
            "Category": "Comparison",
            "Metric": key.replace("_", " ").title(),
            "Value": f"{value:.2%}" if "rate" in key or "growth" in key else value
        })
    
    return pd.DataFrame(summary_data)


def _create_metadata_dataframe(metadata: Dict[str, Any]) -> pd.DataFrame:
    """Create a metadata DataFrame."""
    
    metadata_data = []
    for key, value in metadata.items():
        if key == "pattern_info" and value:
            for pattern_key, pattern_value in value.items():
                metadata_data.append({
                    "Field": f"Pattern - {pattern_key}",
                    "Value": pattern_value
                })
        else:
            metadata_data.append({
                "Field": key.replace("_", " ").title(),
                "Value": value
            })
    
    return pd.DataFrame(metadata_data)


def get_planning_table_summary(
    sku: str,
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13
) -> Dict[str, Any]:
    """Get a quick summary of planning table data without full table generation."""
    
    # Load historical data
    if market_channel:
        series_data = load_aggregated_series(["SKU", "Market-Channel"], freq)
        filtered_data = series_data.filter(
            (pl.col("SKU") == sku) & 
            (pl.col("Market-Channel") == market_channel)
        )
    else:
        series_data = load_aggregated_series(["SKU"], freq)
        filtered_data = series_data.filter(pl.col("SKU") == sku)
    
    if len(filtered_data) == 0:
        return {"error": f"No data found for SKU {sku}"}
    
    # Get basic statistics
    filtered_data = filtered_data.sort("period")
    
    total_units = filtered_data.select("units").sum().item()
    total_sales = filtered_data.select("net_sales").sum().item()
    avg_asp = filtered_data.select("avg_selling_price").mean().item()
    periods = len(filtered_data)
    
    return {
        "sku": sku,
        "market_channel": market_channel,
        "frequency": freq,
        "historical_periods": periods,
        "total_historical_units": total_units,
        "total_historical_sales": total_sales,
        "avg_selling_price": avg_asp,
        "avg_units_per_period": total_units / periods if periods > 0 else 0,
        "date_range": {
            "start": filtered_data.select("period").min().item(),
            "end": filtered_data.select("period").max().item()
        }
    }
