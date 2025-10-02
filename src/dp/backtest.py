"""
Rolling backtests and evaluation metrics for demand planning models.

This module implements rolling origin cross-validation and computes
various accuracy metrics for forecasting models.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from .models.forecast_api import DemandForecaster, forecast
from .aggregate import load_aggregated_series
from .features import load_features

logger = logging.getLogger(__name__)
console = Console()


def rolling_backtest(
    series_data: pl.DataFrame,
    model: str = "auto",
    history_window: int = 52,
    forecast_horizon: int = 13,
    step_size: int = 4,
    min_train_size: int = 26,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Perform rolling origin cross-validation for time series forecasting.
    
    Args:
        series_data: DataFrame with time series data
        model: Model type to use for forecasting
        history_window: Number of periods to use as history for each forecast
        forecast_horizon: Number of periods to forecast ahead
        step_size: Number of periods to step forward between backtests
        min_train_size: Minimum training size required
        **model_kwargs: Additional model parameters
    
    Returns:
        Dictionary with backtest results and metrics
    """
    console.print(f"[blue]Running rolling backtest with {model} model[/blue]")
    console.print(f"History window: {history_window}, Horizon: {forecast_horizon}, Step: {step_size}")
    
    # Sort by period to ensure proper time ordering
    series_data = series_data.sort("period")
    
    # Get unique series identifiers (SKU, Market-Channel, etc.)
    id_cols = [col for col in series_data.columns if col not in ["period", "units", "net_sales", "total_discount", "transactions", "avg_selling_price", "discount_rate", "year", "month"]]
    
    if not id_cols:
        # Single series
        results = _backtest_single_series(series_data, model, history_window, forecast_horizon, step_size, min_train_size, **model_kwargs)
        return results
    else:
        # Multiple series
        results = _backtest_multiple_series(series_data, id_cols, model, history_window, forecast_horizon, step_size, min_train_size, **model_kwargs)
        return results


def _backtest_single_series(
    series_data: pl.DataFrame,
    model: str,
    history_window: int,
    forecast_horizon: int,
    step_size: int,
    min_train_size: int,
    **model_kwargs
) -> Dict[str, Any]:
    """Backtest a single time series."""
    
    total_periods = len(series_data)
    if total_periods < min_train_size + forecast_horizon:
        raise ValueError(f"Insufficient data: need at least {min_train_size + forecast_horizon} periods")
    
    # Generate backtest windows
    windows = _generate_backtest_windows(total_periods, history_window, forecast_horizon, step_size, min_train_size)
    
    all_forecasts = []
    all_actuals = []
    all_errors = []
    
    forecaster = DemandForecaster()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Running backtests...", total=len(windows))
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            try:
                # Split data
                train_data = series_data[train_start:train_end]
                test_data = series_data[test_start:test_end]
                
                # Generate forecast
                forecast_result = forecaster.forecast(
                    series_id=f"series_{i}",
                    series_data=train_data,
                    horizon=forecast_horizon,
                    model=model,
                    **model_kwargs
                )
                
                # Extract actual values
                actual_values = test_data.select("units").to_series().to_list()
                forecast_values = forecast_result["forecast"]
                
                # Calculate errors
                errors = _calculate_errors(actual_values, forecast_values)
                
                # Store results
                all_forecasts.extend(forecast_values)
                all_actuals.extend(actual_values)
                all_errors.append(errors)
                
                progress.update(task, advance=1)
                
            except Exception as e:
                logger.warning(f"Error in backtest window {i}: {e}")
                progress.update(task, advance=1)
                continue
    
    # Aggregate results
    aggregated_metrics = _aggregate_metrics(all_errors)
    
    return {
        "model": model,
        "total_windows": len(windows),
        "successful_windows": len(all_errors),
        "forecast_horizon": forecast_horizon,
        "history_window": history_window,
        "step_size": step_size,
        "metrics": aggregated_metrics,
        "detailed_errors": all_errors,
        "all_forecasts": all_forecasts,
        "all_actuals": all_actuals
    }


def _backtest_multiple_series(
    series_data: pl.DataFrame,
    id_cols: List[str],
    model: str,
    history_window: int,
    forecast_horizon: int,
    step_size: int,
    min_train_size: int,
    **model_kwargs
) -> Dict[str, Any]:
    """Backtest multiple time series."""
    
    # Group by series identifiers
    series_groups = series_data.group_by(id_cols)
    
    all_results = {}
    series_metrics = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        total_series = series_groups.height
        task = progress.add_task("Backtesting series...", total=total_series)
        
        for series_group in series_groups:
            try:
                # Get series identifier
                series_id = "_".join([str(series_group.select(col).item()) for col in id_cols])
                
                # Run backtest for this series
                result = _backtest_single_series(
                    series_group,
                    model,
                    history_window,
                    forecast_horizon,
                    step_size,
                    min_train_size,
                    **model_kwargs
                )
                
                all_results[series_id] = result
                
                # Store summary metrics
                if result["metrics"]:
                    series_metrics.append({
                        "series_id": series_id,
                        **result["metrics"]
                    })
                
                progress.update(task, advance=1)
                
            except Exception as e:
                logger.warning(f"Error backtesting series {series_id}: {e}")
                progress.update(task, advance=1)
                continue
    
    # Aggregate across all series
    if series_metrics:
        aggregated_metrics = _aggregate_series_metrics(series_metrics)
    else:
        aggregated_metrics = {}
    
    return {
        "model": model,
        "total_series": len(all_results),
        "forecast_horizon": forecast_horizon,
        "history_window": history_window,
        "step_size": step_size,
        "aggregated_metrics": aggregated_metrics,
        "series_results": all_results,
        "series_metrics": series_metrics
    }


def _generate_backtest_windows(
    total_periods: int,
    history_window: int,
    forecast_horizon: int,
    step_size: int,
    min_train_size: int
) -> List[Tuple[int, int, int, int]]:
    """Generate backtest window indices."""
    
    windows = []
    start_idx = 0
    
    while start_idx + history_window + forecast_horizon <= total_periods:
        train_start = start_idx
        train_end = start_idx + history_window
        test_start = train_end
        test_end = test_start + forecast_horizon
        
        # Ensure minimum training size
        if train_end - train_start >= min_train_size:
            windows.append((train_start, train_end, test_start, test_end))
        
        start_idx += step_size
    
    return windows


def _calculate_errors(actual: List[float], forecast: List[float]) -> Dict[str, float]:
    """Calculate various error metrics."""
    
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have same length")
    
    if len(actual) == 0:
        return {}
    
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return {}
    
    # Calculate errors
    errors = actual - forecast
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(errors / np.maximum(actual, 1e-8))) * 100
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * abs_errors / (np.abs(actual) + np.abs(forecast) + 1e-8)) * 100
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(abs_errors)
    
    # Weighted MAPE (weighted by actual values)
    total_actual = np.sum(actual)
    if total_actual > 0:
        weights = actual / total_actual
        wmape = np.sum(weights * np.abs(errors / np.maximum(actual, 1e-8))) * 100
    else:
        wmape = 0.0
    
    # Bias (Mean Error)
    bias = np.mean(errors)
    
    # Theil's U statistic
    naive_errors = actual[1:] - actual[:-1]
    naive_mae = np.mean(np.abs(naive_errors)) if len(naive_errors) > 0 else 1.0
    theil_u = mae / naive_mae if naive_mae > 0 else float('inf')
    
    return {
        "mape": mape,
        "smape": smape,
        "rmse": rmse,
        "mae": mae,
        "wmape": wmape,
        "bias": bias,
        "theil_u": theil_u,
        "n_observations": len(actual)
    }


def _aggregate_metrics(error_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple backtest windows."""
    
    if not error_list:
        return {}
    
    # Collect all metrics
    all_metrics = {}
    for errors in error_list:
        for metric, value in errors.items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)
    
    # Calculate aggregated statistics
    aggregated = {}
    for metric, values in all_metrics.items():
        if metric == "n_observations":
            aggregated[metric] = sum(values)
        else:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)
            aggregated[f"{metric}_median"] = np.median(values)
    
    return aggregated


def _aggregate_series_metrics(series_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics across multiple series."""
    
    if not series_metrics:
        return {}
    
    # Convert to DataFrame for easier aggregation
    df = pl.DataFrame(series_metrics)
    
    # Calculate aggregated statistics
    aggregated = {}
    
    for col in df.columns:
        if col == "series_id":
            continue
        
        if col == "n_observations":
            aggregated[col] = df.select(pl.col(col).sum()).item()
        else:
            aggregated[f"{col}_mean"] = df.select(pl.col(col).mean()).item()
            aggregated[f"{col}_std"] = df.select(pl.col(col).std()).item()
            aggregated[f"{col}_min"] = df.select(pl.col(col).min()).item()
            aggregated[f"{col}_max"] = df.select(pl.col(col).max()).item()
            aggregated[f"{col}_median"] = df.select(pl.col(col).median()).item()
    
    return aggregated


def save_backtest_results(
    results: Dict[str, Any],
    output_dir: str = "reports/backtests",
    filename: Optional[str] = None
) -> str:
    """Save backtest results to CSV files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{results['model']}_{timestamp}.csv"
    
    # Save aggregated metrics
    if "aggregated_metrics" in results:
        metrics_df = pl.DataFrame([results["aggregated_metrics"]])
        metrics_file = output_path / f"metrics_{filename}"
        metrics_df.write_csv(metrics_file)
    
    # Save series-level metrics
    if "series_metrics" in results and results["series_metrics"]:
        series_df = pl.DataFrame(results["series_metrics"])
        series_file = output_path / f"series_{filename}"
        series_df.write_csv(series_file)
    
    # Save detailed results
    if "series_results" in results:
        detailed_results = []
        for series_id, series_result in results["series_results"].items():
            if series_result["metrics"]:
                detailed_results.append({
                    "series_id": series_id,
                    **series_result["metrics"]
                })
        
        if detailed_results:
            detailed_df = pl.DataFrame(detailed_results)
            detailed_file = output_path / f"detailed_{filename}"
            detailed_df.write_csv(detailed_file)
    
    console.print(f"[green]✓ Backtest results saved to {output_path}[/green]")
    return str(output_path)


def display_backtest_summary(results: Dict[str, Any]):
    """Display a summary of backtest results."""
    
    table = Table(title="Backtest Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Basic info
    table.add_row("Model", results.get("model", "unknown"))
    table.add_row("Total Series", str(results.get("total_series", 1)))
    table.add_row("Forecast Horizon", str(results.get("forecast_horizon", "unknown")))
    table.add_row("History Window", str(results.get("history_window", "unknown")))
    
    # Metrics
    if "aggregated_metrics" in results:
        metrics = results["aggregated_metrics"]
        for metric in ["mape_mean", "smape_mean", "rmse_mean", "mae_mean", "wmape_mean"]:
            if metric in metrics:
                table.add_row(metric.replace("_mean", "").upper(), f"{metrics[metric]:.2f}")
    
    console.print(table)


def run_backtest_cli(
    grain: List[str],
    freq: str = "W",
    model: str = "auto",
    horizon: int = 13,
    history_window: int = 52,
    step_size: int = 4,
    data_dir: str = "data/processed/series",
    output_dir: str = "reports/backtests"
) -> Dict[str, Any]:
    """
    CLI function to run backtests.
    
    Args:
        grain: List of columns to group by (e.g., ["SKU"])
        freq: Frequency of the series ("W" or "M")
        model: Model type to use
        horizon: Forecast horizon
        history_window: History window size
        step_size: Step size between backtests
        data_dir: Directory containing series data
        output_dir: Directory to save results
    
    Returns:
        Dictionary with backtest results
    """
    console.print(f"[blue]Running backtest for {grain} at {freq} frequency[/blue]")
    
    # Load series data
    try:
        series_data = load_aggregated_series(grain, freq, data_dir)
        console.print(f"✓ Loaded series data: {len(series_data)} periods")
    except FileNotFoundError:
        console.print(f"[red]Series data not found for {grain} at {freq} frequency[/red]")
        return {}
    
    # Run backtest
    results = rolling_backtest(
        series_data=series_data,
        model=model,
        history_window=history_window,
        forecast_horizon=horizon,
        step_size=step_size
    )
    
    # Save results
    save_backtest_results(results, output_dir)
    
    # Display summary
    display_backtest_summary(results)
    
    return results
