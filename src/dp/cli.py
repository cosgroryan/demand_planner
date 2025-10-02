"""
Command-line interface for the demand planning system.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from .etl_reconstruct import reconstruct_table
from .xml_parser import xml_to_parquet
from .data_cleaner import clean_sales_data
from .aggregate import aggregate_sales, aggregate_all_grains, get_series_summary
from .features import engineer_features, get_feature_summary
from .models.forecast_api import forecast, batch_forecast, DemandForecaster
from .backtest import run_backtest_cli

app = typer.Typer(
    name="dp",
    help="Demand Planner CLI - Modern demand planning system for retail sales forecasting",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def etl(
    ctx: typer.Context,
    reconstruct: bool = typer.Option(False, "--reconstruct", help="Reconstruct table from flattened data"),
    src: Optional[str] = typer.Option(None, "--src", help="Source file path"),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path"),
    sample_size: Optional[int] = typer.Option(None, "--sample-size", help="Process only N rows (for testing)"),
):
    """ETL operations for data processing."""
    
    if reconstruct:
        if not src:
            console.print("[red]Error: --src is required for reconstruct operation[/red]")
            raise typer.Exit(1)
        
        if not out:
            out = "data/processed/sales_clean.parquet"
        
        # Check if source is XML file
        if src.endswith('.xml'):
            console.print(f"[blue]Converting XML file {src} to parquet format[/blue]")
            try:
                xml_to_parquet(xml_path=src, output_path=out, sample_size=sample_size)
                console.print(f"[green]Successfully converted XML to {out}[/green]")
                
                # Extract SKU components
                from .sku_utils import extract_sku_components
                console.print(f"[blue]Extracting SKU components...[/blue]")
                import polars as pl
                df = pl.read_parquet(out)
                df_with_sku = extract_sku_components(df)
                df_with_sku.write_parquet(out, compression='snappy')
                console.print(f"[green]Successfully extracted SKU components[/green]")
                
                # Clean the data after conversion
                clean_out = out.replace('.parquet', '_clean.parquet')
                console.print(f"[blue]Cleaning data types and structure...[/blue]")
                clean_sales_data(input_path=out, output_path=clean_out, sample_size=sample_size)
                console.print(f"[green]Successfully cleaned data to {clean_out}[/green]")
                return
            except Exception as e:
                console.print(f"[red]Error converting XML: {e}[/red]")
                raise typer.Exit(1)
        
        console.print(f"[blue]Reconstructing table from {src} to {out}[/blue]")
        
        try:
            stats = reconstruct_table(
                source_path=src,
                output_path=out,
                sample_size=sample_size
            )
            
            # Display results
            table = Table(title="Reconstruction Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Rows", f"{stats['total_rows']:,}")
            table.add_row("Total Columns", f"{stats['total_columns']:,}")
            table.add_row("Unique SKUs", f"{stats['unique_skus']:,}")
            table.add_row("Data Quality Score", f"{stats['data_quality_score']:.2%}")
            
            if stats['date_range']:
                min_date, max_date = stats['date_range']
                table.add_row("Date Range", f"{min_date} to {max_date}")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error during reconstruction: {e}[/red]")
            raise typer.Exit(1)
    
    else:
        console.print("[yellow]No ETL operation specified. Use --reconstruct to reconstruct data.[/yellow]")


@app.command()
def forecast_cmd(
    sku: str = typer.Argument(..., help="SKU to forecast"),
    model: str = typer.Option("auto", "--model", help="Forecasting model type"),
    horizon: int = typer.Option(13, "--horizon", help="Forecast horizon in periods"),
    input_file: str = typer.Option("data/processed/features/features_SKU.parquet", "--input", help="Input features file"),
    output_file: Optional[str] = typer.Option(None, "--output", help="Output forecast file"),
):
    """Generate demand forecasts."""
    
    if not Path(input_file).exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Generating forecast for SKU: {sku}[/blue]")
    console.print(f"Model: {model}, Horizon: {horizon} periods")
    
    import polars as pl
    
    # Load features data
    df = pl.read_parquet(input_file)
    
    # Filter for specific SKU
    sku_data = df.filter(pl.col("SKU") == sku)
    
    if len(sku_data) == 0:
        console.print(f"[red]No data found for SKU: {sku}[/red]")
        raise typer.Exit(1)
    
    # Sort by period
    sku_data = sku_data.sort("period")
    
    console.print(f"Found {len(sku_data)} periods of data for {sku}")
    
    try:
        # Generate forecast
        result = forecast(
            series_id=sku,
            series_data=sku_data,
            horizon=horizon,
            model=model
        )
        
        # Display results
        table = Table(title=f"Forecast Results for {sku}")
        table.add_column("Period", style="cyan")
        table.add_column("Forecast", style="green")
        table.add_column("Lower Bound", style="yellow")
        table.add_column("Upper Bound", style="yellow")
        
        forecast_values = result["forecast"]
        prediction_intervals = result.get("prediction_intervals", [])
        
        for i, (forecast_val, (lower, upper)) in enumerate(zip(forecast_values, prediction_intervals)):
            table.add_row(
                f"T+{i+1}",
                f"{forecast_val:.1f}",
                f"{lower:.1f}",
                f"{upper:.1f}"
            )
        
        console.print(table)
        
        # Display model info
        console.print(f"\n[blue]Model Used: {result['model_used']}[/blue]")
        if result.get("pattern_info"):
            pattern = result["pattern_info"]["pattern"]
            console.print(f"[blue]Demand Pattern: {pattern}[/blue]")
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]✓ Forecast saved to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error generating forecast: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def aggregate(
    ctx: typer.Context,
    all_grains: bool = typer.Option(False, "--all", help="Aggregate all grain combinations"),
    grain: Optional[str] = typer.Option(None, "--grain", help="Grain columns (comma-separated)"),
    freq: str = typer.Option("W", "--freq", help="Frequency (W for weekly, M for monthly)"),
    input_file: str = typer.Option("data/processed/sales_clean.parquet", "--input", help="Input parquet file"),
    output_dir: str = typer.Option("data/processed/series", "--output-dir", help="Output directory"),
):
    """Aggregate sales data into time series."""
    
    if not Path(input_file).exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    if all_grains:
        console.print("[blue]Aggregating all grain combinations[/blue]")
        
        import polars as pl
        df = pl.read_parquet(input_file)
        
        results = aggregate_all_grains(df, output_dir)
        
        # Display summary
        table = Table(title="Aggregation Results")
        table.add_column("Grain", style="cyan")
        table.add_column("Frequency", style="green")
        table.add_column("Periods", style="yellow")
        table.add_column("File", style="blue")
        
        for key, result in results.items():
            grain_part, freq_part = key.rsplit("_", 1)
            table.add_row(
                grain_part.replace("_", ", "),
                freq_part,
                str(result["periods"]),
                Path(result["file_path"]).name if result["file_path"] else "N/A"
            )
        
        console.print(table)
        
    elif grain:
        console.print(f"[blue]Aggregating by {grain} at {freq} frequency[/blue]")
        
        import polars as pl
        df = pl.read_parquet(input_file)
        
        grain_list = [g.strip() for g in grain.split(",")]
        
        result = aggregate_sales(
            df=df,
            grain=grain_list,
            freq=freq,
            output_dir=output_dir
        )
        
        # Display results
        table = Table(title="Aggregation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Grain", ", ".join(grain_list))
        table.add_row("Frequency", freq)
        table.add_row("Periods", str(result["periods"]))
        table.add_row("Total Units", f"{result['stats']['total_units']:,}")
        table.add_row("Total Sales", f"${result['stats']['total_sales']:,.2f}")
        table.add_row("File", Path(result["file_path"]).name if result["file_path"] else "N/A")
        
        console.print(table)
        
    else:
        console.print("[yellow]Specify --grain or use --all for all combinations[/yellow]")


@app.command()
def features(
    input_file: str = typer.Option("data/processed/series/series_SKU_W.parquet", "--input", help="Input series file"),
    output_dir: str = typer.Option("data/processed/features", "--output-dir", help="Output directory"),
    value_col: str = typer.Option("units", "--value-col", help="Value column for features"),
):
    """Engineer features for demand planning."""
    
    if not Path(input_file).exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Engineering features from {input_file}[/blue]")
    
    import polars as pl
    df = pl.read_parquet(input_file)
    
    # Engineer features
    features_df = engineer_features(
        df=df,
        value_col=value_col,
        output_dir=output_dir
    )
    
    # Get feature summary
    summary = get_feature_summary(features_df)
    
    # Display results
    table = Table(title="Feature Engineering Results")
    table.add_column("Feature Type", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Total Features", str(summary["total_features"]))
    table.add_row("Calendar Features", str(summary["calendar_features"]))
    table.add_row("Rolling Features", str(summary["rolling_features"]))
    table.add_row("Lag Features", str(summary["lag_features"]))
    table.add_row("Product Features", str(summary["product_features"]))
    table.add_row("Market Features", str(summary["market_features"]))
    table.add_row("Event Features", str(summary["event_features"]))
    table.add_row("Total Rows", str(summary["total_rows"]))
    
    console.print(table)


@app.command()
def backtest(
    grain: str = typer.Option("SKU", "--grain", help="Grain columns (comma-separated)"),
    freq: str = typer.Option("W", "--freq", help="Frequency (W for weekly, M for monthly)"),
    model: str = typer.Option("auto", "--model", help="Model type to use"),
    horizon: int = typer.Option(13, "--horizon", help="Forecast horizon in periods"),
    history_window: int = typer.Option(52, "--history", help="History window size"),
    step_size: int = typer.Option(4, "--step", help="Step size between backtests"),
    data_dir: str = typer.Option("data/processed/series", "--data-dir", help="Directory containing series data"),
    output_dir: str = typer.Option("reports/backtests", "--output-dir", help="Directory to save results"),
):
    """Run rolling backtests to evaluate forecasting models."""
    
    grain_list = [g.strip() for g in grain.split(",")]
    
    try:
        results = run_backtest_cli(
            grain=grain_list,
            freq=freq,
            model=model,
            horizon=horizon,
            history_window=history_window,
            step_size=step_size,
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        if not results:
            console.print("[red]No backtest results generated[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error running backtest: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    period: str = typer.Option("monthly", "--period", help="Analysis period"),
    market: Optional[str] = typer.Option(None, "--market", help="Market filter"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file path"),
):
    """Analyze sales data patterns."""
    
    console.print(f"[blue]Analyzing sales data with {period} period[/blue]")
    
    if market:
        console.print(f"Market: {market}")
    
    # TODO: Implement actual analysis logic
    console.print("[yellow]Analysis functionality coming soon![/yellow]")


@app.command()
def api(
    serve: bool = typer.Option(False, "--serve", help="Start the API server"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
):
    """API operations."""
    
    if serve:
        console.print(f"[blue]Starting API server on {host}:{port}[/blue]")
        
        import uvicorn
        from .api import app as fastapi_app
        
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    
    else:
        console.print("[yellow]No API operation specified. Use --serve to start the server.[/yellow]")


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"[blue]Demand Planner v{__version__}[/blue]")


if __name__ == "__main__":
    app()
