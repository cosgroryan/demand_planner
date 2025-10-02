"""
ETL pipeline to reconstruct proper table from flattened XML data.

The original XML was parsed cell-by-cell, creating a flattened structure where
each row represents a single cell rather than a complete transaction record.
This module reconstructs the proper table structure.
"""

import polars as pl
import duckdb
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, date
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .constants import CANONICAL_COLUMNS, DT_COLUMNS, NUM_COLUMNS, STR_COLUMNS
from .schemas import SalesRow
from .sku_utils import extract_sku_components

logger = logging.getLogger(__name__)
console = Console()


def reconstruct_table(
    source_path: str,
    output_path: str,
    chunk_size: int = 100_000,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Reconstruct proper table from flattened XML data.
    
    Args:
        source_path: Path to source parquet file
        output_path: Path for output parquet file
        chunk_size: Number of rows to process at once
        sample_size: If provided, only process this many rows (for testing)
    
    Returns:
        Dictionary with reconstruction statistics
    """
    source_file = Path(source_path)
    output_file = Path(output_path)
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Reconstructing table from {source_path}[/bold blue]")
    
    # Read the flattened data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reading source data...", total=None)
        
        # Use DuckDB for efficient large file processing
        conn = duckdb.connect()
        
        # Get total row count
        total_rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{source_path}')").fetchone()[0]
        console.print(f"Total rows in source: {total_rows:,}")
        
        if sample_size:
            total_rows = min(sample_size, total_rows)
            console.print(f"Processing sample of {total_rows:,} rows")
        
        # Read the data column which contains the actual values
        data_col = '{urn:schemas-microsoft-com:office:spreadsheet}Data'
        data_type_col = '{urn:schemas-microsoft-com:office:spreadsheet}Data_{urn:schemas-microsoft-com:office:spreadsheet}Type'
        
        # Get all data values
        query = f"""
        SELECT 
            "{data_col}" as data_value,
            "{data_type_col}" as data_type,
            row_number() OVER (ORDER BY (SELECT 1)) as row_num
        FROM read_parquet('{source_path}')
        WHERE "{data_col}" IS NOT NULL
        """
        
        if sample_size:
            query += f" LIMIT {sample_size}"
        
        df = conn.execute(query).pl()
        
        progress.update(task, description="Processing data values...")
        
        # Identify headers and data patterns
        headers = identify_headers(df)
        console.print(f"Identified {len(headers)} potential headers")
        
        # Reconstruct rows
        reconstructed_data = reconstruct_rows(df, headers, chunk_size)
        
        progress.update(task, description="Saving reconstructed data...")
        
        # Process SKU components
        progress.update(task, description="Processing SKU components...")
        reconstructed_data = extract_sku_components(reconstructed_data)
        
        # Save to parquet with partitioning
        save_reconstructed_data(reconstructed_data, output_file)
        
        conn.close()
    
    # Generate summary statistics
    stats = generate_summary_stats(output_file)
    
    console.print(f"[bold green]✓ Reconstruction complete![/bold green]")
    console.print(f"Output saved to: {output_file}")
    console.print(f"Reconstructed rows: {stats['total_rows']:,}")
    
    return stats


def identify_headers(df: pl.DataFrame) -> list[str]:
    """Identify potential column headers from the data."""
    
    # Headers are typically text values that appear early in the data
    # and are not numeric or date-like
    data_values = df.select("data_value").unique().to_series().to_list()
    
    headers = []
    for value in data_values:
        value_str = str(value).strip()
        
        # Skip if empty or very short
        if len(value_str) < 2:
            continue
            
        # Skip if looks like a number
        if value_str.replace('.', '').replace('-', '').replace('/', '').isdigit():
            continue
            
        # Skip if looks like a date
        if any(char in value_str for char in ['-', '/']) and len(value_str) > 8:
            try:
                datetime.strptime(value_str[:10], '%Y-%m-%d')
                continue
            except:
                pass
        
        # Skip if looks like a SKU (contains numbers and dashes)
        if '-' in value_str and any(c.isdigit() for c in value_str):
            continue
            
        # This looks like a header
        headers.append(value_str)
    
    # Filter to likely headers based on our known column names
    known_headers = set(CANONICAL_COLUMNS)
    likely_headers = [h for h in headers if h in known_headers]
    
    return likely_headers


def reconstruct_rows(df: pl.DataFrame, headers: list[str], chunk_size: int) -> pl.DataFrame:
    """Reconstruct proper rows from flattened data."""
    
    # This is a simplified reconstruction - in practice, you'd need to
    # analyze the XML structure more carefully to understand how cells
    # map to rows and columns
    
    # For now, we'll create a basic structure based on the headers
    # and sample data patterns
    
    data_values = df.select("data_value").to_series().to_list()
    
    # Group data into chunks that might represent rows
    rows = []
    current_row = {}
    header_index = 0
    
    for i, value in enumerate(data_values):
        value_str = str(value).strip()
        
        # If we hit a header, start a new row
        if value_str in headers:
            if current_row:
                rows.append(current_row)
            current_row = {value_str: None}
            header_index = 0
        else:
            # Add value to current row
            if header_index < len(CANONICAL_COLUMNS):
                current_row[CANONICAL_COLUMNS[header_index]] = value_str
                header_index += 1
    
    # Add the last row
    if current_row:
        rows.append(current_row)
    
    # Convert to DataFrame
    if rows:
        reconstructed_df = pl.DataFrame(rows)
        
        # Ensure all canonical columns are present
        for col in CANONICAL_COLUMNS:
            if col not in reconstructed_df.columns:
                reconstructed_df = reconstructed_df.with_columns(pl.lit(None).alias(col))
        
        # Reorder columns
        reconstructed_df = reconstructed_df.select(CANONICAL_COLUMNS)
        
        return reconstructed_df
    else:
        # Return empty DataFrame with correct schema
        return pl.DataFrame({col: [] for col in CANONICAL_COLUMNS})


def save_reconstructed_data(df: pl.DataFrame, output_path: Path) -> None:
    """Save reconstructed data to parquet with partitioning."""
    
    # Add year and month columns for partitioning
    if "Date" in df.columns:
        try:
            df = df.with_columns([
                pl.col("Date").str.to_date("%Y-%m-%d").dt.year().alias("year"),
                pl.col("Date").str.to_date("%Y-%m-%d").dt.month().alias("month")
            ])
        except:
            # If date parsing fails, use default values
            df = df.with_columns([
                pl.lit(2023).alias("year"),
                pl.lit(1).alias("month")
            ])
    else:
        df = df.with_columns([
            pl.lit(2023).alias("year"),
            pl.lit(1).alias("month")
        ])
    
    # Save to parquet
    df.write_parquet(
        output_path,
        compression="snappy",
        use_pyarrow=True
    )


def generate_summary_stats(output_path: Path) -> Dict[str, Any]:
    """Generate summary statistics for the reconstructed data."""
    
    df = pl.read_parquet(output_path)
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_skus": df.select("SKU").n_unique() if "SKU" in df.columns else 0,
        "date_range": None,
        "data_quality_score": 0.0
    }
    
    # Calculate date range if Date column exists
    if "Date" in df.columns:
        try:
            date_col = pl.col("Date").str.to_date()
            min_date = df.select(date_col.min()).item()
            max_date = df.select(date_col.max()).item()
            stats["date_range"] = (min_date, max_date)
        except:
            pass
    
    # Calculate data quality score
    total_cells = stats["total_rows"] * stats["total_columns"]
    null_cells = df.null_count().sum_horizontal().item()
    stats["data_quality_score"] = 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0
    
    return stats


def validate_reconstructed_data(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate the reconstructed data against our schema."""
    
    validation_results = {
        "valid_rows": 0,
        "invalid_rows": 0,
        "errors": []
    }
    
    # Sample validation - check a few rows
    sample_size = min(1000, len(df))
    sample_df = df.head(sample_size)
    
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        try:
            # Convert to dict and validate with Pydantic
            row_dict = {k: v for k, v in row.items() if v is not None}
            
            # Map column names to schema field names
            if "Date" in row_dict:
                row_dict["transaction_date"] = row_dict.pop("Date")
            if "Name" in row_dict:
                row_dict["customer_name"] = row_dict.pop("Name")
            
            SalesRow(**row_dict)
            validation_results["valid_rows"] += 1
            
        except Exception as e:
            validation_results["invalid_rows"] += 1
            validation_results["errors"].append(f"Row {i}: {str(e)}")
    
    return validation_results
