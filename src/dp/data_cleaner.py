"""
Data cleaning utilities for the demand planning system.
"""

import polars as pl
from pathlib import Path
from typing import Optional
import logging
from .sku_utils import create_forecast_sku

logger = logging.getLogger(__name__)


def clean_sales_data(
    input_path: str,
    output_path: str,
    sample_size: Optional[int] = None
) -> pl.DataFrame:
    """
    Clean and convert data types in the sales data.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path for cleaned output file
        sample_size: If provided, only process this many rows (for testing)
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Loading data from {input_path}")
    
    # Load data
    df = pl.read_parquet(input_path)
    
    if sample_size:
        df = df.head(sample_size)
        logger.info(f"Using sample of {sample_size} rows")
    
    logger.info(f"Original data: {len(df)} rows, {len(df.columns)} columns")
    
    # Clean and convert data types
    df_clean = df.with_columns([
        # Convert Date to proper date type
        pl.col("Date").str.to_date("%Y-%m-%d").alias("Date"),
        
        # Convert Quantity to numeric (handle empty strings and weird formats)
        pl.col("Quantity")
        .str.replace_all(r"^0-0\d+$", "0")  # Replace "0-010" type patterns with "0"
        .str.replace_all("^$", "0")  # Replace empty strings with "0"
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Quantity"),
        
        # Convert Amount (Net) to numeric (handle empty strings and currency symbols)
        pl.col("Amount (Net)")
        .str.replace_all(r"[$,]", "")
        .str.replace_all("^$", "0")  # Replace empty strings with "0"
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Amount (Net)"),
        
        # Convert Amount Discount to numeric
        pl.col("Amount Discount")
        .str.replace_all(r"[$,]", "")
        .str.replace_all("^$", "0")  # Replace empty strings with "0"
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Amount Discount"),
    ])
    
    # Filter out rows with invalid data
    df_clean = df_clean.filter(
        (pl.col("Date").is_not_null()) &
        (pl.col("SKU").is_not_null()) &
        (pl.col("SKU") != "") &
        (pl.col("Quantity") > 0)  # Only positive quantities
    )
    
    # Add derived columns
    df_clean = df_clean.with_columns([
        # Calculate net sales (Amount - Discount)
        (pl.col("Amount (Net)") - pl.col("Amount Discount")).alias("net_sales"),
        
        # Calculate average selling price
        (pl.col("Amount (Net)") / pl.col("Quantity")).alias("avg_selling_price"),
        
        # Calculate discount rate
        (pl.col("Amount Discount") / pl.col("Amount (Net)")).fill_null(0.0).alias("discount_rate"),
        
        # Add year and month for aggregation
        pl.col("Date").dt.year().alias("year"),
        pl.col("Date").dt.month().alias("month"),
        pl.col("Date").dt.week().alias("week"),
    ])
    
    # Rename Quantity to units for consistency
    df_clean = df_clean.rename({"Quantity": "units"})
    
    # Create forecast SKU (without size) for better forecasting granularity
    df_clean = create_forecast_sku(df_clean)
    
    logger.info(f"Cleaned data: {len(df_clean)} rows")
    logger.info(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    logger.info(f"Total unique SKUs: {df_clean['SKU'].n_unique()}")
    logger.info(f"Total sales: ${df_clean['Amount (Net)'].sum():,.2f}")
    
    # Save cleaned data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.write_parquet(output_path, compression='snappy')
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df_clean
