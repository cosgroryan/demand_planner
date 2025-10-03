import polars as pl
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .aggregate import load_aggregated_series
from .models.forecast_api import DemandForecaster

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Demand Planning API - Parent SKU Focus",
    description="Simplified API for parent SKU demand planning with historical data and ML forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecaster
forecaster = DemandForecaster()

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

class ParentSKUResponse(BaseModel):
    parent_sku: str
    total_skus: int
    total_units: float
    total_sales: float
    avg_selling_price: float
    date_range: Dict[str, str]

class HistoricalDataResponse(BaseModel):
    parent_sku: str
    periods: List[str]
    units: List[float]
    net_sales: List[float]
    avg_selling_price: List[float]
    transactions: List[int]

class ForecastResponse(BaseModel):
    parent_sku: str
    model_used: str
    forecast_periods: List[str]
    forecast_units: List[float]
    forecast_sales: List[float]
    confidence_intervals: Dict[str, Any]
    historical_periods: List[str]
    historical_units: List[float]
    historical_sales: List[float]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/parent-skus", response_model=List[str])
async def get_parent_skus():
    """Get list of all parent SKUs."""
    try:
        series_data = load_aggregated_series(["SKU"], "W", use_forecast_sku=True)
        
        # Extract parent SKUs (first 6 digits)
        parent_skus = series_data["SKU"].str.slice(0, 6).unique().sort().to_list()
        
        return parent_skus
        
    except Exception as e:
        logger.error(f"Error getting parent SKUs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parent-sku/{parent_sku}/summary", response_model=ParentSKUResponse)
async def get_parent_sku_summary(
    parent_sku: str,
    include_prebook: bool = Query(True, description="Include pre-book sales")
):
    """Get summary information for a parent SKU with configurable pre-book filtering."""
    try:
        # Load raw sales data for consistent calculations
        raw_data = pl.read_parquet("data/processed/sales_clean.parquet")
        
        # Filter for parent SKU
        parent_data = raw_data.filter(pl.col("SKU_Parent") == parent_sku)
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for parent SKU: {parent_sku}"
            )
        
        # Apply pre-book filtering
        if not include_prebook:
            parent_data = parent_data.filter(pl.col("Order Type") != "Prebook")
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No sales data found for parent SKU: {parent_sku} with current filters"
            )
        
        # Calculate summary statistics from filtered data
        total_skus = parent_data["SKU"].n_unique()
        total_units = parent_data["units"].sum()
        total_sales = parent_data["Amount (Net)"].sum()
        avg_selling_price = total_sales / total_units if total_units > 0 else 0.0
        
        # Get date range
        min_date = parent_data["Date"].min()
        max_date = parent_data["Date"].max()
        
        return ParentSKUResponse(
            parent_sku=parent_sku,
            total_skus=total_skus,
            total_units=total_units,
            total_sales=total_sales,
            avg_selling_price=avg_selling_price,
            date_range={
                "start": str(min_date),
                "end": str(max_date)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parent SKU summary for {parent_sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parent-sku/{parent_sku}/historical", response_model=HistoricalDataResponse)
async def get_parent_sku_historical(
    parent_sku: str,
    limit: int = Query(52, ge=1, le=200, description="Number of recent periods to return"),
    include_prebook: bool = Query(True, description="Include pre-book sales")
):
    """Get historical data for a parent SKU with configurable pre-book filtering."""
    try:
        # Load raw sales data for consistent calculations
        raw_data = pl.read_parquet("data/processed/sales_clean.parquet")
        
        # Filter for parent SKU
        parent_data = raw_data.filter(pl.col("SKU_Parent") == parent_sku)
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for parent SKU: {parent_sku}"
            )
        
        # Apply pre-book filtering
        if not include_prebook:
            parent_data = parent_data.filter(pl.col("Order Type") != "Prebook")
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No sales data found for parent SKU: {parent_sku} with current filters"
            )
        
        # Create weekly aggregation from filtered data
        aggregated_data = parent_data.with_columns([
            pl.col("Date").dt.truncate("1w").alias("period")
        ]).group_by("period").agg([
            pl.col("units").sum().alias("units"),
            pl.col("Amount (Net)").sum().alias("net_sales"),
            pl.col("units").count().alias("transactions")
        ]).sort("period").tail(limit)
        
        # Calculate average selling price for each period
        aggregated_data = aggregated_data.with_columns([
            (pl.col("net_sales") / pl.col("units")).alias("avg_selling_price")
        ]).fill_null(0.0)
        
        return HistoricalDataResponse(
            parent_sku=parent_sku,
            periods=[str(p) for p in aggregated_data["period"].to_list()],
            units=aggregated_data["units"].to_list(),
            net_sales=aggregated_data["net_sales"].to_list(),
            avg_selling_price=aggregated_data["avg_selling_price"].to_list(),
            transactions=aggregated_data["transactions"].to_list()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data for parent SKU {parent_sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parent-sku/{parent_sku}/forecast", response_model=ForecastResponse)
async def generate_parent_sku_forecast(
    parent_sku: str,
    horizon: int = Query(13, ge=1, le=52, description="Forecast horizon in weeks"),
    model: str = Query("auto", description="Model type (auto, random_forest, gradient_boosting, arima, etc.)"),
    include_prebook: bool = Query(True, description="Include pre-book sales")
):
    """Generate ML forecast for a parent SKU with configurable pre-book filtering."""
    try:
        logger.info(f"Starting forecast for parent SKU: {parent_sku}, include_prebook: {include_prebook}")
        
        # Load raw sales data for consistent calculations
        raw_data = pl.read_parquet("data/processed/sales_clean.parquet")
        logger.info(f"Loaded raw data: {raw_data.shape}")
        
        # Filter for parent SKU
        parent_data = raw_data.filter(pl.col("SKU_Parent") == parent_sku)
        logger.info(f"Found {len(parent_data)} records for parent SKU")
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for parent SKU: {parent_sku}"
            )
        
        # Apply pre-book filtering
        if not include_prebook:
            parent_data = parent_data.filter(pl.col("Order Type") != "Prebook")
            logger.info(f"After pre-book filtering: {len(parent_data)} records")
        
        if parent_data.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No sales data found for parent SKU: {parent_sku} with current filters"
            )
        
        # Create weekly aggregation from filtered data
        aggregated_data = parent_data.with_columns([
            pl.col("Date").dt.truncate("1w").alias("period")
        ]).group_by("period").agg([
            pl.col("units").sum().alias("units"),
            pl.col("Amount (Net)").sum().alias("net_sales"),
            pl.col("units").count().alias("transactions")
        ]).sort("period")
        logger.info(f"Aggregated data shape: {aggregated_data.shape}")
        
        # Calculate average selling price
        total_units = aggregated_data["units"].sum()
        total_sales = aggregated_data["net_sales"].sum()
        avg_selling_price = total_sales / total_units if total_units > 0 else 25.0
        logger.info(f"Calculated avg_selling_price: {avg_selling_price}")
        
        # Generate forecast
        logger.info(f"Starting forecast generation with model: {model}")
        forecast_result = forecaster.forecast(
            series_id=f"{parent_sku}_parent",
            series_data=aggregated_data,
            horizon=horizon,
            model=model
        )
        logger.info("Forecast generation completed")
        
        # Prepare response
        forecast_periods = []
        forecast_units = forecast_result["forecast"]
        forecast_sales = [units * avg_selling_price for units in forecast_units]
        
        # Generate forecast periods (assuming weekly data)
        last_period = aggregated_data["period"].max()
        for i in range(1, horizon + 1):
            next_period = last_period + pd.Timedelta(weeks=i)
            forecast_periods.append(str(next_period))
        
        # Get historical data for context
        historical_data = aggregated_data.tail(52)  # Last 52 weeks
        
        # Handle confidence intervals
        confidence_intervals = forecast_result.get("prediction_intervals", {})
        if isinstance(confidence_intervals, list):
            confidence_intervals = {"units": confidence_intervals}
        
        return ForecastResponse(
            parent_sku=parent_sku,
            model_used=forecast_result["model_used"],
            forecast_periods=forecast_periods,
            forecast_units=forecast_units,
            forecast_sales=forecast_sales,
            confidence_intervals=confidence_intervals,
            historical_periods=[str(p) for p in historical_data["period"].to_list()],
            historical_units=historical_data["units"].to_list(),
            historical_sales=historical_data["net_sales"].to_list()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for parent SKU {parent_sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
