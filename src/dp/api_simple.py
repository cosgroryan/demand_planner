"""
Simple FastAPI backend for the demand planning system.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .models.forecast_api import DemandForecaster, forecast, batch_forecast
from .aggregate import load_aggregated_series, get_series_summary
from .features import load_features
from .backtest import rolling_backtest, save_backtest_results
from .planner_views import make_planning_table, export_planning_table_to_csv, export_planning_table_to_excel, get_planning_table_summary
from .inventory import calculate_inventory_kpis, batch_inventory_analysis, InventoryParameters

logger = logging.getLogger(__name__)

# Global forecaster instance
forecaster = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global forecaster
    # Startup
    forecaster = DemandForecaster()
    logger.info("Demand forecaster initialized")
    yield
    # Shutdown
    logger.info("Application shutdown")


app = FastAPI(
    title="Demand Planning API",
    description="Modern demand planning system for retail sales forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ForecastRequest(BaseModel):
    sku: str = Field(..., description="SKU identifier")
    market_channel: Optional[str] = Field(None, description="Market channel identifier")
    freq: str = Field("W", description="Frequency (W for weekly, M for monthly)")
    horizon: int = Field(13, description="Forecast horizon in periods")
    model: str = Field("auto", description="Model to use for forecasting")

class ForecastResponse(BaseModel):
    series_id: str
    model_used: str
    forecast: List[float]
    prediction_intervals: List[List[float]]
    horizon: int
    confidence_level: float
    forecast_date: str
    pattern_info: Dict[str, Any]

class SeriesResponse(BaseModel):
    sku: str
    market_channel: Optional[str]
    frequency: str
    periods: List[str]
    units: List[float]
    net_sales: List[float]
    avg_selling_price: List[float]
    transactions: List[int]

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": 0.0
    }

# Catalog endpoints
@app.get("/catalog/sku")
async def get_sku_catalog(
    search: Optional[str] = Query(None, description="Search term for SKU filtering"),
    page: int = Query(1, description="Page number"),
    page_size: int = Query(1000, description="Number of items per page")
):
    """Get available SKUs."""
    try:
        # Load forecast SKU data (without sizes)
        series_data = load_aggregated_series(["SKU"], "W", use_forecast_sku=True)
        
        # Get unique SKUs
        skus = series_data["SKU"].unique().to_list()
        
        # Apply search filter if provided
        if search:
            skus = [sku for sku in skus if search.lower() in sku.lower()]
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_skus = skus[start_idx:end_idx]
        
        return {
            "skus": paginated_skus,
            "total": len(skus),
            "page": page,
            "page_size": page_size,
            "total_pages": (len(skus) + page_size - 1) // page_size
        }
        
    except Exception as e:
        logger.error(f"Error getting SKU catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/catalog/market-channels")
async def get_market_channels():
    """Get available market channels."""
    try:
        # Load market channel data
        series_data = load_aggregated_series(["SKU", "Market-Channel"], "W")
        
        # Get unique market channels
        market_channels = series_data["Market-Channel"].unique().to_list()
        
        return {
            "market_channels": market_channels,
            "total": len(market_channels)
        }
        
    except Exception as e:
        logger.error(f"Error getting market channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecast endpoint
@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate demand forecast for a SKU."""
    
    try:
        # Load series data (use forecast SKU data for better forecasting)
        if request.market_channel:
            # Load SKU-Market-Channel series
            series_data = load_aggregated_series(["SKU", "Market-Channel"], request.freq)
            # Filter for specific SKU and market channel
            filtered_data = series_data.filter(
                (pl.col("SKU") == request.sku) & 
                (pl.col("Market-Channel") == request.market_channel)
            )
        else:
            # Load forecast SKU series (without sizes) for better data density
            try:
                series_data = load_aggregated_series(["SKU"], request.freq, use_forecast_sku=True)
            except:
                # Fallback to regular SKU data if forecast SKU data not available
                series_data = load_aggregated_series(["SKU"], request.freq)
            # Filter for specific SKU
            filtered_data = series_data.filter(pl.col("SKU") == request.sku)
        
        if len(filtered_data) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for SKU {request.sku}" + 
                       (f" and market channel {request.market_channel}" if request.market_channel else "")
            )
        
        # Sort by period
        filtered_data = filtered_data.sort("period")
        
        # Generate forecast
        forecast_result = forecaster.forecast(
            series_id=request.sku,
            series_data=filtered_data,
            horizon=request.horizon,
            model=request.model
        )
        
        return forecast_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for {request.sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Series endpoint
@app.get("/series/{sku}", response_model=SeriesResponse)
async def get_series_data(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market channel identifier"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    limit: int = Query(52, description="Number of periods to return")
):
    """Get historical series data for a SKU."""
    
    try:
        # Load series data
        if market_channel:
            series_data = load_aggregated_series(["SKU", "Market-Channel"], freq)
            # Filter for specific SKU and market channel
            filtered_data = series_data.filter(
                (pl.col("SKU") == sku) & 
                (pl.col("Market-Channel") == market_channel)
            )
        else:
            # Use forecast SKU data for consistency
            try:
                series_data = load_aggregated_series(["SKU"], freq, use_forecast_sku=True)
            except:
                # Fallback to regular SKU data if forecast SKU data not available
                series_data = load_aggregated_series(["SKU"], freq)
            # Filter for specific SKU
            filtered_data = series_data.filter(pl.col("SKU") == sku)
        
        if len(filtered_data) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for SKU {sku}" + 
                       (f" and market channel {market_channel}" if market_channel else "")
            )
        
        # Sort by period and get recent data
        filtered_data = filtered_data.sort("period").tail(limit)
        
        return SeriesResponse(
            sku=sku,
            market_channel=market_channel,
            frequency=freq,
            periods=[str(p) for p in filtered_data["period"].to_list()],
            units=filtered_data["units"].to_list(),
            net_sales=filtered_data["net_sales"].to_list(),
            avg_selling_price=filtered_data["avg_selling_price"].to_list(),
            transactions=filtered_data["transactions"].to_list()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting series data for {sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Parent forecast endpoints
@app.post("/forecast/parent", response_model=ForecastResponse)
async def generate_parent_forecast(request: ForecastRequest):
    """Generate demand forecast for a parent SKU (aggregates all child SKUs)."""
    
    try:
        # Load forecast SKU series data
        series_data = load_aggregated_series(["SKU"], request.freq, use_forecast_sku=True)
        
        # Find all SKUs that start with the parent SKU
        parent_sku = request.sku
        matching_skus = series_data.filter(
            pl.col("SKU").str.starts_with(parent_sku)
        )
        
        if matching_skus.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No SKUs found for parent: {parent_sku}"
            )
        
        # Aggregate data across all matching SKUs
        aggregated_data = matching_skus.group_by("period").agg([
            pl.col("units").sum().alias("units"),
            pl.col("net_sales").sum().alias("net_sales"),
            pl.col("transactions").sum().alias("transactions")
        ]).sort("period")
        
        # Generate forecast
        forecast_result = forecaster.forecast(
            series_id=f"{parent_sku}_parent",
            series_data=aggregated_data,
            horizon=request.horizon,
            model=request.model
        )
        
        return forecast_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating parent forecast for {request.sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/parent-graphic", response_model=ForecastResponse)
async def generate_parent_graphic_forecast(request: ForecastRequest):
    """Generate demand forecast for a parent-graphic SKU (aggregates all colors for a style)."""
    
    try:
        # Load forecast SKU series data
        series_data = load_aggregated_series(["SKU"], request.freq, use_forecast_sku=True)
        
        # Find all SKUs that start with the parent-graphic SKU (first 10 digits)
        parent_graphic_sku = request.sku
        matching_skus = series_data.filter(
            pl.col("SKU").str.starts_with(parent_graphic_sku)
        )
        
        if matching_skus.is_empty():
            raise HTTPException(
                status_code=404, 
                detail=f"No SKUs found for parent-graphic: {parent_graphic_sku}"
            )
        
        # Aggregate data across all matching SKUs
        aggregated_data = matching_skus.group_by("period").agg([
            pl.col("units").sum().alias("units"),
            pl.col("net_sales").sum().alias("net_sales"),
            pl.col("transactions").sum().alias("transactions")
        ]).sort("period")
        
        # Generate forecast
        forecast_result = forecaster.forecast(
            series_id=f"{parent_graphic_sku}_parent_graphic",
            series_data=aggregated_data,
            horizon=request.horizon,
            model=request.model
        )
        
        return forecast_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating parent-graphic forecast for {request.sku}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
