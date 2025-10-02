"""
FastAPI backend for the demand planning system.

This module provides REST API endpoints for forecasting, data access,
and system health monitoring.
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
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ForecastRequest(BaseModel):
    sku: str = Field(..., description="SKU to forecast")
    market_channel: Optional[str] = Field(None, description="Market-Channel filter")
    freq: str = Field("W", description="Frequency (W for weekly, M for monthly)")
    horizon: int = Field(13, description="Forecast horizon in periods")
    model: str = Field("auto", description="Model type to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sku": "SKU001",
                "market_channel": "US-Retail",
                "freq": "W",
                "horizon": 13,
                "model": "auto"
            }
        }


class ForecastResponse(BaseModel):
    series_id: str
    model_used: str
    forecast: List[float]
    prediction_intervals: List[List[float]]
    horizon: int
    confidence_level: float
    forecast_date: str
    pattern_info: Optional[Dict[str, Any]] = None


class SeriesResponse(BaseModel):
    sku: str
    market_channel: Optional[str] = None
    periods: List[str]
    units: List[float]
    net_sales: List[float]
    avg_selling_price: List[float]
    discount_rate: List[float]


class SKUListResponse(BaseModel):
    skus: List[str]
    total_count: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float


# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=0.0  # TODO: Implement actual uptime tracking
    )


@app.get("/catalog/sku", response_model=SKUListResponse)
async def get_sku_catalog(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Page size"),
    search: Optional[str] = Query(None, description="Search term for SKU filtering")
):
    """Get paginated list of available SKUs."""
    
    try:
        # Load forecast SKU data (without sizes) for better forecasting
        try:
            series_data = load_aggregated_series(["SKU"], "W", use_forecast_sku=True)
        except:
            # Fallback to regular SKU data if forecast SKU data not available
            series_data = load_aggregated_series(["SKU"], "W")
        
        # Get unique SKUs
        skus = series_data.select("SKU").unique().to_series().to_list()
        
        # Apply search filter if provided
        if search:
            skus = [sku for sku in skus if search.lower() in sku.lower()]
        
        # Paginate
        total_count = len(skus)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_skus = skus[start_idx:end_idx]
        
        return SKUListResponse(
            skus=paginated_skus,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error getting SKU catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/catalog/market-channels")
async def get_market_channels():
    """Get list of available market channels."""
    
    try:
        # Load series data to get available market channels
        series_data = load_aggregated_series(["SKU", "Market-Channel"], "W")
        
        # Get unique market channels
        market_channels = series_data.select("Market-Channel").unique().to_series().to_list()
        
        return {
            "market_channels": market_channels,
            "total_count": len(market_channels)
        }
        
    except Exception as e:
        logger.error(f"Error getting market channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        series_id = f"{request.sku}_{request.market_channel}" if request.market_channel else request.sku
        result = forecaster.forecast(
            series_id=series_id,
            series_data=filtered_data,
            horizon=request.horizon,
            model=request.model
        )
        
        return ForecastResponse(
            series_id=result["series_id"],
            model_used=result["model_used"],
            forecast=result["forecast"],
            prediction_intervals=result.get("prediction_intervals", []),
            horizon=result["horizon"],
            confidence_level=result["confidence_level"],
            forecast_date=result["forecast_date"],
            pattern_info=result.get("pattern_info")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/series/{sku}", response_model=SeriesResponse)
async def get_series_data(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    limit: int = Query(52, ge=1, le=104, description="Number of recent periods to return")
):
    """Get recent historical data for a SKU."""
    
    try:
        # Load series data (use forecast SKU data for consistency)
        if market_channel:
            # Load SKU-Market-Channel series
            series_data = load_aggregated_series(["SKU", "Market-Channel"], freq)
            # Filter for specific SKU and market channel
            filtered_data = series_data.filter(
                (pl.col("SKU") == sku) & 
                (pl.col("Market-Channel") == market_channel)
            )
        else:
            # Load forecast SKU series (without sizes) for consistency with forecast endpoint
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
            periods=[str(period) for period in filtered_data.select("period").to_series().to_list()],
            units=filtered_data.select("units").to_series().to_list(),
            net_sales=filtered_data.select("net_sales").to_series().to_list(),
            avg_selling_price=filtered_data.select("avg_selling_price").to_series().to_list(),
            discount_rate=filtered_data.select("discount_rate").to_series().to_list()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting series data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/batch")
async def generate_batch_forecast(
    skus: List[str],
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13,
    model: str = "auto"
):
    """Generate forecasts for multiple SKUs."""
    
    try:
        # Load series data
        if market_channel:
            series_data = load_aggregated_series(["SKU", "Market-Channel"], freq)
        else:
            series_data = load_aggregated_series(["SKU"], freq)
        
        # Prepare series dictionary
        series_dict = {}
        for sku in skus:
            if market_channel:
                filtered_data = series_data.filter(
                    (pl.col("SKU") == sku) & 
                    (pl.col("Market-Channel") == market_channel)
                )
                series_id = f"{sku}_{market_channel}"
            else:
                filtered_data = series_data.filter(pl.col("SKU") == sku)
                series_id = sku
            
            if len(filtered_data) > 0:
                series_dict[series_id] = filtered_data.sort("period")
        
        if not series_dict:
            raise HTTPException(status_code=404, detail="No data found for any of the specified SKUs")
        
        # Generate batch forecasts
        results = batch_forecast(
            series_dict=series_dict,
            horizon=horizon,
            model=model
        )
        
        return {
            "total_skus": len(skus),
            "successful_forecasts": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating batch forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/backtest")
async def run_backtest(
    grain: List[str] = ["SKU"],
    freq: str = "W",
    model: str = "auto",
    horizon: int = 13,
    history_window: int = 52,
    step_size: int = 4,
    background_tasks: BackgroundTasks = None
):
    """Run rolling backtest for model evaluation."""
    
    try:
        # Load series data
        series_data = load_aggregated_series(grain, freq)
        
        # Run backtest
        results = rolling_backtest(
            series_data=series_data,
            model=model,
            history_window=history_window,
            forecast_horizon=horizon,
            step_size=step_size
        )
        
        # Save results in background
        if background_tasks:
            background_tasks.add_task(save_backtest_results, results)
        
        return {
            "status": "completed",
            "model": results["model"],
            "total_series": results.get("total_series", 1),
            "aggregated_metrics": results.get("aggregated_metrics", results.get("metrics", {})),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/series/summary")
async def get_series_summary_endpoint():
    """Get summary of all available series."""
    
    try:
        summary_df = get_series_summary()
        
        if len(summary_df) == 0:
            return {"series": [], "total_count": 0}
        
        # Convert to list of dictionaries
        series_list = summary_df.to_dicts()
        
        return {
            "series": series_list,
            "total_count": len(series_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting series summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available")
async def get_available_models():
    """Get list of available forecasting models."""
    
    return {
        "baseline_models": [
            "naive",
            "seasonal_naive", 
            "moving_average",
            "croston",
            "exponential_smoothing"
        ],
        "statistical_models": [
            "arima",
            "sarimax",
            "auto_arima"
        ],
        "ml_models": [
            "random_forest",
            "gradient_boosting"
        ],
        "auto_model": "auto"
    }


@app.get("/models/evaluate/{sku}")
async def evaluate_models_for_sku(
    sku: str,
    market_channel: Optional[str] = None,
    freq: str = "W",
    test_size: int = 13
):
    """Evaluate multiple models for a specific SKU."""
    
    try:
        # Load series data
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
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for SKU {sku}" + 
                       (f" and market channel {market_channel}" if market_channel else "")
            )
        
        # Sort by period
        filtered_data = filtered_data.sort("period")
        
        # Evaluate models
        results = forecaster.evaluate_models(
            series_data=filtered_data,
            test_size=test_size
        )
        
        # Get best model
        best_model = forecaster.get_best_model(results)
        
        return {
            "sku": sku,
            "market_channel": market_channel,
            "best_model": best_model,
            "evaluation_results": results,
            "test_size": test_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/planning/{sku}")
async def get_planning_table(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    horizon: int = Query(13, description="Forecast horizon in periods"),
    history_periods: int = Query(52, description="Number of historical periods to include"),
    model: str = Query("auto", description="Model type to use"),
    include_confidence_bands: bool = Query(True, description="Include confidence intervals"),
    include_revenue_projection: bool = Query(True, description="Include revenue projections")
):
    """Get planner-friendly planning table with past actuals and future forecasts."""
    
    try:
        planning_data = make_planning_table(
            sku=sku,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon,
            history_periods=history_periods,
            model=model,
            include_confidence_bands=include_confidence_bands,
            include_revenue_projection=include_revenue_projection
        )
        
        # Convert planning table to dict for JSON response
        planning_table_dict = planning_data["planning_table"].to_dicts()
        
    return {
            "sku": planning_data["sku"],
            "market_channel": planning_data["market_channel"],
            "frequency": planning_data["frequency"],
            "planning_table": planning_table_dict,
            "summary_stats": planning_data["summary_stats"],
            "forecast_metadata": planning_data["forecast_metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error creating planning table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/planning/{sku}.csv")
async def get_planning_table_csv(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    horizon: int = Query(13, description="Forecast horizon in periods"),
    history_periods: int = Query(52, description="Number of historical periods to include"),
    model: str = Query("auto", description="Model type to use")
):
    """Download planning table as CSV file."""
    
    try:
        planning_data = make_planning_table(
            sku=sku,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon,
            history_periods=history_periods,
            model=model
        )
        
        # Export to CSV
        csv_path = export_planning_table_to_csv(planning_data)
        
        # Return file
        return FileResponse(
            path=csv_path,
            media_type="text/csv",
            filename=f"planning_table_{sku}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except Exception as e:
        logger.error(f"Error creating planning table CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/planning/{sku}.xlsx")
async def get_planning_table_excel(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    horizon: int = Query(13, description="Forecast horizon in periods"),
    history_periods: int = Query(52, description="Number of historical periods to include"),
    model: str = Query("auto", description="Model type to use")
):
    """Download planning table as Excel file."""
    
    try:
        planning_data = make_planning_table(
            sku=sku,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon,
            history_periods=history_periods,
            model=model
        )
        
        # Export to Excel
        excel_path = export_planning_table_to_excel(planning_data)
        
        # Return file
        return FileResponse(
            path=excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"planning_table_{sku}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
    except Exception as e:
        logger.error(f"Error creating planning table Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/planning/{sku}/summary")
async def get_planning_summary(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    horizon: int = Query(13, description="Forecast horizon in periods")
):
    """Get a quick summary of planning table data."""
    
    try:
        summary = get_planning_table_summary(
            sku=sku,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting planning summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/{sku}")
async def get_inventory_signals(
    sku: str,
    market_channel: Optional[str] = Query(None, description="Market-Channel filter"),
    freq: str = Query("W", description="Frequency (W for weekly, M for monthly)"),
    horizon: int = Query(13, description="Forecast horizon in periods"),
    current_on_hand: float = Query(0, description="Current on-hand inventory"),
    on_order: float = Query(0, description="Current on-order quantity"),
    lead_time_periods: int = Query(4, description="Lead time in periods"),
    target_service_level: float = Query(0.95, description="Target service level (0-1)"),
    model: str = Query("auto", description="Model type to use")
):
    """Get inventory signals and KPIs for a SKU."""
    
    try:
        kpis = calculate_inventory_kpis(
            sku=sku,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon,
            current_on_hand=current_on_hand,
            on_order=on_order,
            lead_time_periods=lead_time_periods,
            target_service_level=target_service_level,
            model=model
        )
        
        return kpis
        
    except Exception as e:
        logger.error(f"Error calculating inventory signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inventory/batch")
async def get_batch_inventory_analysis(
    skus: List[str],
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13,
    current_on_hand: float = 0,
    on_order: float = 0,
    lead_time_periods: int = 4,
    target_service_level: float = 0.95,
    model: str = "auto"
):
    """Get inventory signals for multiple SKUs."""
    
    try:
        default_params = InventoryParameters(
            current_on_hand=current_on_hand,
            on_order=on_order,
            lead_time_periods=lead_time_periods,
            target_service_level=target_service_level
        )
        
        results = batch_inventory_analysis(
            sku_list=skus,
            market_channel=market_channel,
            freq=freq,
            horizon=horizon,
            default_inventory_params=default_params,
            model=model
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error performing batch inventory analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.dp.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )