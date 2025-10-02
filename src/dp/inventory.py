"""
Inventory signals and KPIs for demand planning.

This module converts forecasts into inventory decisions including reorder points,
order up-to levels, and suggested purchase orders.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass

from .models.forecast_api import DemandForecaster, forecast
from .aggregate import load_aggregated_series

logger = logging.getLogger(__name__)


@dataclass
class InventoryParameters:
    """Inventory planning parameters."""
    current_on_hand: float
    on_order: float
    lead_time_periods: int
    target_service_level: float
    safety_stock_multiplier: float = 1.0
    review_period: int = 1
    minimum_order_quantity: float = 0.0
    order_multiple: float = 1.0


@dataclass
class InventorySignals:
    """Inventory planning signals and recommendations."""
    reorder_point: float
    order_up_to_level: float
    suggested_order_quantity: float
    stockout_risk_percent: float
    weeks_of_cover: float
    turnover_rate: float
    service_level_achieved: float
    safety_stock: float
    order_schedule: List[Dict[str, Any]]


class InventoryPlanner:
    """Inventory planning system that converts forecasts into inventory decisions."""
    
    def __init__(self):
        self.forecaster = DemandForecaster()
    
    def calculate_inventory_signals(
        self,
        sku: str,
        market_channel: Optional[str] = None,
        freq: str = "W",
        horizon: int = 13,
        inventory_params: InventoryParameters = None,
        model: str = "auto"
    ) -> InventorySignals:
        """
        Calculate inventory signals and recommendations for a SKU.
        
        Args:
            sku: SKU to calculate inventory signals for
            market_channel: Optional market channel filter
            freq: Frequency ("W" for weekly, "M" for monthly)
            horizon: Forecast horizon in periods
            inventory_params: Inventory planning parameters
            model: Model type to use for forecasting
        
        Returns:
            InventorySignals object with recommendations
        """
        logger.info(f"Calculating inventory signals for SKU: {sku}")
        
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
        
        # Sort by period
        filtered_data = filtered_data.sort("period")
        
        # Generate forecast
        series_id = f"{sku}_{market_channel}" if market_channel else sku
        forecast_result = self.forecaster.forecast(
            series_id=series_id,
            series_data=filtered_data,
            horizon=horizon,
            model=model
        )
        
        # Calculate demand statistics
        demand_stats = self._calculate_demand_statistics(filtered_data, forecast_result)
        
        # Calculate inventory signals
        signals = self._calculate_inventory_signals(
            demand_stats=demand_stats,
            inventory_params=inventory_params,
            forecast_result=forecast_result
        )
        
        return signals
    
    def _calculate_demand_statistics(
        self,
        historical_data: pl.DataFrame,
        forecast_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate demand statistics from historical data and forecast."""
        
        # Historical demand
        hist_units = historical_data.select("units").to_series().to_list()
        hist_units = [u for u in hist_units if u >= 0]  # Remove negative values
        
        # Forecast demand
        forecast_units = forecast_result["forecast"]
        
        # Calculate statistics
        mean_demand = np.mean(hist_units) if hist_units else 0
        std_demand = np.std(hist_units) if len(hist_units) > 1 else 0
        cv_demand = std_demand / mean_demand if mean_demand > 0 else 0
        
        # Lead time demand (for reorder point calculation)
        lead_time_demand = np.mean(forecast_units[:4])  # Assume 4-period lead time
        
        return {
            "mean_demand": mean_demand,
            "std_demand": std_demand,
            "cv_demand": cv_demand,
            "lead_time_demand": lead_time_demand,
            "forecast_mean": np.mean(forecast_units),
            "forecast_std": np.std(forecast_units),
            "total_historical_units": sum(hist_units),
            "periods_with_demand": sum(1 for u in hist_units if u > 0),
            "total_periods": len(hist_units)
        }
    
    def _calculate_inventory_signals(
        self,
        demand_stats: Dict[str, float],
        inventory_params: InventoryParameters,
        forecast_result: Dict[str, Any]
    ) -> InventorySignals:
        """Calculate inventory signals based on demand statistics and parameters."""
        
        if inventory_params is None:
            inventory_params = InventoryParameters(
                current_on_hand=0,
                on_order=0,
                lead_time_periods=4,
                target_service_level=0.95
            )
        
        # Extract demand statistics
        mean_demand = demand_stats["mean_demand"]
        std_demand = demand_stats["std_demand"]
        lead_time_demand = demand_stats["lead_time_demand"]
        
        # Calculate safety stock using service level
        z_score = self._get_z_score(inventory_params.target_service_level)
        safety_stock = z_score * std_demand * np.sqrt(inventory_params.lead_time_periods)
        safety_stock *= inventory_params.safety_stock_multiplier
        
        # Calculate reorder point
        reorder_point = (lead_time_demand * inventory_params.lead_time_periods) + safety_stock
        
        # Calculate order up-to level (EOQ-based)
        order_up_to_level = reorder_point + (mean_demand * inventory_params.review_period)
        
        # Calculate current inventory position
        current_inventory_position = inventory_params.current_on_hand + inventory_params.on_order
        
        # Calculate suggested order quantity
        if current_inventory_position <= reorder_point:
            suggested_order_quantity = max(
                order_up_to_level - current_inventory_position,
                inventory_params.minimum_order_quantity
            )
            
            # Apply order multiple
            if inventory_params.order_multiple > 1:
                suggested_order_quantity = np.ceil(
                    suggested_order_quantity / inventory_params.order_multiple
                ) * inventory_params.order_multiple
        else:
            suggested_order_quantity = 0
        
        # Calculate stockout risk
        stockout_risk = self._calculate_stockout_risk(
            current_inventory_position,
            lead_time_demand,
            std_demand,
            inventory_params.lead_time_periods
        )
        
        # Calculate weeks of cover
        weeks_of_cover = current_inventory_position / mean_demand if mean_demand > 0 else float('inf')
        
        # Calculate turnover rate (annual)
        annual_demand = mean_demand * 52  # Assuming weekly data
        avg_inventory = (inventory_params.current_on_hand + inventory_params.on_order) / 2
        turnover_rate = annual_demand / avg_inventory if avg_inventory > 0 else 0
        
        # Calculate achieved service level
        service_level_achieved = 1 - stockout_risk
        
        # Generate order schedule
        order_schedule = self._generate_order_schedule(
            forecast_result=forecast_result,
            inventory_params=inventory_params,
            reorder_point=reorder_point,
            order_up_to_level=order_up_to_level
        )
        
        return InventorySignals(
            reorder_point=reorder_point,
            order_up_to_level=order_up_to_level,
            suggested_order_quantity=suggested_order_quantity,
            stockout_risk_percent=stockout_risk * 100,
            weeks_of_cover=weeks_of_cover,
            turnover_rate=turnover_rate,
            service_level_achieved=service_level_achieved,
            safety_stock=safety_stock,
            order_schedule=order_schedule
        )
    
    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for given service level."""
        # Common z-scores for service levels
        z_scores = {
            0.80: 0.84,
            0.85: 1.04,
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33,
            0.999: 3.09
        }
        
        # Find closest service level
        closest_level = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores[closest_level]
    
    def _calculate_stockout_risk(
        self,
        current_inventory: float,
        lead_time_demand: float,
        demand_std: float,
        lead_time_periods: int
    ) -> float:
        """Calculate stockout risk during lead time."""
        
        # Expected demand during lead time
        expected_lead_time_demand = lead_time_demand * lead_time_periods
        
        # Standard deviation of lead time demand
        lead_time_demand_std = demand_std * np.sqrt(lead_time_periods)
        
        # Calculate stockout probability using normal distribution
        if lead_time_demand_std > 0:
            z_score = (current_inventory - expected_lead_time_demand) / lead_time_demand_std
            # Stockout risk is the probability that demand exceeds current inventory
            stockout_risk = 1 - self._normal_cdf(z_score)
        else:
            stockout_risk = 1.0 if current_inventory < expected_lead_time_demand else 0.0
        
        return max(0, min(1, stockout_risk))
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _generate_order_schedule(
        self,
        forecast_result: Dict[str, Any],
        inventory_params: InventoryParameters,
        reorder_point: float,
        order_up_to_level: float
    ) -> List[Dict[str, Any]]:
        """Generate order schedule for the forecast horizon."""
        
        forecast_values = forecast_result["forecast"]
        order_schedule = []
        
        current_inventory = inventory_params.current_on_hand
        current_on_order = inventory_params.on_order
        
        for period, forecast_demand in enumerate(forecast_values):
            # Update inventory position
            current_inventory_position = current_inventory + current_on_order
            
            # Check if we need to place an order
            if current_inventory_position <= reorder_point:
                order_quantity = max(
                    order_up_to_level - current_inventory_position,
                    inventory_params.minimum_order_quantity
                )
                
                # Apply order multiple
                if inventory_params.order_multiple > 1:
                    order_quantity = np.ceil(
                        order_quantity / inventory_params.order_multiple
                    ) * inventory_params.order_multiple
                
                order_schedule.append({
                    "period": period + 1,
                    "forecast_demand": forecast_demand,
                    "inventory_before": current_inventory,
                    "on_order_before": current_on_order,
                    "order_quantity": order_quantity,
                    "inventory_after": current_inventory + order_quantity,
                    "on_order_after": current_on_order + order_quantity
                })
                
                # Update inventory for next period
                current_inventory += order_quantity
                current_on_order = 0
            else:
                order_schedule.append({
                    "period": period + 1,
                    "forecast_demand": forecast_demand,
                    "inventory_before": current_inventory,
                    "on_order_before": current_on_order,
                    "order_quantity": 0,
                    "inventory_after": current_inventory,
                    "on_order_after": current_on_order
                })
            
            # Reduce inventory by forecast demand
            current_inventory = max(0, current_inventory - forecast_demand)
        
        return order_schedule


def calculate_inventory_kpis(
    sku: str,
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13,
    current_on_hand: float = 0,
    on_order: float = 0,
    lead_time_periods: int = 4,
    target_service_level: float = 0.95,
    model: str = "auto"
) -> Dict[str, Any]:
    """
    Convenience function to calculate inventory KPIs for a SKU.
    
    Args:
        sku: SKU to calculate KPIs for
        market_channel: Optional market channel filter
        freq: Frequency ("W" for weekly, "M" for monthly)
        horizon: Forecast horizon in periods
        current_on_hand: Current on-hand inventory
        on_order: Current on-order quantity
        lead_time_periods: Lead time in periods
        target_service_level: Target service level (0-1)
        model: Model type to use for forecasting
    
    Returns:
        Dictionary with inventory KPIs and recommendations
    """
    planner = InventoryPlanner()
    
    inventory_params = InventoryParameters(
        current_on_hand=current_on_hand,
        on_order=on_order,
        lead_time_periods=lead_time_periods,
        target_service_level=target_service_level
    )
    
    signals = planner.calculate_inventory_signals(
        sku=sku,
        market_channel=market_channel,
        freq=freq,
        horizon=horizon,
        inventory_params=inventory_params,
        model=model
    )
    
    return {
        "sku": sku,
        "market_channel": market_channel,
        "inventory_signals": {
            "reorder_point": signals.reorder_point,
            "order_up_to_level": signals.order_up_to_level,
            "suggested_order_quantity": signals.suggested_order_quantity,
            "stockout_risk_percent": signals.stockout_risk_percent,
            "weeks_of_cover": signals.weeks_of_cover,
            "turnover_rate": signals.turnover_rate,
            "service_level_achieved": signals.service_level_achieved,
            "safety_stock": signals.safety_stock
        },
        "order_schedule": signals.order_schedule,
        "parameters": {
            "current_on_hand": inventory_params.current_on_hand,
            "on_order": inventory_params.on_order,
            "lead_time_periods": inventory_params.lead_time_periods,
            "target_service_level": inventory_params.target_service_level
        }
    }


def batch_inventory_analysis(
    sku_list: List[str],
    market_channel: Optional[str] = None,
    freq: str = "W",
    horizon: int = 13,
    default_inventory_params: InventoryParameters = None,
    model: str = "auto"
) -> Dict[str, Any]:
    """
    Perform batch inventory analysis for multiple SKUs.
    
    Args:
        sku_list: List of SKUs to analyze
        market_channel: Optional market channel filter
        freq: Frequency ("W" for weekly, "M" for monthly)
        horizon: Forecast horizon in periods
        default_inventory_params: Default inventory parameters
        model: Model type to use for forecasting
    
    Returns:
        Dictionary with inventory analysis results for all SKUs
    """
    planner = InventoryPlanner()
    
    if default_inventory_params is None:
        default_inventory_params = InventoryParameters(
            current_on_hand=0,
            on_order=0,
            lead_time_periods=4,
            target_service_level=0.95
        )
    
    results = {}
    
    for sku in sku_list:
        try:
            signals = planner.calculate_inventory_signals(
                sku=sku,
                market_channel=market_channel,
                freq=freq,
                horizon=horizon,
                inventory_params=default_inventory_params,
                model=model
            )
            
            results[sku] = {
                "success": True,
                "signals": {
                    "reorder_point": signals.reorder_point,
                    "order_up_to_level": signals.order_up_to_level,
                    "suggested_order_quantity": signals.suggested_order_quantity,
                    "stockout_risk_percent": signals.stockout_risk_percent,
                    "weeks_of_cover": signals.weeks_of_cover,
                    "turnover_rate": signals.turnover_rate,
                    "service_level_achieved": signals.service_level_achieved,
                    "safety_stock": signals.safety_stock
                },
                "order_schedule": signals.order_schedule
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing inventory for SKU {sku}: {e}")
            results[sku] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "total_skus": len(sku_list),
        "successful_analyses": sum(1 for r in results.values() if r["success"]),
        "failed_analyses": sum(1 for r in results.values() if not r["success"]),
        "results": results
    }
