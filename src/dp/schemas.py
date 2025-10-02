"""
Pydantic schemas for data validation and type safety.

These schemas define the structure of sales data and ensure data quality
throughout the demand planning pipeline.
"""

from datetime import datetime, date
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .constants import (
    OrderType, CustomerCategory, Market, Currency, 
    MIN_QUANTITY, MAX_QUANTITY, MIN_AMOUNT, MAX_AMOUNT,
    MIN_DATE, MAX_DATE
)


class SalesRow(BaseModel):
    """Schema for a single sales transaction row."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Core identifiers
    internal_id: Optional[str] = Field(None, description="Internal transaction ID")
    sku: str = Field(..., description="Product SKU code", min_length=1)
    transaction_date: date = Field(..., description="Transaction date", alias="date")
    period: Optional[str] = Field(None, description="Period identifier (e.g., 'Oct 2023')")
    
    # Transaction details
    document_number: Optional[str] = Field(None, description="Document/order number")
    customer_name: Optional[str] = Field(None, description="Customer or transaction name", alias="name")
    order_type: Optional[OrderType] = Field(None, description="Type of order")
    customer_category: Optional[CustomerCategory] = Field(None, description="Customer category")
    customer_subcategory: Optional[str] = Field(None, description="Customer subcategory")
    market: Optional[Market] = Field(None, description="Market/region")
    
    # Financial details
    po_cheque_number: Optional[str] = Field(None, description="PO or cheque number")
    memo_main: Optional[str] = Field(None, description="Main memo field")
    description: Optional[str] = Field(None, description="Product description")
    quantity: float = Field(..., description="Quantity sold", ge=MIN_QUANTITY, le=MAX_QUANTITY)
    account: Optional[str] = Field(None, description="Account code")
    amount_net: Optional[float] = Field(None, description="Net amount", ge=MIN_AMOUNT, le=MAX_AMOUNT)
    currency: Optional[Currency] = Field(None, description="Currency")
    amount_foreign_currency: Optional[float] = Field(None, description="Amount in foreign currency")
    
    # Product attributes
    product_class: Optional[str] = Field(None, description="Product class")
    product_style: Optional[str] = Field(None, description="Product style")
    style_colour_code: Optional[str] = Field(None, description="Style and color code")
    product_colour: Optional[str] = Field(None, description="Product color")
    gender_category: Optional[str] = Field(None, description="Gender category")
    garment_style: Optional[str] = Field(None, description="Garment style")
    
    # Additional flags and metadata
    is_staff_allowance: Optional[bool] = Field(None, description="Staff allowance flag")
    no_cash_payment: Optional[bool] = Field(None, description="No cash payment flag")
    week_number: Optional[int] = Field(None, description="Week number", ge=1, le=53)
    amount_discount: Optional[float] = Field(None, description="Discount amount", ge=0)
    
    # Organizational structure
    subsidiary: Optional[str] = Field(None, description="Subsidiary")
    reporting_region: Optional[str] = Field(None, description="Reporting region")
    market_channel: Optional[str] = Field(None, description="Market channel")
    range_segment: Optional[str] = Field(None, description="Range segment")
    
    @field_validator('transaction_date')
    @classmethod
    def validate_date_range(cls, v):
        """Validate date is within expected range."""
        min_date = datetime.strptime(MIN_DATE, "%Y-%m-%d").date()
        max_date = datetime.strptime(MAX_DATE, "%Y-%m-%d").date()
        
        if v < min_date or v > max_date:
            raise ValueError(f"Date {v} is outside expected range {MIN_DATE} to {MAX_DATE}")
        return v
    
    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("SKU cannot be empty")
        return v.strip().upper()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity is reasonable."""
        if v < MIN_QUANTITY:
            raise ValueError(f"Quantity {v} is below minimum {MIN_QUANTITY}")
        if v > MAX_QUANTITY:
            raise ValueError(f"Quantity {v} exceeds maximum {MAX_QUANTITY}")
        return v


class SalesDataSummary(BaseModel):
    """Summary statistics for sales data."""
    
    total_rows: int = Field(..., description="Total number of rows")
    date_range: Tuple[date, date] = Field(..., description="Date range (min, max)")
    unique_skus: int = Field(..., description="Number of unique SKUs")
    unique_markets: int = Field(..., description="Number of unique markets")
    total_quantity: float = Field(..., description="Total quantity sold")
    total_amount: float = Field(..., description="Total sales amount")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Data quality score")


class ForecastRequest(BaseModel):
    """Request schema for forecasting."""
    
    sku: str = Field(..., description="SKU to forecast")
    horizon_days: int = Field(30, ge=1, le=365, description="Forecast horizon in days")
    model_type: str = Field("arima", description="Forecasting model type")
    market: Optional[str] = Field(None, description="Market filter")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")


class ForecastResponse(BaseModel):
    """Response schema for forecasting."""
    
    sku: str = Field(..., description="Forecasted SKU")
    forecast_dates: list[date] = Field(..., description="Forecast dates")
    forecast_values: list[float] = Field(..., description="Forecasted quantities")
    confidence_intervals: Optional[list[Tuple[float, float]]] = Field(None, description="Confidence intervals")
    model_metrics: dict = Field(..., description="Model performance metrics")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Forecast creation time")


class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    
    total_rows: int
    valid_rows: int
    invalid_rows: int
    missing_values: dict[str, int]
    data_type_errors: dict[str, int]
    range_violations: dict[str, int]
    duplicate_rows: int
    quality_score: float = Field(..., ge=0.0, le=1.0)