"""
Constants and configuration for the demand planning system.

Note: The original XML was parsed cell-by-cell and must be reconstructed into proper rows.
The dataset contains ~19.2M rows of sales data spanning 2 years.
"""

from enum import Enum
from typing import List

# Canonical column names in order (32 columns total)
CANONICAL_COLUMNS: List[str] = [
    "Internal ID",
    "SKU", 
    "Date",
    "Period",
    "Document Number",
    "Name",
    "Order Type",
    "Customer Category",
    "Customer Subcategory", 
    "Market",
    "PO/Cheque Number",
    "Memo (Main)",
    "Description",
    "Quantity",
    "Account",
    "Amount (Net)",
    "Currency",
    "Amount (Foreign Currency)",
    "Class (MR std)",
    "Style",
    "StyleColourCode",
    "Colour",
    "Gender",
    "Garment Style",
    "Is Staff Allowance",
    "No cash payment",
    "Week #",
    "Amount Discount",
    "Subsidiary",
    "Reporting Region",
    "Market-Channel",
    "Range Segment",
]

# Column type mappings
DT_COLUMNS: List[str] = ["Date", "Period"]
NUM_COLUMNS: List[str] = [
    "Quantity", 
    "Amount (Net)", 
    "Amount (Foreign Currency)", 
    "Amount Discount"
]
STR_COLUMNS: List[str] = [
    col for col in CANONICAL_COLUMNS 
    if col not in DT_COLUMNS and col not in NUM_COLUMNS
]

# Key columns for demand planning
KEY_COLUMNS: List[str] = ["SKU", "Date", "Quantity", "Amount (Net)"]
REQUIRED_COLUMNS: List[str] = ["SKU", "Date", "Quantity"]

# Data quality thresholds
MIN_QUANTITY: float = 0.0
MAX_QUANTITY: float = 10000.0
MIN_AMOUNT: float = 0.0
MAX_AMOUNT: float = 100000.0

# Date range for validation
MIN_DATE: str = "2022-01-01"
MAX_DATE: str = "2024-12-31"


class Grain(str, Enum):
    """Data grain levels for aggregation."""
    SKU = "sku"
    MARKET = "market" 
    CHANNEL = "channel"
    PERIOD = "period"
    CUSTOMER_CATEGORY = "customer_category"
    PRODUCT_STYLE = "product_style"


class OrderType(str, Enum):
    """Order type enumeration."""
    SALE = "Sale"
    RETURN = "Return"
    EXCHANGE = "Exchange"
    REFUND = "Refund"


class CustomerCategory(str, Enum):
    """Customer category enumeration."""
    OWN_RETAIL = "Own Retail"
    BRAND = "Brand"
    WHOLESALE = "Wholesale"
    ONLINE = "Online"


class Market(str, Enum):
    """Market/region enumeration."""
    NZL_NEW_ZEALAND = "NZL_New Zealand"
    AUS_AUSTRALIA = "AUS_Australia"
    # Add more markets as needed


class Currency(str, Enum):
    """Currency enumeration."""
    NZD = "New Zealand Dollar"
    AUD = "Australian Dollar"
    USD = "US Dollar"
    EUR = "Euro"
