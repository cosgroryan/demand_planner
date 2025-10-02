"""
Create test data for demonstrating aggregation functionality.
"""

import polars as pl
from datetime import datetime, timedelta
import random

def create_test_sales_data(n_rows: int = 1000) -> pl.DataFrame:
    """Create synthetic sales data for testing."""
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Generate random dates
    dates = []
    for _ in range(n_rows):
        random_days = random.randint(0, (end_date - start_date).days)
        dates.append(start_date + timedelta(days=random_days))
    
    # Generate test data
    data = {
        "Internal ID": [f"ID_{i:06d}" for i in range(n_rows)],
        "SKU": [f"SKU-{random.randint(1000, 9999)}-{random.choice(['S', 'M', 'L', 'XL'])}" for _ in range(n_rows)],
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Period": [d.strftime("%b %Y") for d in dates],
        "Document Number": [f"DOC-{random.randint(10000, 99999)}" for _ in range(n_rows)],
        "Name": [f"Customer_{random.randint(1, 100)}" for _ in range(n_rows)],
        "Order Type": [random.choice(["Sale", "Return", "Exchange"]) for _ in range(n_rows)],
        "Customer Category": [random.choice(["Own Retail", "Brand", "Wholesale"]) for _ in range(n_rows)],
        "Customer Subcategory": [random.choice(["Premium", "Standard", "Budget"]) for _ in range(n_rows)],
        "Market": [random.choice(["NZL_New Zealand", "AUS_Australia", "USA_United States"]) for _ in range(n_rows)],
        "PO/Cheque Number": [f"PO-{random.randint(1000, 9999)}" for _ in range(n_rows)],
        "Memo (Main)": [f"Memo_{i}" for i in range(n_rows)],
        "Description": [f"Product Description {i}" for i in range(n_rows)],
        "Quantity": [random.randint(1, 10) for _ in range(n_rows)],
        "Account": [f"ACC-{random.randint(100, 999)}" for _ in range(n_rows)],
        "Amount (Net)": [round(random.uniform(50, 500), 2) for _ in range(n_rows)],
        "Currency": [random.choice(["New Zealand Dollar", "Australian Dollar", "US Dollar"]) for _ in range(n_rows)],
        "Amount (Foreign Currency)": [round(random.uniform(50, 500), 2) for _ in range(n_rows)],
        "Class (MR std)": [random.choice(["Outerwear", "Base Layer", "Accessories"]) for _ in range(n_rows)],
        "Style": [random.choice(["Atmos", "Merino", "Tech", "Classic"]) for _ in range(n_rows)],
        "StyleColourCode": [f"{random.choice(['BLK', 'WHT', 'BLU', 'GRN'])}-{random.randint(100, 999)}" for _ in range(n_rows)],
        "Colour": [random.choice(["Black", "White", "Blue", "Green", "Red"]) for _ in range(n_rows)],
        "Gender": [random.choice(["Mens", "Womens", "Unisex"]) for _ in range(n_rows)],
        "Garment Style": [random.choice(["Hoodie", "Jacket", "T-Shirt", "Pants"]) for _ in range(n_rows)],
        "Is Staff Allowance": [random.choice([True, False]) for _ in range(n_rows)],
        "No cash payment": [random.choice([True, False]) for _ in range(n_rows)],
        "Week #": [random.randint(1, 52) for _ in range(n_rows)],
        "Amount Discount": [round(random.uniform(0, 50), 2) for _ in range(n_rows)],
        "Subsidiary": [random.choice(["NZ", "AU", "US"]) for _ in range(n_rows)],
        "Reporting Region": [random.choice(["APAC", "Americas", "Europe"]) for _ in range(n_rows)],
        "Market-Channel": [f"{random.choice(['NZL', 'AUS', 'USA'])}_Retail" for _ in range(n_rows)],
        "Range Segment": [random.choice(["Premium", "Standard", "Budget"]) for _ in range(n_rows)]
    }
    
    return pl.DataFrame(data)

if __name__ == "__main__":
    # Create test data
    test_df = create_test_sales_data(1000)
    
    # Save to parquet
    test_df.write_parquet("data/processed/test_sales.parquet")
    print(f"Created test data with {len(test_df)} rows")
    print(f"Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Unique SKUs: {test_df['SKU'].n_unique()}")
    print(f"Unique Markets: {test_df['Market'].n_unique()}")
