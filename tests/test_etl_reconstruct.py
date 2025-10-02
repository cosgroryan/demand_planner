"""
Tests for the ETL reconstruction pipeline.
"""

import pytest
import polars as pl
from pathlib import Path
import tempfile
import shutil
from datetime import date

from dp.constants import CANONICAL_COLUMNS, DT_COLUMNS, NUM_COLUMNS
from dp.etl_reconstruct import (
    reconstruct_table,
    identify_headers,
    reconstruct_rows,
    validate_reconstructed_data
)
from dp.schemas import SalesRow


class TestETLReconstruct:
    """Test cases for ETL reconstruction functionality."""
    
    def test_canonical_columns_defined(self):
        """Test that canonical columns are properly defined."""
        assert len(CANONICAL_COLUMNS) == 32
        assert "SKU" in CANONICAL_COLUMNS
        assert "Date" in CANONICAL_COLUMNS
        assert "Quantity" in CANONICAL_COLUMNS
        assert "Amount (Net)" in CANONICAL_COLUMNS
    
    def test_column_types_defined(self):
        """Test that column type mappings are defined."""
        assert len(DT_COLUMNS) > 0
        assert len(NUM_COLUMNS) > 0
        assert "Date" in DT_COLUMNS
        assert "Quantity" in NUM_COLUMNS
        assert "Amount (Net)" in NUM_COLUMNS
    
    def test_identify_headers(self):
        """Test header identification logic."""
        # Create test data with known headers
        test_data = pl.DataFrame({
            "data_value": [
                "Internal ID", "SKU", "Date", "Quantity",  # Headers
                "12345", "ABC-123", "2023-01-01", "5",     # Data
                "67890", "DEF-456", "2023-01-02", "3"      # More data
            ],
            "data_type": ["String"] * 12
        })
        
        headers = identify_headers(test_data)
        
        # Should identify the known headers
        assert "Internal ID" in headers
        assert "SKU" in headers
        assert "Date" in headers
        assert "Quantity" in headers
    
    def test_reconstruct_rows_basic(self):
        """Test basic row reconstruction."""
        # Create test data
        test_data = pl.DataFrame({
            "data_value": [
                "Internal ID", "12345",
                "SKU", "ABC-123",
                "Date", "2023-01-01",
                "Quantity", "5"
            ],
            "data_type": ["String"] * 8
        })
        
        headers = ["Internal ID", "SKU", "Date", "Quantity"]
        result = reconstruct_rows(test_data, headers, 1000)
        
        # Should create a DataFrame with the expected columns
        assert isinstance(result, pl.DataFrame)
        assert len(result.columns) == len(CANONICAL_COLUMNS)
        
        # Check that canonical columns are present
        for col in CANONICAL_COLUMNS:
            assert col in result.columns
    
    def test_reconstruct_table_integration(self):
        """Test the full reconstruction pipeline."""
        # Create a temporary test file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a minimal test parquet file
            test_data = pl.DataFrame({
                '{urn:schemas-microsoft-com:office:spreadsheet}Data': [
                    "Internal ID", "SKU", "Date", "Quantity",
                    "12345", "ABC-123", "2023-01-01", "5"
                ],
                '{urn:schemas-microsoft-com:office:spreadsheet}Data_{urn:schemas-microsoft-com:office:spreadsheet}Type': [
                    "String", "String", "String", "String",
                    "String", "String", "String", "String"
                ]
            })
            
            source_file = temp_path / "test_source.parquet"
            output_file = temp_path / "test_output.parquet"
            
            test_data.write_parquet(source_file)
            
            # Run reconstruction
            stats = reconstruct_table(
                source_path=str(source_file),
                output_path=str(output_file),
                sample_size=100
            )
            
            # Verify output file exists
            assert output_file.exists()
            
            # Verify stats
            assert "total_rows" in stats
            assert "total_columns" in stats
            assert "data_quality_score" in stats
            assert stats["total_columns"] == len(CANONICAL_COLUMNS) + 2  # +2 for year/month
    
    def test_validate_reconstructed_data(self):
        """Test data validation against schema."""
        # Create test data that should validate
        test_data = pl.DataFrame({
            "SKU": ["ABC-123", "DEF-456"],
            "Date": ["2023-01-01", "2023-01-02"],
            "Quantity": [5.0, 3.0],
            "Amount (Net)": [100.0, 75.0]
        })
        
        # Add all canonical columns with None values
        for col in CANONICAL_COLUMNS:
            if col not in test_data.columns:
                test_data = test_data.with_columns(pl.lit(None).alias(col))
        
        validation_results = validate_reconstructed_data(test_data)
        
        # Should have some valid rows
        assert "valid_rows" in validation_results
        assert "invalid_rows" in validation_results
        assert "errors" in validation_results
    
    def test_sales_row_schema_validation(self):
        """Test that SalesRow schema validates correctly."""
        # Test valid data
        valid_data = {
            "sku": "ABC-123",
            "date": date(2023, 1, 1),
            "quantity": 5.0,
            "amount_net": 100.0
        }
        
        sales_row = SalesRow(**valid_data)
        assert sales_row.sku == "ABC-123"
        assert sales_row.quantity == 5.0
        
        # Test invalid data
        with pytest.raises(ValueError):
            SalesRow(sku="", date=date(2023, 1, 1), quantity=5.0)
        
        with pytest.raises(ValueError):
            SalesRow(sku="ABC-123", date=date(2023, 1, 1), quantity=-1.0)
    
    def test_data_quality_metrics(self):
        """Test data quality calculation."""
        # Create test data with some missing values
        test_data = pl.DataFrame({
            "SKU": ["ABC-123", "DEF-456", None],
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Quantity": [5.0, 3.0, 2.0]
        })
        
        # Add all canonical columns
        for col in CANONICAL_COLUMNS:
            if col not in test_data.columns:
                test_data = test_data.with_columns(pl.lit(None).alias(col))
        
        # Calculate quality score
        total_cells = len(test_data) * len(test_data.columns)
        null_cells = test_data.null_count().sum_horizontal().item()
        quality_score = 1.0 - (null_cells / total_cells)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score < 1.0  # Should be less than 1 due to missing values
    
    def test_date_validation(self):
        """Test date validation in schema."""
        # Test valid date range
        valid_date = date(2023, 6, 15)
        sales_row = SalesRow(
            sku="ABC-123",
            date=valid_date,
            quantity=5.0
        )
        assert sales_row.transaction_date == valid_date
        
        # Test date outside range
        with pytest.raises(ValueError):
            SalesRow(
                sku="ABC-123",
                date=date(2020, 1, 1),  # Before MIN_DATE
                quantity=5.0
            )
    
    def test_sku_normalization(self):
        """Test SKU normalization in schema."""
        # Test SKU gets normalized to uppercase
        sales_row = SalesRow(
            sku="abc-123",
            date=date(2023, 1, 1),
            quantity=5.0
        )
        assert sales_row.sku == "ABC-123"
        
        # Test whitespace gets stripped
        sales_row = SalesRow(
            sku="  ABC-123  ",
            date=date(2023, 1, 1),
            quantity=5.0
        )
        assert sales_row.sku == "ABC-123"
    
    def test_quantity_validation(self):
        """Test quantity validation in schema."""
        # Test valid quantities
        valid_quantities = [0.0, 1.0, 100.0, 9999.0]
        for qty in valid_quantities:
            sales_row = SalesRow(
                sku="ABC-123",
                date=date(2023, 1, 1),
                quantity=qty
            )
            assert sales_row.quantity == qty
        
        # Test invalid quantities
        with pytest.raises(ValueError):
            SalesRow(
                sku="ABC-123",
                date=date(2023, 1, 1),
                quantity=-1.0
            )
        
        with pytest.raises(ValueError):
            SalesRow(
                sku="ABC-123",
                date=date(2023, 1, 1),
                quantity=10001.0
            )
