# Demand Planner

A modern demand planning system for retail sales forecasting built with Python, FastAPI, and Streamlit.

## 🚀 Features

- **Multi-Model Forecasting**: Automatic model selection from baseline, statistical, and ML models
- **Time Series Aggregation**: Weekly and monthly demand series at SKU and Market-Channel levels
- **Feature Engineering**: Calendar features, rolling statistics, seasonality indicators, and product attributes
- **Rolling Backtests**: Comprehensive model evaluation with MAPE, sMAPE, RMSE metrics
- **Inventory Planning**: Reorder points, safety stock, and order scheduling
- **REST API**: FastAPI backend with comprehensive endpoints
- **Web UI**: Streamlit interface for interactive forecasting
- **Excel Integration**: Planner-friendly tables with CSV/Excel export

## 📋 Requirements

- Python 3.9+
- uv (recommended) or pip
- 8GB+ RAM for large datasets
- 10GB+ disk space for data processing

## 🛠️ Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd demand_planner

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd demand_planner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 📊 Data Structure

### Input Data Format

The system expects sales data in Parquet format with the following columns:

- `Date`: Transaction date (YYYY-MM-DD)
- `SKU`: Product identifier
- `Market-Channel`: Market and channel combination (e.g., "US-Retail")
- `Style`, `Colour`, `Gender`, `Range Segment`: Product attributes
- `Quantity`: Units sold
- `Amount (Net)`: Net sales amount
- `Amount Discount`: Discount amount
- `Customer Category`: Customer type
- `Is Staff Allowance`: Boolean flag

### Data Placement Rules

```
data/
├── raw/                    # Raw input data
│   └── sales.parquet
├── processed/              # Cleaned and processed data
│   ├── sales_clean.parquet
│   ├── series/             # Time series aggregations
│   │   ├── series_SKU_W.parquet
│   │   └── series_SKU_Market-Channel_W.parquet
│   └── features/           # Engineered features
│       └── features_SKU.parquet
└── reports/                # Analysis outputs
    ├── backtests/          # Backtest results
    └── planning/           # Planning tables
```

## 🚀 Quick Start

### 1. Data Processing

```bash
# Reconstruct and clean raw data
uv run dp etl --reconstruct --src data/raw/sales.parquet --out data/processed/sales_clean.parquet

# Aggregate into time series
uv run dp aggregate --all

# Engineer features
uv run dp features --input data/processed/series/series_SKU_W.parquet
```

### 2. Generate Forecasts

```bash
# Generate forecast for a specific SKU
uv run dp forecast SKU001 --model auto --horizon 13

# Run backtests
uv run dp backtest --grain SKU --model auto --horizon 13
```

### 3. Start API Server

```bash
# Start FastAPI server
uv run dp api --serve --host 0.0.0.0 --port 8000
```

### 4. Launch Web UI

```bash
# Start Streamlit UI
uv run streamlit run app/app.py
```

## 📚 Usage Guide

### CLI Commands

#### ETL Operations
```bash
# Reconstruct flattened data
dp etl --reconstruct --src data/raw/sales.parquet --out data/processed/sales_clean.parquet

# Process sample data
dp etl --reconstruct --src data/raw/sales.parquet --sample-size 10000
```

#### Aggregation
```bash
# Aggregate all grain combinations
dp aggregate --all

# Aggregate specific grain
dp aggregate --grain SKU,Market-Channel --freq W
```

#### Feature Engineering
```bash
# Engineer features from series data
dp features --input data/processed/series/series_SKU_W.parquet
```

#### Forecasting
```bash
# Generate forecast
dp forecast SKU001 --model auto --horizon 13

# Use specific model
dp forecast SKU001 --model random_forest --horizon 26
```

#### Backtesting
```bash
# Run backtests
dp backtest --grain SKU --model auto --horizon 13 --history 52

# Monthly backtests
dp backtest --grain SKU --freq M --horizon 6
```

#### API Server
```bash
# Start server
dp api --serve --host 0.0.0.0 --port 8000

# Production server
dp api --serve --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Available SKUs
```bash
curl "http://localhost:8000/catalog/sku?page=1&page_size=50"
```

#### Generate Forecast
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "sku": "SKU001",
    "market_channel": "US-Retail",
    "freq": "W",
    "horizon": 13,
    "model": "auto"
  }'
```

#### Get Planning Table
```bash
curl "http://localhost:8000/planning/SKU001?horizon=13&include_confidence_bands=true"
```

#### Download Planning Table as CSV
```bash
curl "http://localhost:8000/planning/SKU001.csv?horizon=13" -o planning_table.csv
```

#### Get Inventory Signals
```bash
curl "http://localhost:8000/inventory/SKU001?current_on_hand=100&lead_time_periods=4&target_service_level=0.95"
```

### Web UI

The Streamlit UI provides an interactive interface for:

- **SKU Selection**: Browse and search available SKUs
- **Forecast Generation**: Configure and generate forecasts
- **Visualization**: Interactive charts with confidence intervals
- **Export**: Download forecasts as CSV
- **Model Information**: View model selection and demand patterns

Access the UI at: http://localhost:8501

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_LOG_LEVEL=info

# Data Configuration
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
DATA_SERIES_DIR=data/processed/series
DATA_FEATURES_DIR=data/processed/features
REPORTS_DIR=reports
BACKTESTS_DIR=reports/backtests
PLANNING_DIR=reports/planning

# Model Configuration
DEFAULT_MODEL=auto
DEFAULT_HORIZON=13
DEFAULT_HISTORY_WINDOW=52
DEFAULT_STEP_SIZE=4
AUTO_MODEL_SELECTION=true
PARALLEL_PROCESSING=true
MAX_WORKERS=4

# Inventory Configuration
DEFAULT_SERVICE_LEVEL=0.95
DEFAULT_LEAD_TIME_PERIODS=4
DEFAULT_SAFETY_STOCK_MULTIPLIER=1.0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/demand_planner.log
LOG_CONSOLE_OUTPUT=true
LOG_RICH_LOGGING=true

# UI Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
API_BASE_URL=http://localhost:8000
```

### Configuration File

The system uses `src/dp/config.py` for configuration management. Settings can be overridden via environment variables.

## 📈 Models

### Baseline Models
- **Naive**: Last observed value
- **Seasonal Naive**: Same season last year
- **Moving Average**: Simple moving average
- **Croston's Method**: For intermittent demand
- **Exponential Smoothing**: Simple exponential smoothing

### Statistical Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMAX**: Seasonal ARIMA with exogenous variables
- **Auto ARIMA**: Automatic parameter selection

### Machine Learning Models
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Gradient boosted trees

### Automatic Model Selection

The system automatically selects the best model based on demand characteristics:

- **Intermittent demand** (>50% zero periods): Croston's method
- **Smooth demand** (CV < 0.3): Exponential smoothing
- **Erratic demand** (CV > 1.5): Moving average
- **Sufficient data** (≥52 periods): Auto ARIMA
- **Short series**: Naive forecast

## 📊 Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **Weighted MAPE**: Volume-weighted MAPE
- **Theil's U**: Relative to naive forecast

## 🏭 Inventory Planning

### Key Metrics
- **Reorder Point**: When to place new orders
- **Order Up-To Level**: Target inventory level
- **Safety Stock**: Buffer for demand variability
- **Stockout Risk**: Probability of stockout
- **Weeks of Cover**: Current inventory coverage
- **Turnover Rate**: Annual inventory turnover

### Order Scheduling
The system generates suggested order schedules based on:
- Forecasted demand
- Lead times
- Service level targets
- Minimum order quantities
- Order multiples

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Issues with Large Datasets
```bash
# Process data in chunks
dp etl --reconstruct --src data/raw/sales.parquet --sample-size 100000

# Use fewer workers
export MAX_WORKERS=2
```

#### 2. API Connection Issues
```bash
# Check API health
curl http://localhost:8000/health

# Check logs
tail -f logs/demand_planner.log
```

#### 3. Missing Dependencies
```bash
# Reinstall dependencies
uv sync --reinstall

# Or with pip
pip install -r requirements.txt
```

#### 4. Data Format Issues
- Ensure dates are in YYYY-MM-DD format
- Check for missing required columns
- Validate numeric columns don't contain text

### Performance Optimization

#### For Large Datasets (>1M rows)
- Use `--sample-size` for initial testing
- Process data in chunks
- Increase `MAX_WORKERS` for parallel processing
- Use SSD storage for better I/O performance

#### For High-Frequency Updates
- Enable model caching
- Use background tasks for long-running operations
- Implement data partitioning by date

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation at http://localhost:8000/docs
- Open an issue on GitHub

## 🔄 Version History

- **v1.0.0**: Initial release with core forecasting capabilities
- **v1.1.0**: Added inventory planning and Excel integration
- **v1.2.0**: Enhanced UI and batch processing capabilities

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)