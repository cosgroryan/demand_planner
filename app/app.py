"""
Streamlit UI for the Demand Planning system.

This provides a web interface for generating forecasts, viewing historical data,
and analyzing demand patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

# Page configuration
st.set_page_config(
    page_title="Demand Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_parent_skus() -> List[str]:
    """Get list of available parent SKUs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/parent-skus")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def get_available_market_channels() -> List[str]:
    """Get list of available market channels from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/catalog/market-channels")
        if response.status_code == 200:
            data = response.json()
            return data.get("market_channels", [])
    except:
        pass
    return []

def get_parent_series_data(parent_sku: str, limit: int = 52) -> Optional[Dict]:
    """Get historical data for a parent SKU."""
    try:
        params = {"limit": limit}
        response = requests.get(f"{API_BASE_URL}/parent-sku/{parent_sku}/historical", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Series API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error getting series data: {e}")
    return None

def get_grouped_series_data(sku_pattern: str, market_channel: Optional[str] = None, 
                           freq: str = "W", limit: int = 52, grouping_level: str = "Group by Parent") -> Optional[Dict]:
    """Get aggregated data for multiple SKUs matching a pattern."""
    try:
        print(f"DEBUG: get_grouped_series_data called with sku_pattern='{sku_pattern}', grouping_level='{grouping_level}'")
        
        # Get all available SKUs
        available_skus = get_available_skus()
        print(f"DEBUG: Found {len(available_skus)} available SKUs")
        
        # Find matching SKUs based on the pattern
        if grouping_level == "Group by Parent":
            # Match by parent (first 6 digits)
            matching_skus = [sku for sku in available_skus if sku.startswith(sku_pattern)]
            print(f"DEBUG: Group by Parent - Found {len(matching_skus)} matching SKUs for pattern '{sku_pattern}'")
        elif grouping_level == "Group by Parent-Graphic":
            # Match by parent-graphic (first 6-4 digits)
            matching_skus = [sku for sku in available_skus if sku.startswith(sku_pattern)]
            print(f"DEBUG: Group by Parent-Graphic - Found {len(matching_skus)} matching SKUs for pattern '{sku_pattern}'")
        else:
            matching_skus = [sku_pattern]
            print(f"DEBUG: Individual SKU - Using pattern '{sku_pattern}' directly")
        
        if not matching_skus:
            print(f"DEBUG: No matching SKUs found for pattern '{sku_pattern}'")
            return None
        
        print(f"DEBUG: Matching SKUs: {matching_skus[:5]}...")  # Show first 5
        
        # Get data for all matching SKUs
        all_series_data = []
        print(f"DEBUG: Fetching data for {len(matching_skus)} SKUs...")
        for i, sku in enumerate(matching_skus):
            params = {"freq": freq, "limit": limit}
            if market_channel:
                params["market_channel"] = market_channel
            
            response = requests.get(f"{API_BASE_URL}/series/{sku}", params=params)
            print(f"DEBUG: SKU {i+1}/{len(matching_skus)}: {sku} - Status: {response.status_code}")
            if response.status_code == 200:
                all_series_data.append(response.json())
            else:
                print(f"DEBUG: Failed to get data for SKU {sku}: {response.text}")
        
        print(f"DEBUG: Successfully fetched data for {len(all_series_data)} SKUs")
        if not all_series_data:
            print("DEBUG: No series data collected, returning None")
            return None
        
        # Aggregate the data
        print("DEBUG: Aggregating series data...")
        result = aggregate_series_data(all_series_data, sku_pattern, grouping_level)
        print(f"DEBUG: Aggregation complete, result type: {type(result)}")
        return result
        
    except Exception as e:
        print(f"Error in get_grouped_series_data: {e}")
        return None

def aggregate_series_data(series_data_list: List[Dict], sku_pattern: str, grouping_level: str) -> Dict:
    """Aggregate multiple series data into a single series."""
    if not series_data_list:
        return None
    
    # Get the first series as base
    base_series = series_data_list[0]
    
    # Initialize aggregated data
    aggregated = {
        "sku": sku_pattern,
        "market_channel": base_series.get("market_channel"),
        "periods": base_series["periods"],
        "units": [0.0] * len(base_series["periods"]),
        "net_sales": [0.0] * len(base_series["periods"]),
        "avg_selling_price": [0.0] * len(base_series["periods"]),
        "discount_rate": [0.0] * len(base_series["periods"])
    }
    
    # Sum up all the data
    for series in series_data_list:
        for i in range(len(series["periods"])):
            aggregated["units"][i] += series["units"][i]
            aggregated["net_sales"][i] += series["net_sales"][i]
            # For avg_selling_price, we'll calculate it from net_sales / units
            if series["units"][i] > 0:
                price = series["net_sales"][i] / series["units"][i]
                aggregated["avg_selling_price"][i] = price
    
    # Calculate weighted average selling price as a single number
    total_units = sum(aggregated["units"])
    if total_units > 0:
        total_sales = sum(aggregated["net_sales"])
        avg_price = total_sales / total_units
        aggregated["avg_selling_price"] = avg_price  # Single number, not a list
    else:
        aggregated["avg_selling_price"] = 25.0  # Default price
    
    return aggregated

def generate_parent_forecast(parent_sku: str, horizon: int = 13, model: str = "auto") -> Optional[Dict]:
    """Generate forecast for a parent SKU."""
    try:
        endpoint = f"{API_BASE_URL}/parent-sku/{parent_sku}/forecast"
        params = {
            "horizon": horizon,
            "model": model
        }
        
        response = requests.post(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Forecast API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error generating forecast: {e}")
    return None


def create_parent_forecast_chart(series_data: Dict, forecast_data: Dict) -> go.Figure:
    """Create a plotly chart showing historical data and forecast for parent SKU."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Units Forecast", "Sales Forecast"),
        vertical_spacing=0.1
    )
    
    # Historical data
    hist_periods = pd.to_datetime(series_data["periods"])
    hist_units = series_data["units"]
    hist_sales = series_data["net_sales"]
    
    # Forecast data
    forecast_periods = pd.to_datetime(forecast_data["forecast_periods"])
    forecast_units = forecast_data["forecast_units"]
    forecast_sales = forecast_data["forecast_sales"]
    
    # Add historical data (solid lines)
    fig.add_trace(
        go.Scatter(
            x=hist_periods, y=hist_units,
            mode='lines',
            name='Historical Units',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist_periods, y=hist_sales,
            mode='lines',
            name='Historical Sales',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Add forecast data (dotted lines)
    fig.add_trace(
        go.Scatter(
            x=forecast_periods, y=forecast_units,
            mode='lines',
            name='Forecast Units',
            line=dict(color='red', width=2, dash='dot')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_periods, y=forecast_sales,
            mode='lines',
            name='Forecast Sales',
            line=dict(color='orange', width=2, dash='dot')
        ),
        row=2, col=1
    )
    
    # Add "Today" marker
    today = datetime.now()
    fig.add_vline(x=today, line_dash="dash", line_color="red", opacity=0.7)
    fig.add_annotation(
        x=today, y=max(max(hist_units), max(forecast_units)),
        text="Today",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red"
    )
    
    # Update layout
    fig.update_layout(
        title=f"Parent SKU {series_data['parent_sku']} - Demand Forecast",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Sales ($)", row=2, col=1)
    
    return fig
    hist_periods = pd.to_datetime(series_data["periods"])
    hist_units = series_data["units"]
    hist_sales = series_data["net_sales"]
    
    # Prepare forecast data with bulletproof extraction
    forecast_values = forecast_data["forecast"]
    prediction_intervals = forecast_data.get("prediction_intervals", [])
    
    # Debug: print the raw forecast data
    print(f"DEBUG: Raw forecast_values type: {type(forecast_values)}")
    print(f"DEBUG: Raw forecast_values length: {len(forecast_values)}")
    if forecast_values:
        print(f"DEBUG: Raw first element type: {type(forecast_values[0])}")
        print(f"DEBUG: Raw first element: {forecast_values[0]}")
    
    # Bulletproof forecast values extraction
    if isinstance(forecast_values, list):
        if forecast_values and isinstance(forecast_values[0], list):
            # If it's a list of lists, flatten it
            forecast_values = [item for sublist in forecast_values for item in sublist]
        # Ensure all values are numbers
        forecast_values = [float(val) if isinstance(val, (int, float)) else 0.0 for val in forecast_values]
    else:
        forecast_values = [0.0]
    
    # Create future periods
    last_period = hist_periods[-1]  # Fix: use indexing instead of iloc for DatetimeIndex
    if series_data.get("freq", "W") == "W":
        freq_str = "W"
    else:
        freq_str = "M"
    
    future_periods = pd.date_range(
        start=last_period + pd.Timedelta(weeks=1 if freq_str == "W" else pd.DateOffset(months=1)),
        periods=len(forecast_values),
        freq=freq_str
    )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Units Forecast", "Net Sales Forecast"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Historical data (solid line)
    fig.add_trace(
        go.Scatter(
            x=hist_periods,
            y=hist_units,
            mode='lines+markers',
            name='Historical Units',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=5),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Units: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist_periods,
            y=hist_sales,
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=5),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add "Today" marker (vertical line at the end of historical data)
    today_x = last_period
    
    # Add vertical line without annotation to avoid Plotly arithmetic issues
    fig.add_vline(
        x=today_x,
        line_dash="dot",
        line_color="red",
        line_width=2,
        row=1, col=1
    )
    fig.add_vline(
        x=today_x,
        line_dash="dot", 
        line_color="red",
        line_width=2,
        row=2, col=1
    )
    
    # Add "Today" text annotation manually
    fig.add_annotation(
        x=today_x,
        y=0.95,
        yref="paper",
        text="Today",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=0,
        ay=-40,
        font=dict(color="red", size=12),
        row=1, col=1
    )
    
    # Forecast data (dotted line)
    fig.add_trace(
        go.Scatter(
            x=future_periods,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast Units',
            line=dict(color='#2ca02c', width=3, dash='dot'),
            marker=dict(size=5),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Units: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add forecast sales (dotted line)
    # Handle avg_selling_price - it might be a list (from individual SKU) or single number (from aggregated data)
    avg_price = series_data.get("avg_selling_price", 25)
    if isinstance(avg_price, list):
        # Find first non-zero value or use default
        avg_price = next((p for p in avg_price if p > 0), 25)
    
    # Ensure both forecast_values and avg_price are numbers
    forecast_values = [float(val) if isinstance(val, (int, float, str)) else 0.0 for val in forecast_values]
    avg_price = float(avg_price) if isinstance(avg_price, (int, float, str)) else 25.0
    
    forecast_sales = [val * avg_price for val in forecast_values]
    fig.add_trace(
        go.Scatter(
            x=future_periods,
            y=forecast_sales,
            mode='lines+markers',
            name='Forecast Sales',
            line=dict(color='#d62728', width=3, dash='dot'),
            marker=dict(size=5),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Sales: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Prediction intervals with better shading
    if prediction_intervals:
        lower_bounds = [interval[0] for interval in prediction_intervals]
        upper_bounds = [interval[1] for interval in prediction_intervals]
        
        # Units prediction interval
        fig.add_trace(
            go.Scatter(
                x=future_periods,
                y=upper_bounds,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_periods,
                y=lower_bounds,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.3)',
                name='95% Confidence Interval',
                hoverinfo='skip',
                hovertemplate='<b>Confidence Interval</b><br>Date: %{x}<br>Range: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Sales prediction interval
        # Process bounds the same way as forecast values
        if isinstance(lower_bounds, list) and lower_bounds and isinstance(lower_bounds[0], list):
            lower_bounds = [item for sublist in lower_bounds for item in sublist]
        lower_bounds = [float(val) if isinstance(val, (int, float)) else 0.0 for val in lower_bounds]
        
        if isinstance(upper_bounds, list) and upper_bounds and isinstance(upper_bounds[0], list):
            upper_bounds = [item for sublist in upper_bounds for item in sublist]
        upper_bounds = [float(val) if isinstance(val, (int, float)) else 0.0 for val in upper_bounds]
            
        lower_sales = [val * avg_price for val in lower_bounds]
        upper_sales = [val * avg_price for val in upper_bounds]
        
        fig.add_trace(
            go.Scatter(
                x=future_periods,
                y=upper_sales,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_periods,
                y=lower_sales,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(214, 39, 40, 0.3)',
                name='95% Confidence Interval',
                hoverinfo='skip',
                hovertemplate='<b>Confidence Interval</b><br>Date: %{x}<br>Range: $%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Demand Forecast for {series_data['sku']}",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Net Sales ($)", row=2, col=1)
    
    return fig

def create_metrics_dashboard(series_data: Dict, forecast_data: Dict) -> None:
    """Create a metrics dashboard."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_units = np.mean(series_data["units"])
        st.metric(
            label="Avg Historical Units",
            value=f"{avg_units:.1f}",
            delta=None
        )
    
    with col2:
        avg_forecast = np.mean(forecast_data["forecast"])
        st.metric(
            label="Avg Forecast Units",
            value=f"{avg_forecast:.1f}",
            delta=f"{((avg_forecast - avg_units) / avg_units * 100):+.1f}%" if avg_units > 0 else None
        )
    
    with col3:
        total_historical_sales = sum(series_data["net_sales"])
        st.metric(
            label="Total Historical Sales",
            value=f"${total_historical_sales:,.0f}",
            delta=None
        )
    
    with col4:
        model_used = forecast_data.get("model_used", "Unknown")
        st.metric(
            label="Model Used",
            value=model_used,
            delta=None
        )

def main():
    """Main Streamlit application."""
    
    # Initialize session state for SKU selection
    if 'selected_sku_index' not in st.session_state:
        st.session_state.selected_sku_index = 0
    
    # Header
    st.markdown('<h1 class="main-header">📊 Demand Planner</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API server is not running. Please start the API server first:")
        st.code("uv run uvicorn src.dp.api:app --reload")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Get available data
    with st.spinner("Loading available parent SKUs..."):
        available_parent_skus = get_available_parent_skus()
    
    with st.spinner("Loading market channels..."):
        available_channels = get_available_market_channels()
    
    if not available_parent_skus:
        st.error("No SKUs found. Please ensure data has been processed.")
        st.stop()
    
    # Parent SKU Selection
    st.sidebar.markdown("### 🔍 Parent SKU Selection")
    st.sidebar.markdown("**Format**: `100275` (Parent SKU only)")
    st.sidebar.markdown("**The system will aggregate all child SKUs under this parent.**")
    
    # Parent SKU selection
    selected_parent_sku = st.sidebar.selectbox(
        "Select Parent SKU",
        available_parent_skus,
        index=st.session_state.selected_sku_index if st.session_state.selected_sku_index < len(available_parent_skus) else 0,
        key="parent_sku_selectbox"
    )
    
    # Update session state when selection changes
    if selected_parent_sku:
        st.session_state.selected_sku_index = available_parent_skus.index(selected_parent_sku)
    
    # Display parent SKU info
    if selected_parent_sku:
        st.sidebar.success(f"Selected Parent SKU: {selected_parent_sku}")
    
    selected_channel = st.sidebar.selectbox(
        "Select Market Channel (Optional)",
        ["None"] + available_channels,
        index=0
    )
    
    if selected_channel == "None":
        selected_channel = None
    
    freq = st.sidebar.selectbox(
        "Frequency",
        ["W", "M"],
        index=0,
        format_func=lambda x: "Weekly" if x == "W" else "Monthly"
    )
    
    horizon = st.sidebar.slider(
        "Forecast Horizon",
        min_value=1,
        max_value=52,
        value=13,
        help="Number of periods to forecast ahead"
    )
    
    model = st.sidebar.selectbox(
        "Model",
        ["auto", "naive", "seasonal_naive", "moving_average", "croston", 
         "exponential_smoothing", "arima", "sarimax", "auto_arima", 
         "random_forest", "gradient_boosting"],
        index=0,
        help="Forecasting model to use"
    )
    
    # Main content
    if st.sidebar.button("🚀 Generate Forecast", type="primary"):
        
        with st.spinner("Loading historical data..."):
            series_data = get_parent_series_data(
                selected_parent_sku, 
                limit=52
            )
        
        if not series_data:
            st.error(f"No historical data found for Parent SKU: {selected_parent_sku}")
            st.stop()
        
        with st.spinner("Generating forecast..."):
            forecast_data = generate_parent_forecast(
                selected_parent_sku,
                horizon,
                model
            )
        
        if not forecast_data:
            st.error("Failed to generate forecast. Please check the logs.")
            st.stop()
        
        # Display results
        st.success("✅ Forecast generated successfully!")
        
        # Metrics dashboard
        create_metrics_dashboard(series_data, forecast_data)
        
        # Forecast chart
        st.markdown("### 📈 Forecast Visualization")
        fig = create_parent_forecast_chart(series_data, forecast_data)
        st.plotly_chart(fig, config={"displayModeBar": True, "responsive": True})
        
        # Forecast table
        st.markdown("### 📋 Forecast Details")
        
        # Prepare forecast table
        forecast_df = pd.DataFrame({
            "Period": [f"T+{i+1}" for i in range(len(forecast_data["forecast"]))],
            "Forecast": forecast_data["forecast"],
            "Lower Bound": [interval[0] for interval in forecast_data.get("prediction_intervals", [])],
            "Upper Bound": [interval[1] for interval in forecast_data.get("prediction_intervals", [])]
        })
        
        st.dataframe(forecast_df, width='stretch')
        
        # Model information
        st.markdown("### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model Used:** {forecast_data.get('model_used', 'Unknown')}")
            st.info(f"**Confidence Level:** {forecast_data.get('confidence_level', 0.95):.1%}")
        
        with col2:
            if forecast_data.get("pattern_info"):
                pattern = forecast_data["pattern_info"].get("pattern", "Unknown")
                st.info(f"**Demand Pattern:** {pattern}")
            
            st.info(f"**Forecast Date:** {forecast_data.get('forecast_date', 'Unknown')}")
        
        # Download forecast
        st.markdown("### 💾 Download Forecast")
        
        csv_data = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data,
            file_name=f"forecast_{selected_sku}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to the Demand Planner! 🎯
        
        This application helps you generate accurate demand forecasts for your retail products.
        
        **Features:**
        - 📊 Interactive forecasting with multiple models
        - 📈 Visual charts with confidence intervals
        - 🎯 Automatic model selection based on demand patterns
        - 📋 Detailed forecast tables
        - 💾 Export forecasts to CSV
        - 🏷️ SKU hierarchy support (Parent-Style-Colour-Size)
        
        **Getting Started:**
        1. Select a SKU from the sidebar
        2. Optionally choose a market channel
        3. Configure forecast parameters
        4. Click "Generate Forecast" to see results
        
        **Available Models:**
        - **Auto**: Automatically selects the best model based on demand characteristics
        - **Baseline**: Naive, seasonal naive, moving average, Croston's method
        - **Statistical**: ARIMA, SARIMAX with automatic parameter selection
        - **Machine Learning**: Random Forest, Gradient Boosting
        """)
        
        # Show available data summary
        st.markdown("### 📊 Available Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Available Parent SKUs", len(available_parent_skus))
        
        with col2:
            st.metric("Market Channels", len(available_channels))

if __name__ == "__main__":
    main()
