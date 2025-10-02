"""
Streamlit UI for the Parent SKU Demand Planning system.

This provides a simplified web interface for generating forecasts for parent SKUs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Parent SKU Demand Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if the API server is running."""
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

def get_parent_summary(parent_sku: str) -> Optional[Dict]:
    """Get summary data for a parent SKU."""
    try:
        response = requests.get(f"{API_BASE_URL}/parent-sku/{parent_sku}/summary")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Summary API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error getting summary data: {e}")
    return None

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

def create_metrics_dashboard(summary_data: Dict, series_data: Dict, forecast_data: Dict) -> None:
    """Create a metrics dashboard."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use total from summary (all time) vs recent 52 weeks
        total_historical_units = summary_data["total_units"]
        recent_units = sum(series_data["units"])
        st.metric(
            label="Total Historical Units",
            value=f"{total_historical_units:,.0f}",
            delta=f"Recent 52w: {recent_units:,.0f}"
        )
    
    with col2:
        total_historical_sales = summary_data["total_sales"]
        recent_sales = sum(series_data["net_sales"])
        st.metric(
            label="Total Historical Sales",
            value=f"${total_historical_sales:,.0f}",
            delta=f"Recent 52w: ${recent_sales:,.0f}"
        )
    
    with col3:
        avg_forecast_units = np.mean(forecast_data["forecast_units"])
        st.metric(
            label="Avg Forecast Units/Week",
            value=f"{avg_forecast_units:,.0f}"
        )
    
    with col4:
        total_forecast_sales = sum(forecast_data["forecast_sales"])
        st.metric(
            label="Total Forecast Sales",
            value=f"${total_forecast_sales:,.0f}"
        )

def main():
    """Main Streamlit application."""
    
    # Initialize session state for SKU selection
    if 'selected_sku_index' not in st.session_state:
        st.session_state.selected_sku_index = 0
    
    # Header
    st.markdown('<h1 class="main-header">📊 Parent SKU Demand Planner</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API server is not running. Please start the API server first:")
        st.code("uv run uvicorn src.dp.api_parent:app --reload --port 8000")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Get available data
    with st.spinner("Loading available parent SKUs..."):
        available_parent_skus = get_available_parent_skus()
    
    if not available_parent_skus:
        st.error("No parent SKUs found. Please ensure data has been processed.")
        st.stop()
    
    # Parent SKU Selection
    st.sidebar.markdown("### 🔍 Parent SKU Selection")
    st.sidebar.markdown("**Format**: `100275` (Parent SKU only)")
    st.sidebar.markdown("**The system will aggregate all child SKUs under this parent.**")
    
    # Parent SKU input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["📝 Manual Entry", "📋 Select from List"],
        horizontal=True
    )
    
    if input_method == "📝 Manual Entry":
        # Manual text input
        selected_parent_sku = st.sidebar.text_input(
            "Enter Parent SKU",
            value=st.session_state.get("manual_sku", ""),
            placeholder="e.g., 100275",
            help="Enter a parent SKU manually (6 digits)",
            key="manual_sku_input"
        )
        
        # Store in session state
        if selected_parent_sku:
            st.session_state.manual_sku = selected_parent_sku
            
        # Validate format
        if selected_parent_sku and not selected_parent_sku.isdigit():
            st.sidebar.warning("⚠️ Parent SKU should be numeric (e.g., 100275)")
        elif selected_parent_sku and len(selected_parent_sku) != 6:
            st.sidebar.warning("⚠️ Parent SKU should be 6 digits (e.g., 100275)")
        
        # Show popular SKUs for quick selection
        if not selected_parent_sku:
            st.sidebar.markdown("**💡 Popular Parent SKUs:**")
            popular_skus = available_parent_skus[:5]  # Show first 5
            for sku in popular_skus:
                if st.sidebar.button(f"📌 {sku}", key=f"quick_select_{sku}"):
                    st.session_state.manual_sku = sku
                    st.rerun()
            
    else:
        # Dropdown selection
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
        
        # Show quick stats if available
        if selected_parent_sku in available_parent_skus:
            st.sidebar.info("✅ This SKU has historical data available")
        else:
            st.sidebar.warning("⚠️ This SKU may not have historical data")
    
    # Forecast parameters
    st.sidebar.markdown("### ⚙️ Forecast Parameters")
    
    horizon = st.sidebar.slider(
        "Forecast Horizon (weeks)",
        min_value=1,
        max_value=52,
        value=13,
        help="Number of weeks to forecast into the future"
    )
    
    model = st.sidebar.selectbox(
        "Forecast Model",
        ["auto", "exponential_smoothing", "random_forest", "gradient_boosting", "moving_average", "arima", "seasonal_naive"],
        help="Choose the forecasting model to use",
        format_func=lambda x: {
            "auto": "🤖 Auto (Best for your data)",
            "exponential_smoothing": "📈 Holt-Winters (Seasonal + Trend)",
            "random_forest": "🌲 Random Forest (ML)",
            "gradient_boosting": "⚡ Gradient Boosting (ML)",
            "moving_average": "📊 Moving Average (Simple)",
            "arima": "📉 ARIMA (Statistical)",
            "seasonal_naive": "🔄 Seasonal Naive (Copy-paste)"
        }.get(x, x)
    )
    
    # Model descriptions
    st.sidebar.markdown("### 📚 Model Descriptions")
    model_descriptions = {
        "auto": "**Auto**: Automatically selects the best model for your data",
        "exponential_smoothing": "**Holt-Winters**: Learns seasonal patterns and trends - **Recommended for seasonal data**",
        "random_forest": "**Random Forest**: Machine learning model that learns complex patterns",
        "gradient_boosting": "**Gradient Boosting**: Advanced ML model with high accuracy",
        "moving_average": "**Moving Average**: Simple baseline using recent averages",
        "arima": "**ARIMA**: Statistical model for time series with trends",
        "seasonal_naive": "**Seasonal Naive**: Uses same week from previous year"
    }
    
    if model in model_descriptions:
        st.sidebar.info(model_descriptions[model])
    
    # Main content
    if st.sidebar.button("🚀 Generate Forecast", type="primary"):
        
        with st.spinner("Loading data..."):
            # Get summary data (all time totals)
            summary_data = get_parent_summary(selected_parent_sku)
            
            # Get historical data (recent 52 weeks for chart)
            series_data = get_parent_series_data(
                selected_parent_sku, 
                limit=52
            )
        
        if not summary_data or not series_data:
            st.error(f"No data found for Parent SKU: {selected_parent_sku}")
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
        st.success(f"✅ Forecast generated successfully using {forecast_data['model_used']} model!")
        
        # Metrics dashboard
        st.markdown("### 📊 Key Metrics")
        create_metrics_dashboard(summary_data, series_data, forecast_data)
        
        # Forecast chart
        st.markdown("### 📈 Forecast Visualization")
        fig = create_parent_forecast_chart(series_data, forecast_data)
        st.plotly_chart(fig, width='stretch')
        
        # Forecast table
        st.markdown("### 📋 Forecast Details")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Period': forecast_data['forecast_periods'],
            'Forecast Units': forecast_data['forecast_units'],
            'Forecast Sales ($)': forecast_data['forecast_sales']
        })
        
        st.dataframe(forecast_df, width='stretch')
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"forecast_{selected_parent_sku}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
