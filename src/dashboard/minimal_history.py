import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Dashboard setup
st.set_page_config(page_title="Market History", page_icon="📈")
st.title("📈 Financial Markets - 14-Day History")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Market configuration
markets = {
    "DAX": "^GDAXI",
    "DowJones": "^DJI",
    "USD_EUR": "EURUSD=X"
}

# Create demo data
def create_demo_data(market_name):
    # Base values
    if market_name == "DAX":
        base_value = 18500
        volatility = 100
    elif market_name == "DowJones":
        base_value = 39000
        volatility = 200
    else:  # USD_EUR
        base_value = 0.92
        volatility = 0.005
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate prices
    np.random.seed(42 + hash(market_name) % 100)
    prices = []
    price = base_value
    
    for date in date_range:
        # Simple random walk
        change = np.random.normal(0, volatility/1000)
        price *= (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'Close': prices
    }, index=date_range)
    
    return df

# Fetch data
def fetch_data(market_name, ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return create_demo_data(market_name)
        
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data
    except:
        return create_demo_data(market_name)

# Plot single market
def plot_market(data, market_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=market_name
    ))
    
    # Add current time line
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.add_shape(
        type="line",
        x0=current_time_str,
        y0=0,
        x1=current_time_str,
        y1=1,
        line=dict(color="green", width=2),
        xref="x",
        yref="paper"
    )
    
    fig.update_layout(
        title=f"{market_name} - Last 14 Days",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    return fig

# Data source selection
use_live_data = st.sidebar.checkbox("Use Live Data", value=True)

# Get data for each market
market_data = {}

for market_name, ticker in markets.items():
    if use_live_data:
        data = fetch_data(market_name, ticker)
    else:
        data = create_demo_data(market_name)
    
    market_data[market_name] = data

# Display each market in tabs
market_tabs = st.tabs(list(markets.keys()))

for i, market_name in enumerate(markets.keys()):
    with market_tabs[i]:
        data = market_data[market_name]
        
        # Display chart
        fig = plot_market(data, market_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current price
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"{current_price:.2f}")
        
        # Display data table
        st.subheader("Historical Data")
        st.dataframe(data[['Close']].tail(24))

# Footer
st.markdown("---")
st.write("This dashboard shows 14 days of historical price data for major financial indices.")
