import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
import traceback

# Dashboard setup
st.set_page_config(page_title="DAX History", page_icon="📈")
st.title("📈 DAX - 14-Day History")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Create demo data for DAX
def create_demo_data():
    # Base values for DAX
    base_value = 18500
    volatility = 100
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate prices
    np.random.seed(42)
    prices = []
    price = base_value
    
    for date in date_range:
        # Simple random walk
        change = np.random.normal(0, volatility/1000)
        price *= (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.002 for p in prices],
        'Low': [p * 0.998 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, size=len(date_range))
    }, index=date_range)
    
    return df

# Fetch DAX data
def fetch_dax_data():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=16)  # Get extra days to ensure we have 14 days
        
        st.sidebar.info("Fetching DAX data from Yahoo Finance...")
        
        # Try different ticker symbols for DAX
        for ticker in ["^GDAXI", "DAX", "DAX.DE"]:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) > 5:
                    st.sidebar.success(f"Successfully fetched data using ticker: {ticker}")
                    
                    # Make index timezone-naive
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Ensure we have at least 14 days of data
                    if (data.index[-1] - data.index[0]).days >= 14:
                        return data
                    else:
                        st.sidebar.warning(f"Not enough days in data: {(data.index[-1] - data.index[0]).days} days")
            except Exception as e:
                st.sidebar.error(f"Error with ticker {ticker}: {str(e)}")
                continue
        
        # If we get here, we couldn't fetch valid data
        st.sidebar.warning("Could not fetch valid DAX data. Using demo data.")
        return None
        
    except Exception as e:
        st.sidebar.error(f"Error fetching DAX data: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return None

# Plot DAX data
def plot_dax(data):
    fig = go.Figure()
    
    # Add OHLC candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='DAX'
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
        title="DAX - Last 14 Days",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    return fig

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Get DAX data
if data_source == "Live Data":
    dax_data = fetch_dax_data()
    if dax_data is None or dax_data.empty:
        st.warning("Could not fetch live DAX data. Using demo data instead.")
        dax_data = create_demo_data()
else:
    dax_data = create_demo_data()

# Display DAX data
st.subheader("DAX Market Data")

# Display detailed data information
if dax_data is not None and not dax_data.empty:
    st.write(f"Data range: {dax_data.index[0]} to {dax_data.index[-1]}")
    st.write(f"Number of data points: {len(dax_data)}")
    
    # Ensure we have the last 14 days
    days_in_data = (dax_data.index[-1] - dax_data.index[0]).days
    st.write(f"Days covered: {days_in_data}")
    
    # Try to get current price safely
    try:
        if 'Close' in dax_data.columns and not dax_data['Close'].empty:
            current_price = dax_data['Close'].iloc[-1]
            if pd.notna(current_price):  # Check if not NaN
                st.write(f"Current DAX value: {current_price:.2f}")
            else:
                st.write("Current DAX value: Not available (NaN)")
        else:
            st.write("Close price column not found or empty")
    except Exception as e:
        st.error(f"Error displaying current price: {str(e)}")
        st.write("Data columns available:", dax_data.columns.tolist())
    
    # Plot data
    try:
        fig = plot_dax(dax_data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        st.error(traceback.format_exc())
    
    # Show recent data
    st.subheader("Recent Data")
    st.dataframe(dax_data.tail(10))
else:
    st.error("No DAX data available to display.")

# Footer
st.markdown("---")
st.write("This dashboard shows 14 days of historical price data for the DAX (German Stock Index).")
