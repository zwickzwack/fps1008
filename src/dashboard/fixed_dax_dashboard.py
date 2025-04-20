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
        ticker = "^GDAXI"
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
        if not data.empty and len(data) > 5:
            st.sidebar.success(f"Successfully fetched data using ticker: {ticker}")
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                st.sidebar.info("MultiIndex columns detected")
                # Create a flattened version of the data with simple column names
                flat_data = pd.DataFrame(index=data.index)
                
                # Extract data for our ticker
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns.get_level_values(0):
                        flat_data[col] = data[col, ticker] if (col, ticker) in data.columns else data[col][0]
                data = flat_data
            
            # Make index timezone-naive
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
        else:
            st.sidebar.warning("Could not fetch valid DAX data. Using demo data.")
            return None
        
    except Exception as e:
        st.sidebar.error(f"Error fetching DAX data: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return None

# Plot DAX data
def plot_dax(data):
    fig = go.Figure()
    
    # Add line chart for Close price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='DAX Close'
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

# Display DAX data structure
st.subheader("Data Structure Information")
st.write("Column Names:", dax_data.columns.tolist())
st.write("Data Shape:", dax_data.shape)
st.write("Index Type:", type(dax_data.index).__name__)
if isinstance(dax_data.columns, pd.MultiIndex):
    st.write("Column Levels:", dax_data.columns.levels)

# Display DAX data
st.subheader("DAX Market Data")

# Only keep data from the last 14 days
start_date = current_time - timedelta(days=14)
dax_data = dax_data[dax_data.index >= start_date]

# Display detailed data information
if dax_data is not None and not dax_data.empty:
    st.write(f"Data range: {dax_data.index[0]} to {dax_data.index[-1]}")
    st.write(f"Number of data points: {len(dax_data)}")
    
    # Get days covered
    days_in_data = (dax_data.index[-1] - dax_data.index[0]).days
    st.write(f"Days covered: {days_in_data}")
    
    # Show the raw data for the last entry
    st.write("Last data entry:")
    st.write(dax_data.iloc[-1])
    
    # Get current price
    try:
        if 'Close' in dax_data.columns:
            # Access the last value safely
            current_price = dax_data['Close'].iloc[-1]
            st.metric("Current DAX Value", f"{current_price:.2f}")
        else:
            st.error("Close column not found. Available columns: " + ", ".join(dax_data.columns))
    except Exception as e:
        st.error(f"Error displaying current price: {str(e)}")
        st.error(traceback.format_exc())
    
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
