import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Dashboard setup
st.set_page_config(page_title="Market History Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Market Historical Data")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Market information
MARKETS = {
    "DAX": {
        "ticker": "^GDAXI",
        "description": "German Stock Index",
        "opening_time": time(9, 0),
        "closing_time": time(17, 30),
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
    },
    "DowJones": {
        "ticker": "^DJI",
        "description": "Dow Jones Industrial Average",
        "opening_time": time(9, 30),
        "closing_time": time(16, 0),
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
    },
    "USD_EUR": {
        "ticker": "EURUSD=X",
        "description": "Euro to US Dollar Exchange Rate",
        "opening_time": time(0, 0),  # Forex markets run 24 hours
        "closing_time": time(23, 59),
        "trading_days": [0, 1, 2, 3, 4, 6],  # Monday to Friday + Sunday
    }
}

# Check if market is open
def is_market_open(market_name, check_time=None):
    if check_time is None:
        check_time = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return False
    
    # Check if it's a trading day
    if check_time.weekday() not in market["trading_days"]:
        return False
    
    # Check trading hours (except for forex which is 24/7 on trading days)
    if market_name == "USD_EUR":
        return True
    
    current_time = check_time.time()
    return market["opening_time"] <= current_time <= market["closing_time"]

# Create demo data
def create_demo_data(market_name, days_back=14):
    # Set base values
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
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate price data
    np.random.seed(42 + hash(market_name) % 100)
    price_data = []
    current_price = base_value
    
    for date in date_range:
        # Time and day effects
        hour = date.hour
        day = date.weekday()
        
        # Market hours effect
        market_open = is_market_open(market_name, date)
        hour_volatility = volatility if market_open else volatility * 0.3
        
        # Day effect
        day_factor = 1.0 if day <= 4 else 0.3
        day_trend = 0.0001 * (2 - day) if day <= 4 else 0
        
        # Price change components
        random_component = np.random.normal(0, hour_volatility * day_factor / 1000)
        price_change = day_trend + random_component
        current_price *= (1 + price_change)
        
        price_data.append(current_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': price_data,
        'High': [p * (1 + np.random.uniform(0.001, 0.003)) for p in price_data],
        'Low': [p * (1 - np.random.uniform(0.001, 0.003)) for p in price_data],
        'Close': [p * (1 + np.random.normal(0, 0.001)) for p in price_data],
        'Volume': np.random.randint(1000, 10000, size=len(date_range))
    }, index=date_range)
    
    # Mark market open/close
    df['MarketOpen'] = [is_market_open(market_name, dt) for dt in df.index]
    
    return df

# Fetch historical data
def fetch_historical_data(market_name, days_back=15):
    try:
        ticker = MARKETS.get(market_name, {}).get("ticker")
        if not ticker:
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try hourly data
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
        
        if data.empty or len(data) < 24:
            # Fallback to daily data
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty:
                # Convert to hourly
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            # Make index timezone-naive
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Mark market status
            data['MarketOpen'] = [is_market_open(market_name, dt) for dt in data.index]
            return data
        
        return None
    except Exception as e:
        st.error(f"Error fetching {market_name} data: {str(e)}")
        return None

# Generate individual market plot
def generate_market_plot(data, market_name):
    if data is None:
        return None
    
    fig = go.Figure()
    
    # Current time
    current_time = datetime.now()
    
    # Historical data (last 14 days)
    historical_start = current_time - timedelta(days=14)
    historical_data = data[data.index >= historical_start]
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=historical_data.index,
        open=historical_data['Open'],
        high=historical_data['High'],
        low=historical_data['Low'],
        close=historical_data['Close'],
        name='Price Data',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add market open/close markers
    if market_name != "USD_EUR":  # Skip for forex which is always open
        opens = []
        closes = []
        
        # Find market openings and closings
        for j in range(1, len(historical_data)):
            if historical_data['MarketOpen'].iloc[j] and not historical_data['MarketOpen'].iloc[j-1]:
                opens.append(historical_data.index[j])
            elif not historical_data['MarketOpen'].iloc[j] and historical_data['MarketOpen'].iloc[j-1]:
                closes.append(historical_data.index[j])
        
        # Add markers
        if opens:
            fig.add_trace(go.Scatter(
                x=opens,
                y=[historical_data.loc[open_time, 'High'] * 1.005 for open_time in opens],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Market Open'
            ))
        
        if closes:
            fig.add_trace(go.Scatter(
                x=closes,
                y=[historical_data.loc[close_time, 'Low'] * 0.995 for close_time in closes],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Market Close'
            ))
    
    # Add vertical line at current time
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.add_shape(
        type="line",
        x0=current_time_str,
        y0=0,
        x1=current_time_str,
        y1=1,
        line=dict(color="green", width=2, dash="solid"),
        xref="x",
        yref="paper"
    )
    
    fig.add_annotation(
        x=current_time_str,
        y=1,
        text="Current",
        showarrow=False,
        xref="x",
        yref="paper",
        yanchor="bottom",
        font=dict(color="green", size=12)
    )
    
    # Set view range to 14 days
    view_start = (current_time - timedelta(days=14)).strftime('%Y-%m-%d')
    view_end = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.update_layout(
        title=f"{market_name} - Historical Market Data (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(range=[view_start, view_end]),
        hovermode="x unified"
    )
    
    return fig

# Generate combined plot
def generate_combined_plot(datasets):
    fig = make_subplots(rows=len(datasets), cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.05,
                       subplot_titles=list(datasets.keys()))
    
    # Current time
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Process each market
    for i, (market_name, data) in enumerate(datasets.items(), 1):
        if data is None:
            continue
        
        # Historical data (last 14 days)
        historical_start = current_time - timedelta(days=14)
        historical_data = data[data.index >= historical_start]
        
        # Add line chart
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name=f'{market_name}',
            line=dict(color='blue', width=2),
            legendgroup=market_name
        ), row=i, col=1)
        
        # Add market open/close shading
        if market_name != "USD_EUR":  # Skip for forex which is always open
            # Find periods when market is open
            open_periods = []
            start_open = None
            
            for j in range(len(historical_data)):
                if historical_data['MarketOpen'].iloc[j]:
                    if start_open is None:
                        start_open = historical_data.index[j]
                else:
                    if start_open is not None:
                        end_open = historical_data.index[j-1]
                        open_periods.append((start_open, end_open))
                        start_open = None
            
            # If still open at the end
            if start_open is not None:
                open_periods.append((start_open, historical_data.index[-1]))
            
            # Add shading for open periods
            for start, end in open_periods:
                fig.add_shape(
                    type="rect",
                    x0=start,
                    y0=0,
                    x1=end,
                    y1=1,
                    line=dict(width=0),
                    fillcolor="green",
                    opacity=0.1,
                    layer="below",
                    xref=f"x{i}" if i > 1 else "x",
                    yref=f"y{i} domain"
                )
    
        # Add current time line
        fig.add_shape(
            type="line",
            x0=current_time_str,
            y0=0,
            x1=current_time_str,
            y1=1,
            line=dict(color="green", width=2),
            xref=f"x{i}" if i > 1 else "x",
            yref="paper"
        )
    
    # Set view range
    view_start = (current_time - timedelta(days=14)).strftime('%Y-%m-%d')
    view_end = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.update_layout(
        title="Financial Markets - Historical Data (Last 14 Days)",
        height=200 * len(datasets),
        xaxis=dict(range=[view_start, view_end]),
        hovermode="x unified"
    )
    
    return fig

# Display market status
st.sidebar.header("Market Status")

for market_name, market_info in MARKETS.items():
    is_open = is_market_open(market_name)
    status = "OPEN" if is_open else "CLOSED"
    status_color = "green" if is_open else "red"
    
    st.sidebar.markdown(f"**{market_name}:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Fetch market data
with st.spinner("Fetching market data..."):
    market_data = {}
    
    for market_name in MARKETS.keys():
        # Get historical data
        if data_source == "Live Data":
            data = fetch_historical_data(market_name)
            if data is None or data.empty:
                st.sidebar.warning(f"Could not fetch live data for {market_name}. Using demo data.")
                data = create_demo_data(market_name)
        else:
            data = create_demo_data(market_name)
        
        market_data[market_name] = data

# Create combined visualization
st.subheader("Combined Market View")
combined_fig = generate_combined_plot(market_data)
st.plotly_chart(combined_fig, use_container_width=True)

# Create individual market tabs
st.subheader("Detailed Market View")
market_tabs = st.tabs(list(MARKETS.keys()))

for i, market_name in enumerate(MARKETS.keys()):
    with market_tabs[i]:
        data = market_data.get(market_name)
        
        if data is not None:
            # Show candlestick chart
            market_fig = generate_market_plot(data, market_name)
            st.plotly_chart(market_fig, use_container_width=True)
            
            # Show current price and stats
            current_price = float(data['Close'].iloc[-1])
            open_price = float(data['Close'].iloc[-24]) if len(data) >= 24 else None
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
            
            with col2:
                # 24-hour change
                if open_price:
                    change_24h = ((current_price - open_price) / open_price) * 100
                    st.metric("24-Hour Change", f"{change_24h:+.2f}%")
            
            with col3:
                # Market status
                is_open = is_market_open(market_name)
                st.metric("Market Status", "OPEN" if is_open else "CLOSED")
            
            # Show data table
            with st.expander("Show Historical Data Table"):
                # Get last 14 days of data
                start_date = datetime.now() - timedelta(days=14)
                display_data = data[data.index >= start_date].copy()
                
                # Format the data for display
                display_data = display_data.reset_index()
                display_data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Open']
                display_data['Datetime'] = display_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(display_data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Market Open']])

# Footer
st.markdown("---")
st.write("This dashboard displays the last 14 days of historical data for major financial indices.")
st.write("Market opening and closing times are marked, and the dashboard shows the current market status.")
