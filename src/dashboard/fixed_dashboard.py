import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Dashboard setup
st.set_page_config(page_title="Financial Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Market Dashboard")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/forecasts", exist_ok=True)

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

# Get next trading day
def get_next_trading_day(market_name, from_date=None):
    if from_date is None:
        from_date = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return None
    
    # Start from tomorrow
    next_day = from_date + timedelta(days=1)
    
    # Find next trading day
    while next_day.weekday() not in market["trading_days"]:
        next_day = next_day + timedelta(days=1)
    
    # Set opening time
    next_opening = datetime.combine(next_day.date(), market["opening_time"])
    return next_opening

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

# Create features for model
def create_features(data, price_col):
    features = pd.DataFrame(index=data.index)
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Market open feature
    if 'MarketOpen' in data.columns:
        features['market_open'] = data['MarketOpen'].astype(int)
    
    # Technical indicators
    for window in [3, 6, 12, 24]:
        features[f'ma_{window}'] = data[price_col].rolling(window=window).mean()
        features[f'std_{window}'] = data[price_col].rolling(window=window).std()
    
    features['momentum'] = data[price_col].diff(periods=1)
    features['momentum_6h'] = data[price_col].diff(periods=6)
    
    # Lag features
    for i in range(1, 25):
        features[f'lag_{i}'] = data[price_col].shift(i)
    
    # Fill missing values
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# Generate forecasts
def make_forecasts(data, price_col):
    try:
        if data is None or len(data) < 24:
            return None
        
        # Create features
        features = create_features(data, price_col)
        feature_names = features.columns.tolist()
        
        # Ensure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(0)  # Replace NaNs with zeros
        
        # Train model
        X = features.values
        y = data[price_col].values
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Setup for forecast
        last_date = data.index[-1]
        forecast_hours = 14 * 24  # 14 days
        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=forecast_hours, freq='1H')
        
        # Make rolling forecast
        forecast_values = []
        current_data = data.copy()
        
        for i in range(forecast_hours):
            # Generate features
            current_features = create_features(current_data, price_col)
            
            # Ensure feature consistency
            for feat in feature_names:
                if feat not in current_features.columns:
                    current_features[feat] = 0
            
            current_features = current_features[feature_names]
            
            # Convert any remaining NaNs to 0
            current_features = current_features.fillna(0)
            
            # Make prediction
            current_X = scaler.transform(current_features.values[-1:])
            prediction = model.predict(current_X)[0]
            forecast_values.append(prediction)
            
            # Update for next iteration
            if i < forecast_hours - 1:
                new_date = future_dates[i]
                new_row = pd.DataFrame({
                    price_col: [prediction], 
                    'MarketOpen': [is_market_open(data.name, new_date)]
                }, index=[new_date])
                current_data = pd.concat([current_data, new_row])
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            price_col: forecast_values, 
            'MarketOpen': [is_market_open(data.name, dt) for dt in future_dates]
        }, index=future_dates)
        
        return forecast_df, model, feature_names
    
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        return None

# Generate market plot
def generate_market_plot(data, forecast, market_name):
    if data is None:
        return None
    
    fig = go.Figure()
    
    # Current time
    current_time = datetime.now()
    
    # Historical data (last 14 days)
    historical_start = current_time - timedelta(days=14)
    historical_data = data[data.index >= historical_start]
    
    # Add historical line
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
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
                y=[historical_data.loc[open_time, 'Close'] for open_time in opens],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Market Open'
            ))
        
        if closes:
            fig.add_trace(go.Scatter(
                x=closes,
                y=[historical_data.loc[close_time, 'Close'] for close_time in closes],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Market Close'
            ))
    
    # Add forecast if available
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['Close'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Add market open/close markers for forecast
        if market_name != "USD_EUR":
            forecast_opens = []
            forecast_closes = []
            
            for j in range(1, len(forecast)):
                if forecast['MarketOpen'].iloc[j] and not forecast['MarketOpen'].iloc[j-1]:
                    forecast_opens.append(forecast.index[j])
                elif not forecast['MarketOpen'].iloc[j] and forecast['MarketOpen'].iloc[j-1]:
                    forecast_closes.append(forecast.index[j])
            
            if forecast_opens:
                fig.add_trace(go.Scatter(
                    x=forecast_opens,
                    y=[forecast.loc[open_time, 'Close'] for open_time in forecast_opens],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green', opacity=0.5),
                    name='Forecast Market Open'
                ))
            
            if forecast_closes:
                fig.add_trace(go.Scatter(
                    x=forecast_closes,
                    y=[forecast.loc[close_time, 'Close'] for close_time in forecast_closes],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red', opacity=0.5),
                    name='Forecast Market Close'
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
    
    # Set view range to 14 days before and after
    view_start = (current_time - timedelta(days=14)).strftime('%Y-%m-%d')
    view_end = (current_time + timedelta(days=14)).strftime('%Y-%m-%d')
    
    fig.update_layout(
        title=f"{market_name} - Historical Data and Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(range=[view_start, view_end]),
        hovermode="x unified"
    )
    
    return fig

# Generate combined plot
def generate_combined_plot(datasets, forecasts):
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
        
        forecast = forecasts.get(market_name)
        
        # Historical data (last 14 days)
        historical_start = current_time - timedelta(days=14)
        historical_data = data[data.index >= historical_start]
        
        # Add historical line
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name=f'{market_name} Historical',
            line=dict(color='blue', width=2),
            legendgroup=market_name
        ), row=i, col=1)
        
        # Add forecast if available
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['Close'],
                mode='lines',
                name=f'{market_name} Forecast',
                line=dict(color='blue', width=2, dash='dash'),
                legendgroup=market_name
            ), row=i, col=1)
    
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
    view_end = (current_time + timedelta(days=14)).strftime('%Y-%m-%d')
    
    fig.update_layout(
        title="Financial Markets - 14-Day Historical Data and 14-Day Forecast",
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
    
    st.sidebar.markdown(f"**{market_name}:** {status}")
    
    if not is_open:
        next_open = get_next_trading_day(market_name)
        if next_open:
            st.sidebar.markdown(f"Next opening: {next_open.strftime('%Y-%m-%d %H:%M')}")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Fetch market data
with st.spinner("Fetching market data..."):
    market_data = {}
    market_forecasts = {}
    
    for market_name in MARKETS.keys():
        # Get historical data
        if data_source == "Live Data":
            data = fetch_historical_data(market_name)
            if data is None or data.empty:
                st.sidebar.warning(f"Could not fetch live data for {market_name}. Using demo data.")
                data = create_demo_data(market_name)
        else:
            data = create_demo_data(market_name)
        
        # Set market name attribute
        data.name = market_name
        market_data[market_name] = data
        
        # Generate forecast
        forecast_result = make_forecasts(data, 'Close')
        
        if forecast_result:
            forecast_df, model, feature_names = forecast_result
            market_forecasts[market_name] = forecast_df
            
            # Store feature importance
            if model is not None and hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                market_name_lower = market_name.lower()
                if not hasattr(st.session_state, f"{market_name_lower}_importance"):
                    setattr(st.session_state, f"{market_name_lower}_importance", importance_df)

# Create combined visualization
st.subheader("Market Visualization")
combined_fig = generate_combined_plot(market_data, market_forecasts)
st.plotly_chart(combined_fig, use_container_width=True)

# Create individual market tabs
st.subheader("Detailed Market Analysis")
market_tabs = st.tabs(list(MARKETS.keys()))

for i, market_name in enumerate(MARKETS.keys()):
    with market_tabs[i]:
        col1, col2 = st.columns([2, 1])
        
        data = market_data.get(market_name)
        forecast = market_forecasts.get(market_name)
        
        with col1:
            if data is not None:
                market_fig = generate_market_plot(data, forecast, market_name)
                st.plotly_chart(market_fig, use_container_width=True)
        
        with col2:
            if data is not None:
                current_price = float(data['Close'].iloc[-1])
                st.metric("Current Price", f"{current_price:.2f}")
                
                # Market status
                is_open = is_market_open(market_name)
                st.metric("Market Status", "OPEN" if is_open else "CLOSED")
                
                # Next opening
                if not is_open:
                    next_open = get_next_trading_day(market_name)
                    if next_open:
                        st.write(f"Next opening: {next_open.strftime('%Y-%m-%d %H:%M')}")
                
                # Forecast for next opening
                if forecast is not None and market_name != "USD_EUR":
                    # Find next market opening in forecast
                    next_open_idx = None
                    
                    for j in range(1, len(forecast)):
                        if forecast['MarketOpen'].iloc[j] and not forecast['MarketOpen'].iloc[j-1]:
                            next_open_idx = j
                            break
                    
                    if next_open_idx is not None:
                        next_open_time = forecast.index[next_open_idx]
                        next_open_price = float(forecast['Close'].iloc[next_open_idx])
                        next_open_change = ((next_open_price - current_price) / current_price) * 100
                        
                        st.write(f"**Next Market Open:** {next_open_time.strftime('%Y-%m-%d %H:%M')}")
                        st.metric("Predicted Opening", f"{next_open_price:.2f}", f"{next_open_change:+.2f}%")

# Show model insights
st.subheader("Model Insights")

for market_name in MARKETS.keys():
    market_name_lower = market_name.lower()
    importance_df = getattr(st.session_state, f"{market_name_lower}_importance", None)
    
    if importance_df is not None:
        st.write(f"**{market_name} - Top 5 Factors**")
        st.dataframe(importance_df.head(5))

# News simulation and impact (simplified)
st.subheader("Market News and Impact")

# Simulate some news
sample_news = [
    {"date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), 
     "headline": "Central bank signals continued support for economy", 
     "impact": "Positive"},
    {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), 
     "headline": "Inflation concerns weigh on market sentiment", 
     "impact": "Negative"},
    {"date": datetime.now().strftime("%Y-%m-%d"), 
     "headline": "Strong economic data boosts market optimism", 
     "impact": "Positive"},
]

for news in sample_news:
    impact_color = "green" if news["impact"] == "Positive" else "red"
    st.markdown(f"**{news['date']}**: {news['headline']} - <span style='color:{impact_color}'>{news['impact']}</span>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("This dashboard provides 14-day historical data and 14-day forecasts for major financial indices.")
st.write("Market opening and closing times are marked, and the dashboard shows predictions for the next market opening.")
