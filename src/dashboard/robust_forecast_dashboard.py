import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import traceback
import yfinance as yf
import requests

# Dashboard title and setup
st.set_page_config(page_title="Financial Index Forecast", page_icon="📈", layout="wide")
st.title("📈 Financial Index Multi-Horizon Forecast Dashboard")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"Current User: zwickzwack")

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/forecasts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Index configuration
index_options = {
    "DAX": {
        "ticker": "^GDAXI",
        "alt_ticker": "DAX",
        "description": "German Stock Index",
        "default_value": 18500,
        "volatility": 100
    },
    "DowJones": {
        "ticker": "^DJI",
        "alt_ticker": "DJI",
        "description": "Dow Jones Industrial Average",
        "default_value": 39000,
        "volatility": 200
    },
    "USD_EUR": {
        "ticker": "EURUSD=X",
        "alt_ticker": "EUR=X",
        "description": "Euro to US Dollar Exchange Rate",
        "default_value": 0.92,
        "volatility": 0.005
    }
}

# Select index
selected_index = st.sidebar.selectbox(
    "Select Financial Index:",
    options=list(index_options.keys())
)

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Function to fetch real-time data from Yahoo Finance with retries
def fetch_live_data(index_name, days_back=20):  # Request more days to ensure we get 14 days
    index_info = index_options.get(index_name, {})
    ticker = index_info.get("ticker")
    alt_ticker = index_info.get("alt_ticker")
    
    if not ticker:
        st.sidebar.error(f"No ticker configured for {index_name}")
        return None
    
    data = None
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Status update
            st.sidebar.info(f"Attempt {attempt+1}/{max_retries}: Fetching data for {index_name} ({ticker})...")
            
            # Get data from Yahoo Finance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Try with primary ticker, hourly data
            data = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
            
            if data.empty or len(data) < 24:
                st.sidebar.warning(f"Limited hourly data for {ticker}. Trying daily data...")
                # Try daily data
                data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                
                if not data.empty:
                    # Convert daily to hourly
                    new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                    data = data.reindex(new_index, method='ffill')
            
            # If still no data, try alternate ticker
            if (data.empty or len(data) < 24) and alt_ticker:
                st.sidebar.warning(f"Trying alternate ticker: {alt_ticker}...")
                data = yf.download(alt_ticker, start=start_date, end=end_date, interval="1h", progress=False)
                
                if data.empty or len(data) < 24:
                    # Try daily data with alternate ticker
                    data = yf.download(alt_ticker, start=start_date, end=end_date, interval="1d", progress=False)
                    
                    if not data.empty:
                        # Convert daily to hourly
                        new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                        data = data.reindex(new_index, method='ffill')
            
            # Check if we got enough data
            if not data.empty and len(data) >= 24:
                # Make index timezone-naive
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                st.sidebar.success(f"Successfully loaded {len(data)} data points")
                break  # Success, exit retry loop
            else:
                st.sidebar.error(f"Attempt {attempt+1} failed: Not enough data")
                time.sleep(1)  # Wait before retry
                
        except Exception as e:
            error_msg = str(e)
            st.sidebar.error(f"Attempt {attempt+1} failed: {error_msg}")
            
            # Log the error with traceback
            with open("logs/data_errors.log", "a") as f:
                f.write(f"{datetime.now()}: Error fetching {index_name}: {error_msg}\n")
                f.write(traceback.format_exc() + "\n\n")
            
            time.sleep(1)  # Wait before retry
    
    # If we didn't get data after all retries
    if data is None or data.empty or len(data) < 24:
        st.sidebar.error(f"Failed to fetch data for {index_name} after {max_retries} attempts")
        return None
        
    return data

# Function to create realistic demo data
def create_demo_data(index_name, days_back=14):
    st.sidebar.info(f"Creating demo data for {index_name}")
    
    # Get parameters from configuration
    index_info = index_options.get(index_name, {})
    base_value = index_info.get("default_value", 1000)
    volatility = index_info.get("volatility", 10)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate price data with realistic patterns
    np.random.seed(42 + hash(index_name) % 100)  # Different seed for each index but consistent
    price_data = []
    current_price = base_value
    
    # Create a trend component (up or down)
    trend_direction = np.random.choice([-1, 1])
    trend_strength = np.random.uniform(0.0001, 0.0005)
    
    for i, date in enumerate(date_range):
        # Time of day effect
        hour = date.hour
        # Higher volatility during market hours
        hour_volatility = volatility if 9 <= hour <= 16 else volatility * 0.4
        
        # Day of week effect
        day = date.dayofweek
        # Weekday vs weekend effect
        if day <= 4:  # Weekday
            day_volatility = 1.0
            # Monday more positive, Friday more negative
            day_trend = 0.0002 * (2 - day) 
        else:  # Weekend
            day_volatility = 0.3
            day_trend = 0
        
        # Add trend, cyclical, day and random components
        trend_component = trend_direction * trend_strength * i
        day_component = day_trend
        random_component = np.random.normal(0, hour_volatility * day_volatility / 1000)
        
        # Calculate price change
        price_change = trend_component + day_component + random_component
        current_price *= (1 + price_change)
        
        price_data.append(current_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': price_data,
        'High': [p * (1 + np.random.uniform(0.001, 0.005)) for p in price_data],
        'Low': [p * (1 - np.random.uniform(0.001, 0.005)) for p in price_data],
        'Close': [p * (1 + np.random.normal(0, 0.002)) for p in price_data],
        'Volume': np.random.randint(1000, 10000, size=len(date_range))
    }, index=date_range)
    
    # Ensure Close is realistic
    for i in range(len(df)):
        df.iloc[i, df.columns.get_indexer(['Close'])[0]] = max(
            df.iloc[i, df.columns.get_indexer(['Low'])[0]],
            min(
                df.iloc[i, df.columns.get_indexer(['High'])[0]],
                df.iloc[i, df.columns.get_indexer(['Close'])[0]]
            )
        )
    
    return df

# Function to create model features
def create_features(data, price_col):
    features = pd.DataFrame(index=data.index)
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['day_of_month'] = data.index.day
    features['month'] = data.index.month
    features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    features['is_market_hours'] = ((data.index.hour >= 9) & (data.index.hour <= 16)).astype(int)
    
    # Technical indicators
    for window in [3, 6, 12, 24]:
        features[f'ma_{window}'] = data[price_col].rolling(window=window).mean()
        features[f'std_{window}'] = data[price_col].rolling(window=window).std()
    
    features['momentum'] = data[price_col].diff(periods=1)
    features['momentum_3'] = data[price_col].diff(periods=3)
    features['rate_of_change'] = data[price_col].pct_change(periods=1) * 100
    
    # Lag features
    for i in range(1, 25):  # Increased lag features
        features[f'lag_{i}'] = data[price_col].shift(i)
    
    # Fill any missing values
    features = features.fillna(method='bfill').fillna(method='ffill')
    return features

# Function to make forecasts
def make_forecasts(data, price_col):
    if data is None or len(data) < 24:  # Need at least a day of data
        st.error("Insufficient data for forecasting. Need at least 24 hourly data points.")
        return None
    
    try:
        with st.spinner("Extracting features..."):
            # Create features
            features = create_features(data, price_col)
            
            # Ensure all features are numeric
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].fillna(features[col].mean())
            
            # Keep track of feature names
            feature_names = features.columns.tolist()
        
        with st.spinner("Training model..."):
            # Train model
            X = features.values
            y = data[price_col].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
        
        # Create forecast DataFrames
        last_date = data.index[-1]
        current_value = float(data[price_col].iloc[-1])
        
        # Define forecast horizons
        horizons = {
            "1h": 1,
            "4h": 4,
            "8h": 8,
            "1d": 24,
            "7d": 168,
            "14d": 336
        }
        
        forecasts = {}
        
        with st.spinner("Generating forecasts..."):
            # Make predictions for each horizon
            for name, hours in horizons.items():
                st.sidebar.info(f"Generating {name} forecast...")
                future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=hours, freq='1H')
                forecast_values = []
                
                # Rolling forecast for this horizon
                current_data = data.copy()
                
                for i in range(hours):
                    # Get current features
                    current_features = create_features(current_data, price_col)
                    
                    # Ensure feature consistency
                    current_features = current_features[feature_names]
                    
                    # Scale features
                    current_X = scaler.transform(current_features.values[-1:])
                    
                    # Make prediction
                    prediction = model.predict(current_X)[0]
                    
                    # Add to forecast values
                    forecast_values.append(prediction)
                    
                    # Add to current data for next iteration
                    if i < hours - 1:
                        new_date = future_dates[i]
                        new_row = pd.DataFrame({price_col: [prediction]}, index=[new_date])
                        current_data = pd.concat([current_data, new_row])
                
                # Create forecast DataFrame
                forecasts[name] = pd.DataFrame(
                    {price_col: forecast_values},
                    index=future_dates
                )
                st.sidebar.success(f"{name} forecast complete")
        
        return forecasts, model, feature_names
    
    except Exception as e:
        st.error(f"Error making forecasts: {str(e)}")
        st.error(traceback.format_exc())
        
        # Log the error
        with open("logs/forecast_errors.log", "a") as f:
            f.write(f"{datetime.now()}: Error forecasting {selected_index}: {str(e)}\n")
            f.write(traceback.format_exc() + "\n\n")
            
        return None

# Get data based on selected source
if data_source == "Live Data":
    st.sidebar.info(f"Fetching live data for {selected_index}...")
    data = fetch_live_data(selected_index)
else:
    data = create_demo_data(selected_index)

if data is not None and len(data) >= 24:
    price_col = 'Close'
    
    # Show data info
    st.subheader(f"{selected_index} Multi-Horizon Forecast")
    
    with st.expander("Data Information", expanded=False):
        st.write(f"Data source: {'Real-time market data' if data_source == 'Live Data' else 'Demo data'}")
        st.write(f"Data range: {data.index[0]} to {data.index[-1]}")
        st.write(f"Number of data points: {len(data)}")
        current_price = float(data[price_col].iloc[-1])
        st.write(f"Current {price_col}: {current_price:.2f}")
        
        # Show the last 5 data points
        st.write("Last 5 data points:")
        st.dataframe(data.tail(5))
        
        # Show statistics of the historic data
        st.write("Historical data statistics:")
        st.dataframe(data.describe())
    
    # Generate forecasts
    with st.spinner("Generating multi-horizon forecasts..."):
        result = make_forecasts(data, price_col)
        
        if result:
            forecasts, model, feature_names = result
            
            # Current date and price
            current_date = data.index[-1]
            current_price = float(data[price_col].iloc[-1])
            
            # Create visualization
            st.subheader("Forecast Visualization")
            
            fig = go.Figure()
            
            # Ensure we show full 14 days of historical data
            historical_start = current_date - timedelta(days=14)
            historical_data = data[data.index >= historical_start]
            
            if len(historical_data) < 24:  # If less than a day of actual historical data
                st.warning(f"Limited historical data available: only {len(historical_data)} hours. Showing all available data.")
                historical_data = data  # Show all available data
            
            # Add historical data trace with clear label
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[price_col],
                mode='lines',
                name='Historical Data (Last 14 days)',
                line=dict(color='black', width=2)
            ))
            
            # Colors for different horizons
            horizon_colors = {
                '1h': 'red',
                '4h': 'orange',
                '8h': 'green',
                '1d': 'blue',
                '7d': 'purple',
                '14d': 'brown'
            }
            
            # Add each forecast
            for horizon_name, forecast_df in forecasts.items():
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df[price_col],
                    mode='lines',
                    name=f'{horizon_name} Forecast',
                    line=dict(color=horizon_colors.get(horizon_name, 'gray'), dash='dash')
                ))
            
            # Add vertical line at current time - using string format for compatibility
            current_date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d %H:%M:%S')
            
            # Use shape for vertical line
            fig.add_shape(
                type="line",
                x0=current_date_str,
                y0=0,
                x1=current_date_str,
                y1=1,
                line=dict(color="green", width=2, dash="solid"),
                xref="x",
                yref="paper"
            )
            
            # Add "Current" annotation
            fig.add_annotation(
                x=current_date_str,
                y=1.05,
                text="Current",
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(color="green", size=14)
            )
            
            # Center the view with current date in the middle
            view_start = pd.Timestamp(current_date - timedelta(days=14)).strftime('%Y-%m-%d')
            view_end = pd.Timestamp(current_date + timedelta(days=14)).strftime('%Y-%m-%d')
            
            # Update layout with clear labels
            fig.update_layout(
                title=f"{selected_index} - Historical Data and Multi-Horizon Forecasts",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis=dict(range=[view_start, view_end])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast statistics
            st.subheader("Forecast Statistics")
            
            stats = []
            
            for horizon_name, forecast_df in forecasts.items():
                if not forecast_df.empty:
                    value = float(forecast_df[price_col].iloc[-1])
                    change = ((value - current_price) / current_price) * 100
                    stats.append({
                        'Horizon': horizon_name,
                        'Forecast Value': f"{value:.2f}",
                        'Change': f"{change:+.2f}%",
                        'Direction': "📈 Up" if change > 0 else "📉 Down",
                        'End Date': forecast_df.index[-1].strftime("%Y-%m-%d %H:%M")
                    })
            
            if stats:
                st.table(pd.DataFrame(stats))
            
            # Show feature importance
            if model is not None and hasattr(model, 'feature_importances_') and feature_names:
                with st.expander("Model Information", expanded=False):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=feature_importance['Feature'][:10],
                        y=feature_importance['Importance'][:10],
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title="Top 10 Features by Importance",
                        xaxis_title="Feature",
                        yaxis_title="Importance",
                        xaxis={'categoryorder': 'total descending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save forecasts for future evaluation
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    for horizon_name, forecast_df in forecasts.items():
                        forecast_file = f"data/forecasts/{selected_index}_{horizon_name}_{timestamp}.csv"
                        forecast_df.to_csv(forecast_file)
                        
            # Toggle to see full historical data
            if st.checkbox("Show Full Historical Range"):
                fig_full = go.Figure()
                
                fig_full.add_trace(go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name='Complete Historical Data',
                    line=dict(color='black', width=2)
                ))
                
                # Add current time marker
                fig_full.add_shape(
                    type="line",
                    x0=current_date_str,
                    y0=0,
                    x1=current_date_str,
                    y1=1,
                    line=dict(color="green", width=2, dash="solid"),
                    xref="x",
                    yref="paper"
                )
                
                fig_full.add_annotation(
                    x=current_date_str,
                    y=1.05,
                    text="Current",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    font=dict(color="green", size=14)
                )
                
                fig_full.update_layout(
                    title=f"{selected_index} - Complete Historical Data Range",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_full, use_container_width=True)
        else:
            st.error("Could not generate forecasts. Please check the data.")
else:
    st.error("Insufficient data available.")
    st.info("""
    This dashboard requires sufficient financial data to create forecasts. When working correctly, it should:
    
    1. Display historical data for the past 14 days
    2. Generate forecasts for multiple time horizons (1h, 4h, 8h, 1 day, 7 days, and 14 days)
    3. Show the current date in the middle with a vertical line
    4. Display forecast statistics and model insights
    
    Try selecting "Demo Data" from the sidebar if live data is unavailable.
    """)
