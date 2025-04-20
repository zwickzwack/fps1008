import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import traceback
import yfinance as yf

# Dashboard title
st.set_page_config(page_title="Financial Index Forecast", page_icon="📈", layout="wide")
st.title("📈 Financial Index Forecast Dashboard")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Select index
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox("Select Index:", options=index_options)

# Map index names to Yahoo Finance tickers
index_tickers = {
    "DAX": "^GDAXI",
    "DowJones": "^DJI", 
    "USD_EUR": "EURUSD=X"
}

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/forecasts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Function to fetch real-time data from Yahoo Finance
def fetch_live_data(index_name, days_back=14):
    try:
        ticker = index_tickers.get(index_name)
        if not ticker:
            return None
        
        # Get data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try hourly data
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h")
        
        if data.empty:
            # Try daily data as fallback
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            if not data.empty:
                # Convert daily to hourly by forward-filling
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            # Make index timezone-naive
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to create demo data
def create_demo_data(index_name, days_back=14):
    # Set parameters based on index
    if index_name == "DAX":
        base_value = 18500
        volatility = 100
    elif index_name == "DowJones":
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
    np.random.seed(42)
    price_data = []
    current_price = base_value
    
    for i, date in enumerate(date_range):
        # Add time and day effects
        hour = date.hour
        day = date.dayofweek
        
        hour_volatility = volatility if 9 <= hour <= 16 else volatility * 0.5
        day_trend = 0.0001 * (2 - day) if day <= 4 else 0
        
        # Random price movement
        price_change = np.random.normal(day_trend, hour_volatility/1000)
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
    
    return df

# Function to create model features
def create_features(data, price_col):
    features = pd.DataFrame(index=data.index)
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    
    # Technical indicators
    features['ma_3'] = data[price_col].rolling(window=3).mean()
    features['ma_6'] = data[price_col].rolling(window=6).mean()
    features['ma_12'] = data[price_col].rolling(window=12).mean()
    features['momentum'] = data[price_col].diff(periods=1)
    features['volatility'] = data[price_col].rolling(window=12).std()
    
    # Lag features
    for i in range(1, 13):
        features[f'lag_{i}'] = data[price_col].shift(i)
    
    # Fill any missing values
    features = features.fillna(method='bfill').fillna(method='ffill')
    return features

# Function to make forecasts
def make_forecasts(data, price_col):
    if data is None or len(data) < 24:  # Need at least a day of data
        return None
    
    try:
        # Create features
        features = create_features(data, price_col)
        
        # Ensure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(features[col].mean())
        
        # Keep track of feature names
        feature_names = features.columns.tolist()
        
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
        
        # Make predictions for each horizon
        for name, hours in horizons.items():
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
        
        return forecasts, model, feature_names
    
    except Exception as e:
        st.error(f"Error making forecasts: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Get data
st.sidebar.info(f"Fetching data for {selected_index}...")
data = fetch_live_data(selected_index)

if data is None:
    st.sidebar.warning("Using demo data")
    data = create_demo_data(selected_index)

if data is not None:
    price_col = 'Close'
    
    # Show data info
    st.subheader(f"{selected_index} Forecast")
    
    with st.expander("Data Information"):
        st.write(f"Data range: {data.index[0]} to {data.index[-1]}")
        current_price = float(data[price_col].iloc[-1])
        st.write(f"Current {price_col}: {current_price:.2f}")
        st.dataframe(data.tail(5))
    
    # Generate forecasts
    with st.spinner("Generating forecasts..."):
        result = make_forecasts(data, price_col)
        
        if result:
            forecasts, model, feature_names = result
            
            # Current date and price
            current_date = data.index[-1]
            current_price = float(data[price_col].iloc[-1])
            
            # Create visualization
            st.subheader("Forecast Visualization")
            
            fig = go.Figure()
            
            # Add historical data (last 7 days)
            historical_start = current_date - timedelta(days=7)
            historical_data = data[data.index >= historical_start]
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[price_col],
                mode='lines',
                name='Historical Data',
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
            
            # Add vertical line at current time - FIXED BY CONVERTING TO STRING
            current_date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d %H:%M:%S')
            
            # Use shape instead of add_vline for better compatibility
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
            
            # Add text annotation for current line
            fig.add_annotation(
                x=current_date_str,
                y=1.05,
                text="Current",
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(color="green")
            )
            
            # Center the view (+/- 7 days)
            view_start = pd.Timestamp(current_date - timedelta(days=7)).strftime('%Y-%m-%d')
            view_end = pd.Timestamp(current_date + timedelta(days=7)).strftime('%Y-%m-%d')
            
            fig.update_layout(
                title=f"{selected_index} - Multi-Horizon Forecasts",
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
                with st.expander("Model Information"):
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
            
            # Create full view toggle
            if st.checkbox("Show Full Forecast Range (14 days)"):
                # Create a new visualization with full range
                fig2 = go.Figure()
                
                # Add all historical data
                fig2.add_trace(go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='black', width=2)
                ))
                
                # Add each forecast
                for horizon_name, forecast_df in forecasts.items():
                    fig2.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[price_col],
                        mode='lines',
                        name=f'{horizon_name} Forecast',
                        line=dict(color=horizon_colors.get(horizon_name, 'gray'), dash='dash')
                    ))
                
                # Add vertical line at current time - FIXED BY CONVERTING TO STRING
                fig2.add_shape(
                    type="line",
                    x0=current_date_str,
                    y0=0,
                    x1=current_date_str,
                    y1=1,
                    line=dict(color="green", width=2, dash="solid"),
                    xref="x",
                    yref="paper"
                )
                
                # Add text annotation for current line
                fig2.add_annotation(
                    x=current_date_str,
                    y=1.05,
                    text="Current",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    font=dict(color="green")
                )
                
                fig2.update_layout(
                    title=f"{selected_index} - Complete Historical Data and Forecasts",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Could not generate forecasts. Please check the data.")
else:
    st.error("No data available. Please check your internet connection.")
