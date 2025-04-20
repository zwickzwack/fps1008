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
import json
import time

# Dashboard title
st.set_page_config(page_title="Financial Index Multi-Horizon Forecast", page_icon="📈", layout="wide")
st.title("📈 Financial Index Multi-Horizon Forecast Dashboard")

# Data directories
DATA_DIR = "data/raw"
MODELS_DIR = "data/models"
LOG_DIR = "logs"
FORECAST_DIR = "data/forecasts"

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

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

# Function to fetch real-time data from Yahoo Finance
def fetch_live_data(index_name, days_back=14):
    try:
        ticker = index_tickers.get(index_name)
        if not ticker:
            st.warning(f"No ticker found for {index_name}")
            return None
        
        st.sidebar.info(f"Fetching data for {index_name} ({ticker})...")
        
        # Get data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Download data with 1h interval if possible
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h")
        
        if data.empty:
            st.warning(f"No hourly data available for {index_name}. Trying with daily interval...")
            # Try with daily data if hourly is not available
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            if not data.empty:
                # Convert daily data to hourly by forward-filling
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            st.sidebar.success(f"Successfully loaded {len(data)} records for {index_name}")
            
            # Make sure index is timezone-naive to avoid issues
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
                
            return data
        else:
            st.warning(f"No data found for {index_name}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data for {index_name}: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Create demo data if no real data is available
def create_demo_data(index_name, days_back=14):
    st.warning(f"Creating demo data for {index_name}")
    
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
    
    # Generate dates for the past X days with hourly intervals
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate price data with realistic patterns
    np.random.seed(42)
    price_data = []
    current_price = base_value
    
    for i, date in enumerate(date_range):
        # Add time-of-day effect (more volatile during market open)
        hour = date.hour
        if 9 <= hour <= 16:  # Market hours
            hour_volatility = volatility
        else:
            hour_volatility = volatility * 0.5
        
        # Add day-of-week effect
        day = date.dayofweek
        if day <= 4:  # Weekdays
            # More positive trend on Monday, more negative on Friday
            day_trend = 0.0001 * (2 - day)
        else:  # Weekend
            day_trend = 0
            hour_volatility *= 0.3  # Less volatile on weekends
        
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

# Function to create features for prediction
def extract_features(df, price_col, window_size=12):
    # Create a DataFrame with the same index as the input
    features = pd.DataFrame(index=df.index)
    
    # Extract time-based features
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    
    # Calculate moving averages
    features['ma_3'] = df[price_col].rolling(window=3).mean()
    features['ma_6'] = df[price_col].rolling(window=6).mean()
    features['ma_12'] = df[price_col].rolling(window=12).mean()
    
    # Calculate momentum indicators
    features['momentum'] = df[price_col].diff(periods=1)
    features['rate_of_change'] = df[price_col].pct_change(periods=1) * 100
    features['volatility'] = df[price_col].rolling(window=window_size).std()
    
    # Create lag features
    for i in range(1, window_size + 1):
        features[f'lag_{i}'] = df[price_col].shift(i)
    
    # Fill missing values
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    return features

# Function to create forecasts for different time horizons
def create_multi_horizon_forecasts(data, price_col, window_size=12):
    try:
        if data is None or len(data) < window_size * 2:
            st.warning(f"Not enough data for forecasting. Need at least {window_size * 2} data points.")
            return None, None, None
            
        # Define forecast horizons (in hours)
        horizons = {
            "1h": 1,
            "4h": 4,
            "8h": 8,
            "1d": 24,
            "7d": 168,  # 7 days * 24 hours
            "14d": 336  # 14 days * 24 hours
        }
        
        # Make a copy of the data to avoid modifying the original
        input_data = data.copy()
        
        # Extract features from the data
        features = extract_features(input_data, price_col, window_size)
        
        # Make sure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(features[col].mean())
        
        # Keep track of feature names for consistency
        feature_names = list(features.columns)
        
        # Train scaler
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train model on historical data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, input_data[price_col])
        
        # Current timestamp and value
        current_date = input_data.index[-1]
        current_value = input_data[price_col].iloc[-1]
        
        # Dictionary to store all forecasts
        all_forecasts = {}
        
        # Generate forecasts for each horizon
        for horizon_name, hours in horizons.items():
            # Create future dates
            future_dates = pd.date_range(start=current_date + timedelta(hours=1), periods=hours, freq='1H')
            
            # Initialize forecast DataFrame
            forecast_df = pd.DataFrame(index=future_dates, columns=[price_col])
            
            # Use a temporary DataFrame for rolling predictions
            temp_data = input_data.copy()
            
            # Generate predictions for each step in the horizon
            for i in range(hours):
                # Extract features
                temp_features = extract_features(temp_data, price_col, window_size)
                
                # Ensure feature ordering is consistent
                temp_features = temp_features[feature_names]
                
                # Get the latest features
                latest_features = temp_features.iloc[-1:].values
                
                # Scale features
                latest_features_scaled = scaler.transform(latest_features)
                
                # Make prediction
                prediction = model.predict(latest_features_scaled)[0]
                
                # Get next date
                next_date = future_dates[i]
                
                # Store prediction
                forecast_df.loc[next_date, price_col] = prediction
                
                # Add prediction to temp_data for next iteration
                new_row = pd.DataFrame({price_col: [prediction]}, index=[next_date])
                temp_data = pd.concat([temp_data, new_row])
            
            # Store the forecast
            all_forecasts[horizon_name] = forecast_df
            
            # Save the forecast for future evaluation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            forecast_file = f"{FORECAST_DIR}/{selected_index}_{horizon_name}_{timestamp}.csv"
            forecast_df.to_csv(forecast_file)
            st.sidebar.info(f"Saved {horizon_name} forecast to {os.path.basename(forecast_file)}")
        
        # Create a combined DataFrame for visualization
        # Start with full historical data
        combined_df = pd.DataFrame(index=pd.date_range(
            start=input_data.index[0], 
            end=current_date + timedelta(hours=horizons["14d"]), 
            freq='1H'
        ))
        
        # Add historical data
        combined_df.loc[input_data.index, 'historical'] = input_data[price_col]
        
        # Add each forecast
        for horizon_name, forecast_df in all_forecasts.items():
            combined_df.loc[forecast_df.index, horizon_name] = forecast_df[price_col]
        
        return combined_df, model, feature_names
        
    except Exception as e:
        st.error(f"Error creating forecasts: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

# Function to evaluate historical forecasts
def load_historical_forecasts(index_name, horizon="1d"):
    try:
        # Find all forecast files for this index and horizon
        pattern = f"{FORECAST_DIR}/{index_name}_{horizon}_*.csv"
        forecast_files = glob.glob(pattern)
        
        if not forecast_files:
            return None
            
        # Sort by creation time (newest first)
        forecast_files.sort(key=os.path.getctime, reverse=True)
        
        # Load the forecasts
        forecasts = []
        timestamps = []
        
        for file in forecast_files[:5]:  # Load the 5 most recent forecasts
            try:
                # Extract timestamp from filename
                filename = os.path.basename(file)
                parts = filename.split('_')
                timestamp = '_'.join(parts[2:]).replace('.csv', '')
                creation_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                
                # Load forecast
                forecast = pd.read_csv(file, index_col=0)
                forecast.index = pd.to_datetime(forecast.index)
                
                forecasts.append(forecast)
                timestamps.append(creation_time)
            except Exception as e:
                st.warning(f"Could not load forecast file {file}: {e}")
        
        return forecasts, timestamps
    except Exception as e:
        st.error(f"Error loading historical forecasts: {e}")
        return None

# Try to fetch live data, use demo data as fallback
data = fetch_live_data(selected_index, days_back=14)

if data is None:
    data = create_demo_data(selected_index, days_back=14)

if data is not None:
    # Find price column (Yahoo Finance data uses 'Close')
    price_col = 'Close'
    
    # Main content
    st.subheader(f"{selected_index} Multi-Horizon Forecast Dashboard")
    
    # Data info
    with st.expander("Data Information"):
        st.write(f"Data shape: {data.shape}")
        st.write(f"Data range: {data.index[0]} to {data.index[-1]}")
        
        # Fixed current price display to avoid formatting error
        current_price = float(data[price_col].iloc[-1])
        st.write(f"Current {price_col}: {current_price:.2f}")
        
        st.dataframe(data.tail(5))
    
    # Generate new forecasts or show previous ones
    forecast_action = st.radio(
        "Forecast Action:", 
        options=["Generate New Forecasts", "View Historical Forecasts"],
        index=0
    )
    
    if forecast_action == "Generate New Forecasts":
        with st.spinner("Generating multi-horizon forecasts..."):
            # Create forecasts
            forecasts, model, feature_names = create_multi_horizon_forecasts(data, price_col)
            
            if forecasts is not None:
                # Current date and value
                current_date = data.index[-1]
                current_value = float(data[price_col].iloc[-1])
                
                # Create the visualization
                st.subheader("Multi-Horizon Forecast Visualization")
                
                # Create figure
                fig = go.Figure()
                
                # Add historical data
                historical_mask = ~pd.isna(forecasts['historical'])
                
                fig.add_trace(go.Scatter(
                    x=forecasts.index[historical_mask],
                    y=forecasts['historical'][historical_mask],
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
                for horizon_name in ['1h', '4h', '8h', '1d', '7d', '14d']:
                    if horizon_name in forecasts.columns:
                        # Mask for non-NA values
                        mask = ~pd.isna(forecasts[horizon_name])
                        
                        fig.add_trace(go.Scatter(
                            x=forecasts.index[mask],
                            y=forecasts[horizon_name][mask],
                            mode='lines',
                            name=f'{horizon_name} Forecast',
                            line=dict(color=horizon_colors.get(horizon_name, 'gray'), dash='dash')
                        ))
                
                # Add a vertical line at the current time
                fig.add_vline(
                    x=current_date,
                    line_width=2,
                    line_dash="solid",
                    line_color="green",
                    annotation_text="Current"
                )
                
                # Center the view around the current date
                # 7 days before and 7 days after
                view_start = current_date - timedelta(days=7)
                view_end = current_date + timedelta(days=7)
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_index} - Multi-Horizon Forecasts",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    xaxis=dict(range=[view_start, view_end])
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast statistics
                st.subheader("Forecast Statistics")
                
                # Create a table for forecast values and changes
                stats = []
                
                for horizon_name in ['1h', '4h', '8h', '1d', '7d', '14d']:
                    if horizon_name in forecasts.columns:
                        # Get the last forecast value for this horizon
                        horizon_df = forecasts[horizon_name].dropna()
                        if not horizon_df.empty:
                            # Convert to float to avoid formatting issues
                            value = float(horizon_df.iloc[-1])
                            change = ((value - current_value) / current_value) * 100
                            stats.append({
                                'Horizon': horizon_name,
                                'Forecast Value': f"{value:.2f}",
                                'Change': f"{change:+.2f}%",
                                'Direction': "📈 Up" if change > 0 else "📉 Down",
                                'End Date': horizon_df.index[-1].strftime("%Y-%m-%d %H:%M")
                            })
                
                if stats:
                    st.table(pd.DataFrame(stats))
                
                # Show model feature importance
                if model is not None and hasattr(model, 'feature_importances_') and feature_names is not None:
                    with st.expander("Model Information"):
                        # Create feature importance DataFrame
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=feature_importance['Feature'][:10],  # Top 10 features
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
            else:
                st.error("Could not generate forecasts. Please check the data.")
    else:
        # View historical forecasts
        st.subheader("Historical Forecast Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            horizon_to_view = st.selectbox(
                "Select Forecast Horizon:", 
                options=["1h", "4h", "8h", "1d", "7d", "14d"],
                index=3  # Default to 1d
            )
        
        with col2:
            # Placeholder for additional controls if needed
            st.write("View how past forecasts have performed compared to actual data.")
        
        # Load historical forecasts
        historical_data = load_historical_forecasts(selected_index, horizon_to_view)
        
        if historical_data:
            forecasts, timestamps = historical_data
            
            if forecasts:
                # Create visualization
                fig = go.Figure()
                
                # Add actual data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name='Actual Data',
                    line=dict(color='black', width=2)
                ))
                
                # Add each historical forecast
                colors = ['red', 'blue', 'green', 'purple', 'orange']
                
                for i, (forecast, timestamp) in enumerate(zip(forecasts, timestamps)):
                    forecast_label = f"Forecast ({timestamp.strftime('%Y-%m-%d %H:%M')})"
                    
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.iloc[:, 0],  # First column has the forecast values
                        mode='lines',
                        name=forecast_label,
                        line=dict(color=colors[i % len(colors)], dash='dash')
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_index} - Historical {horizon_to_view} Forecasts vs. Actual",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Evaluate forecast accuracy
                st.subheader("Forecast Accuracy Evaluation")
                
                accuracy_stats = []
                
                for i, (forecast, timestamp) in enumerate(zip(forecasts, timestamps)):
                    # Find overlapping dates between forecast and actual data
                    common_dates = forecast.index.intersection(data.index)
                    
                    if len(common_dates) > 0:
                        # Get actual and forecasted values for common dates
                        actual_values = data.loc[common_dates, price_col]
                        forecast_values = forecast.loc[common_dates].iloc[:, 0]
                        
                        # Calculate errors
                        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
                        mae = np.mean(np.abs(actual_values - forecast_values))
                        
                        # Direction accuracy
                        if len(common_dates) > 1:
                            actual_direction = np.diff(actual_values) > 0
                            forecast_direction = np.diff(forecast_values) > 0
                            direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
                        else:
                            direction_accuracy = np.nan
                        
                        accuracy_stats.append({
                            'Forecast Date': timestamp.strftime("%Y-%m-%d %H:%M"),
                            'MAPE (%)': f"{mape:.2f}%",
                            'MAE': f"{mae:.4f}",
                            'Direction Accuracy (%)': f"{direction_accuracy:.1f}%" if not np.isnan(direction_accuracy) else "N/A",
                            'Data Points': len(common_dates)
                        })
                
                if accuracy_stats:
                    st.table(pd.DataFrame(accuracy_stats))
                else:
                    st.info("No overlapping data available to evaluate forecast accuracy.")
            else:
                st.info(f"No historical {horizon_to_view} forecasts found for {selected_index}.")
        else:
            st.info(f"No historical forecasts found for {selected_index}.")
else:
    st.error(f"No data available for {selected_index}. Please check your internet connection.")
    
    # Show a placeholder message
    st.info("""
    This dashboard requires financial data. When working properly, it will display:
    
    - Historical data for the past 14 days with current date centered
    - Multiple forecast horizons: 1h, 4h, 8h, 1 day, 7 days, and 14 days
    - Evaluation of previous forecasts against actual data
    
    Please ensure you have an active internet connection and try again.
    """)
