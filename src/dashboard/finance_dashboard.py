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
st.set_page_config(page_title="Financial Index Rolling Forecast", page_icon="📈", layout="wide")
st.title("📈 Financial Index Rolling Forecast Dashboard")

# Data directories
DATA_DIR = "data/raw"
MODELS_DIR = "data/models"
LOG_DIR = "logs"

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"Current User's Login: zwickzwack")

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
            st.warning(f"No data available for {index_name} with hourly interval. Trying with daily interval...")
            # Try with daily data if hourly is not available
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            if not data.empty:
                # Convert daily data to hourly by forward-filling
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            st.sidebar.success(f"Successfully loaded {len(data)} records for {index_name}")
            
            # Make sure index is timezone-naive to avoid issues
            if data.index.tzinfo is not None:
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
    # Extract time-based features
    features = pd.DataFrame(index=df.index)
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

# Function to create rolling forecast
def create_rolling_forecast(df, price_col, forecast_hours=336, window_size=12): # 336 hours = 14 days
    try:
        if df is None or df.empty:
            st.warning("No data available for forecasting")
            return None, None, None
        
        # Create deep copy to avoid modifying original
        data = df.copy()
        
        # Make sure we have enough data
        if len(data) < window_size * 2:
            st.warning(f"Not enough data points ({len(data)}) for forecasting. Need at least {window_size * 2}.")
            return None, None, None
        
        # Extract features from historical data
        features = extract_features(data, price_col, window_size)
        
        # Make sure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            # Fill NA values with column mean or 0
            if features[col].isna().any():
                mean_val = features[col].mean()
                features[col] = features[col].fillna(mean_val if not np.isnan(mean_val) else 0)
        
        # Keep track of feature names for consistency
        feature_names = features.columns.tolist()
        
        # Create scaler and scale the features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get the target values (price)
        y = data[price_col].values
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, y)
        
        # Get current value and dates
        last_date = data.index[-1]
        last_value = data[price_col].iloc[-1]
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=forecast_hours, freq='1H')
        
        # Create DataFrame for storing the forecast
        historical_df = pd.DataFrame({price_col: data[price_col]})
        forecast_df = pd.DataFrame(index=future_dates, columns=[price_col])
        
        # Current state for rolling forecast
        current_df = data.copy()
        
        # Generate rolling forecast
        for i, future_date in enumerate(future_dates):
            # Extract features for the latest available data
            latest_features = extract_features(current_df, price_col, window_size)
            
            # Ensure feature ordering is consistent
            latest_features = latest_features[feature_names]
            
            # Scale features
            latest_features_scaled = scaler.transform(latest_features.iloc[-1:])
            
            # Make prediction
            prediction = model.predict(latest_features_scaled)[0]
            
            # Store prediction in forecast DataFrame
            forecast_df.loc[future_date, price_col] = prediction
            
            # Add prediction to current_df for next iteration
            new_row = pd.DataFrame({price_col: [prediction]}, index=[future_date])
            current_df = pd.concat([current_df, new_row])
        
        # Combine historical and forecast data
        combined_df = pd.concat([historical_df, forecast_df])
        
        # Calculate confidence intervals
        volatility = data[price_col].pct_change().std()
        combined_df['upper_bound'] = combined_df[price_col]
        combined_df['lower_bound'] = combined_df[price_col]
        
        # Apply confidence intervals only to forecasted values
        for i, date in enumerate(future_dates):
            confidence_margin = combined_df.loc[date, price_col] * volatility * np.sqrt(i+1) * 1.96  # 95% confidence
            combined_df.loc[date, 'upper_bound'] = combined_df.loc[date, price_col] + confidence_margin
            combined_df.loc[date, 'lower_bound'] = combined_df.loc[date, price_col] - confidence_margin
        
        return combined_df, model, feature_names
        
    except Exception as e:
        st.error(f"Error creating rolling forecast: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

# Try to fetch live data, use demo data as fallback
data = fetch_live_data(selected_index, days_back=14)

if data is None:
    data = create_demo_data(selected_index, days_back=14)

if data is not None:
    # Find price column (Yahoo Finance data uses 'Close')
    price_col = 'Close'
    
    # Main content
    st.subheader(f"{selected_index} Rolling Forecast (14 Days)")
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(data.tail(10))
        st.write(f"Data shape: {data.shape}")
        st.write(f"Data types: {data.dtypes}")
        st.write(f"Index type: {type(data.index)}")
        st.write(f"First date: {data.index[0]}")
        st.write(f"Last date: {data.index[-1]}")
    
    with st.spinner("Generating rolling forecast..."):
        # Create the rolling forecast
        rolling_forecast, model, feature_names = create_rolling_forecast(data, price_col, forecast_hours=336)
        
        if rolling_forecast is not None:
            # Plot the data and forecast
            fig = go.Figure()
            
            # Historical data
            historical_end = data.index[-1]
            historical_data = rolling_forecast.loc[:historical_end]
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[price_col],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Forecasted data
            forecast_data = rolling_forecast.loc[historical_end:].iloc[1:]  # Skip the overlap point
            
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data[price_col],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            if 'upper_bound' in forecast_data.columns and 'lower_bound' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['upper_bound'],
                    mode='lines',
                    name='Upper 95% CI',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['lower_bound'],
                    mode='lines',
                    name='Lower 95% CI',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(width=0),
                    showlegend=False
                ))
            
            # Add a vertical line at the current time
            fig.add_vline(
                x=historical_end,
                line_width=2,
                line_dash="solid",
                line_color="green",
                annotation_text="Current"
            )
            
            # Adjust layout
            fig.update_layout(
                title=f"{selected_index} - Historical Data and 14-Day Hourly Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast statistics
            st.subheader("Forecast Statistics")
            
            # Calculate changes over different periods
            last_value = historical_data[price_col].iloc[-1] if not historical_data.empty else None
            forecast_24h = forecast_data[price_col].iloc[23] if len(forecast_data) > 23 else None
            forecast_7d = forecast_data[price_col].iloc[167] if len(forecast_data) > 167 else None
            forecast_14d = forecast_data[price_col].iloc[-1] if not forecast_data.empty else None
            
            col1, col2, col3, col4 = st.columns(4)
            
            if last_value is not None:
                with col1:
                    st.metric(
                        label="Current Value",
                        value=f"{last_value:.2f}"
                    )
            
            if last_value is not None and forecast_24h is not None:
                change_24h = ((forecast_24h - last_value) / last_value) * 100
                with col2:
                    st.metric(
                        label="24-Hour Forecast",
                        value=f"{forecast_24h:.2f}",
                        delta=f"{change_24h:+.2f}%"
                    )
            
            if last_value is not None and forecast_7d is not None:
                change_7d = ((forecast_7d - last_value) / last_value) * 100
                with col3:
                    st.metric(
                        label="7-Day Forecast",
                        value=f"{forecast_7d:.2f}",
                        delta=f"{change_7d:+.2f}%"
                    )
            
            if last_value is not None and forecast_14d is not None:
                change_14d = ((forecast_14d - last_value) / last_value) * 100
                with col4:
                    st.metric(
                        label="14-Day Forecast",
                        value=f"{forecast_14d:.2f}",
                        delta=f"{change_14d:+.2f}%"
                    )
            
            # Display trend analysis
            st.subheader("Forecast Trend Analysis")
            
            # Create hourly trend visualization
            hourly_forecasts = forecast_data[price_col].values
            
            if len(hourly_forecasts) > 0 and last_value is not None:
                hourly_changes = np.diff(np.append([last_value], hourly_forecasts))
                hourly_pct_changes = (hourly_changes / np.append([last_value], hourly_forecasts)[:-1]) * 100
                
                # Group by day for clearer visualization
                daily_labels = []
                daily_changes = []
                daily_avg_values = []
                
                for day in range(14):
                    start_idx = day * 24
                    end_idx = min(start_idx + 24, len(hourly_pct_changes))
                    
                    if start_idx < len(hourly_pct_changes):
                        day_changes = hourly_pct_changes[start_idx:end_idx]
                        daily_labels.append(f"Day {day+1}")
                        daily_changes.append(np.sum(day_changes))
                        
                        if start_idx < len(hourly_forecasts):
                            day_values = hourly_forecasts[start_idx:end_idx]
                            daily_avg_values.append(np.mean(day_values))
                
                if daily_labels and daily_changes and daily_avg_values:
                    # Plot daily changes
                    fig = go.Figure()
                    
                    # Add bar chart for daily changes
                    fig.add_trace(go.Bar(
                        x=daily_labels,
                        y=daily_changes,
                        marker_color=np.where(np.array(daily_changes) >= 0, 'green', 'red'),
                        name="Daily % Change"
                    ))
                    
                    # Add line for average values
                    fig.add_trace(go.Scatter(
                        x=daily_labels,
                        y=daily_avg_values,
                        mode='lines+markers',
                        name="Average Daily Value",
                        yaxis="y2"
                    ))
                    
                    # Update layout with secondary y-axis
                    fig.update_layout(
                        title="Forecasted Daily Price Changes",
                        xaxis_title="Day",
                        yaxis=dict(
                            title="Daily % Change",
                            side="left"
                        ),
                        yaxis2=dict(
                            title="Average Daily Value",
                            side="right",
                            overlaying="y",
                            showgrid=False
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display model information
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
            
            # Table with raw forecast data
            with st.expander("View Raw Forecast Data (Hourly)"):
                # Format the forecast data for display
                display_data = forecast_data[[price_col]].copy()
                display_data.index = display_data.index.strftime("%Y-%m-%d %H:%M")
                display_data.index.name = "Date & Time"
                display_data.columns = ["Forecasted Price"]
                
                st.dataframe(display_data)
        else:
            st.error("Could not generate forecast. Please check the data.")
else:
    st.error("No data available. Please check your internet connection.")
    
    # Show a placeholder message
    st.info("""
    This dashboard requires financial data. When working properly, it will display:
    
    - Historical data for the past 14 days
    - Rolling hourly forecast for the next 14 days
    - Trend analysis for the forecasted period
    - Confidence intervals for the predictions
    
    Please ensure you have an active internet connection and try again.
    """)
