import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Setup
st.set_page_config(page_title="Financial Index Forecast", page_icon="📈")
st.title("📈 Financial Index Forecast")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Create directories
os.makedirs("data/forecasts", exist_ok=True)

# Select index
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox("Select Index:", options=index_options)

# Data source
use_demo = st.sidebar.checkbox("Use Demo Data", value=False)

# Index tickers
tickers = {"DAX": "^GDAXI", "DowJones": "^DJI", "USD_EUR": "EURUSD=X"}

# Create demo data
def create_demo_data(index_name):
    # Set base parameters
    if index_name == "DAX":
        base_value = 18500
        volatility = 100
    elif index_name == "DowJones":
        base_value = 39000
        volatility = 200
    else:  # USD_EUR
        base_value = 0.92
        volatility = 0.005
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate price
    np.random.seed(42)
    prices = []
    price = base_value
    
    for i, date in enumerate(dates):
        # Add time-of-day and day-of-week effects
        hour, day = date.hour, date.dayofweek
        hour_vol = volatility if 9 <= hour <= 16 else volatility * 0.5
        day_factor = 0.0001 * (2 - day) if day <= 4 else 0
        
        # Calculate price change
        change = np.random.normal(day_factor, hour_vol/1000)
        price *= (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0.001, 0.005)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0.001, 0.005)) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'Volume': np.random.randint(1000, 10000, size=len(dates))
    }, index=dates)
    
    return df

# Fetch data
def fetch_data(index_name):
    try:
        ticker = tickers.get(index_name)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h")
        
        if data.empty:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if not data.empty:
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data
        return None
    except:
        return None

# Get features
def get_features(data, price_col):
    features = pd.DataFrame(index=data.index)
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    
    # Technical indicators
    features['ma_3'] = data[price_col].rolling(window=3).mean()
    features['ma_6'] = data[price_col].rolling(window=6).mean()
    features['momentum'] = data[price_col].diff(periods=1)
    
    # Lag features
    for i in range(1, 7):
        features[f'lag_{i}'] = data[price_col].shift(i)
    
    # Fill missing values
    features = features.fillna(method='bfill').fillna(method='ffill')
    return features

# Generate forecasts
def generate_forecasts(data, price_col):
    try:
        # Extract features
        features = get_features(data, price_col)
        feature_names = features.columns.tolist()
        
        # Train model
        X = features.values
        y = data[price_col].values
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Create forecasts
        last_date = data.index[-1]
        current_value = float(data[price_col].iloc[-1])
        
        # Define forecast horizons
        horizons = {"1h": 1, "4h": 4, "8h": 8, "1d": 24, "7d": 168, "14d": 336}
        forecasts = {}
        
        # Generate each forecast
        for name, hours in horizons.items():
            future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=hours, freq='1H')
            values = []
            
            # Create rolling forecast
            temp_data = data.copy()
            
            for i in range(hours):
                # Get features
                temp_features = get_features(temp_data, price_col)
                temp_features = temp_features[feature_names]
                
                # Make prediction
                current_X = scaler.transform(temp_features.values[-1:])
                prediction = model.predict(current_X)[0]
                values.append(prediction)
                
                # Add prediction to temp data
                if i < hours - 1:
                    new_date = future_dates[i]
                    new_row = pd.DataFrame({price_col: [prediction]}, index=[new_date])
                    temp_data = pd.concat([temp_data, new_row])
            
            # Store forecast
            forecasts[name] = pd.DataFrame({price_col: values}, index=future_dates)
        
        return forecasts, model, feature_names
    except Exception as e:
        st.error(f"Error generating forecasts: {str(e)}")
        return None

# Main execution
if use_demo:
    data = create_demo_data(selected_index)
    st.sidebar.success("Using demo data")
else:
    data = fetch_data(selected_index)
    if data is None or data.empty:
        st.sidebar.warning("Could not fetch live data. Using demo data instead.")
        data = create_demo_data(selected_index)

if data is not None:
    price_col = 'Close'
    
    # Show basic info
    st.subheader(f"{selected_index} Forecast")
    
    with st.expander("Data Information"):
        st.write(f"Data range: {data.index[0]} to {data.index[-1]}")
        current_price = float(data[price_col].iloc[-1])
        st.write(f"Current price: {current_price:.2f}")
        st.dataframe(data.tail())
    
    # Generate forecasts
    with st.spinner("Generating forecasts..."):
        result = generate_forecasts(data, price_col)
        
        if result:
            forecasts, model, feature_names = result
            
            # Current date and value
            current_date = data.index[-1]
            
            # Create visualization
            fig = go.Figure()
            
            # Add historical data
            historical_start = current_date - timedelta(days=14)
            historical_data = data[data.index >= historical_start]
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[price_col],
                mode='lines',
                name='Historical Data',
                line=dict(color='black', width=2)
            ))
            
            # Colors for forecasts
            colors = {'1h': 'red', '4h': 'orange', '8h': 'green', 
                      '1d': 'blue', '7d': 'purple', '14d': 'brown'}
            
            # Add forecasts
            for name, forecast in forecasts.items():
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=forecast[price_col],
                    mode='lines',
                    name=f'{name} Forecast',
                    line=dict(color=colors.get(name, 'gray'), dash='dash')
                ))
            
            # Add current line
            current_date_str = current_date.strftime('%Y-%m-%d %H:%M:%S')
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
            
            # Add annotation
            fig.add_annotation(
                x=current_date_str,
                y=1.05,
                text="Current",
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(color="green")
            )
            
            # Set view range
            view_start = (current_date - timedelta(days=14)).strftime('%Y-%m-%d')
            view_end = (current_date + timedelta(days=14)).strftime('%Y-%m-%d')
            
            # Update layout
            fig.update_layout(
                title=f"{selected_index} - Historical Data and Forecasts",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis=dict(range=[view_start, view_end])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast stats
            st.subheader("Forecast Statistics")
            
            stats = []
            for name, forecast in forecasts.items():
                if not forecast.empty:
                    value = float(forecast[price_col].iloc[-1])
                    change = ((value - current_price) / current_price) * 100
                    stats.append({
                        'Horizon': name,
                        'Value': f"{value:.2f}",
                        'Change': f"{change:+.2f}%",
                        'Direction': "📈 Up" if change > 0 else "📉 Down",
                        'End Date': forecast.index[-1].strftime("%Y-%m-%d %H:%M")
                    })
            
            if stats:
                st.table(pd.DataFrame(stats))
            
            # Show feature importance
            with st.expander("Model Information"):
                importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance['Feature'][:10],
                    y=importance['Importance'][:10],
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    title="Top Features by Importance",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
