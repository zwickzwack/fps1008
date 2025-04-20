import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import calendar
import traceback

# Dashboard setup
st.set_page_config(page_title="Market Predictor", page_icon="📈", layout="wide")
st.title("📈 Financial Market Predictor")

# Current time
current_time = datetime.now()
st.sidebar.info(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Market configuration
MARKETS = {
    "DAX": {
        "ticker": "^GDAXI",
        "description": "German Stock Index",
        "currency": "EUR",
        "opening_time": time(9, 0),  # 9:00 AM
        "closing_time": time(17, 30),  # 5:30 PM
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
    },
    "DowJones": {
        "ticker": "^DJI",
        "description": "Dow Jones Industrial Average",
        "currency": "USD",
        "opening_time": time(9, 30),  # 9:30 AM
        "closing_time": time(16, 0),  # 4:00 PM
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
    }
}

# Select market
selected_market = st.sidebar.selectbox(
    "Select Market:",
    options=list(MARKETS.keys())
)

# Check if market is open
def is_market_open(market_name, check_time=None):
    if check_time is None:
        check_time = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return False
    
    # Check if it's a trading day (weekday)
    if check_time.weekday() not in market["trading_days"]:
        return False
    
    # Check if within trading hours
    current_time = check_time.time()
    return market["opening_time"] <= current_time <= market["closing_time"]

# Get the next trading day
def get_next_trading_day(market_name, from_date=None):
    if from_date is None:
        from_date = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return None
    
    # Start with tomorrow
    next_day = from_date + timedelta(days=1)
    
    # Find the next trading day
    while next_day.weekday() not in market["trading_days"]:
        next_day = next_day + timedelta(days=1)
    
    # Return the opening time on that day
    next_opening = datetime.combine(next_day.date(), market["opening_time"])
    return next_opening

# Get the end of the next trading day
def get_end_of_next_trading_day(market_name, from_date=None):
    next_opening = get_next_trading_day(market_name, from_date)
    
    if next_opening is None:
        return None
    
    market = MARKETS.get(market_name)
    if not market:
        return None
    
    # Return the closing time on the same day
    next_closing = datetime.combine(next_opening.date(), market["closing_time"])
    return next_closing

# Format date for display
def format_trading_day(date_obj):
    # Get the day name and format the date
    day_name = calendar.day_name[date_obj.weekday()]
    return f"{day_name}, {date_obj.strftime('%Y-%m-%d')}"

# Create demo data
def create_demo_data(market_name):
    # Set base values
    if market_name == "DAX":
        base_value = 18500
        volatility = 100
    else:  # DowJones
        base_value = 39000
        volatility = 200
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 days for better model training
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
def fetch_historical_data(market_name):
    try:
        market_info = MARKETS.get(market_name)
        if not market_info:
            return None
        
        ticker = market_info["ticker"]
        
        # Get data from Yahoo Finance - 30 days for better model training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        st.sidebar.info(f"Fetching data for {market_name} ({ticker})...")
        
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
            
            # Mark market open/close
            data['MarketOpen'] = [is_market_open(market_name, dt) for dt in data.index]
            
            st.sidebar.success(f"Successfully loaded {len(data)} data points")
            return data
        else:
            st.sidebar.warning(f"No data found for {market_name}")
            return None
    except Exception as e:
        st.sidebar.error(f"Error fetching data: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return None

# Create features for model
def create_features(data):
    features = pd.DataFrame(index=data.index)
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['day_of_month'] = data.index.day
    features['month'] = data.index.month
    features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Market open feature
    if 'MarketOpen' in data.columns:
        features['market_open'] = data['MarketOpen'].astype(int)
    
    # Technical indicators
    for window in [3, 6, 12, 24, 48, 72]:
        features[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
        features[f'std_{window}'] = data['Close'].rolling(window=window).std()
    
    features['momentum_1h'] = data['Close'].diff(periods=1)
    features['momentum_6h'] = data['Close'].diff(periods=6)
    features['momentum_12h'] = data['Close'].diff(periods=12)
    features['momentum_24h'] = data['Close'].diff(periods=24)
    
    features['return_1h'] = data['Close'].pct_change(periods=1)
    features['return_6h'] = data['Close'].pct_change(periods=6)
    features['return_24h'] = data['Close'].pct_change(periods=24)
    
    # Volatility
    features['volatility_12h'] = features['return_1h'].rolling(window=12).std()
    features['volatility_24h'] = features['return_1h'].rolling(window=24).std()
    
    # Price difference from OHLC
    features['high_low_diff'] = (data['High'] - data['Low']) / data['Low']
    features['open_close_diff'] = (data['Close'] - data['Open']) / data['Open']
    
    # Lag features
    for i in range(1, 25):
        features[f'lag_{i}'] = data['Close'].shift(i)
    
    # Fill missing values
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# Make predictions for next trading day
def predict_next_trading_day(data, market_name):
    if data is None or len(data) < 48:  # Need enough data for training
        return None
    
    try:
        with st.spinner("Training model and generating predictions..."):
            # Create features
            features = create_features(data)
            feature_names = features.columns.tolist()
            
            # Ensure all features are numeric
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].fillna(0)
            
            # Train model
            X = features.values
            y = data['Close'].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Get next trading day timestamps
            next_open = get_next_trading_day(market_name)
            next_close = get_end_of_next_trading_day(market_name)
            
            if next_open is None or next_close is None:
                return None
            
            # Current price
            current_price = data['Close'].iloc[-1]
            
            # Generate predictions for next opening
            hours_to_opening = int((next_open - current_time).total_seconds() / 3600) + 1
            
            # Future dates for prediction
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(hours=1),
                periods=hours_to_opening,
                freq='1H'
            )
            
            # Make rolling predictions up to opening
            temp_data = data.copy()
            open_prediction = None
            
            for i, future_date in enumerate(future_dates):
                # Generate features
                temp_features = create_features(temp_data)
                
                # Ensure feature consistency
                for feat in feature_names:
                    if feat not in temp_features.columns:
                        temp_features[feat] = 0
                
                temp_features = temp_features[feature_names]
                temp_features = temp_features.fillna(0)
                
                # Make prediction
                X_new = scaler.transform(temp_features.values[-1:])
                prediction = model.predict(X_new)[0]
                
                # Add to temp data for next iteration
                new_row = pd.DataFrame({
                    'Open': [prediction],
                    'High': [prediction * 1.001],
                    'Low': [prediction * 0.999],
                    'Close': [prediction],
                    'Volume': [data['Volume'].mean()],
                    'MarketOpen': [is_market_open(market_name, future_date)]
                }, index=[future_date])
                
                temp_data = pd.concat([temp_data, new_row])
                
                # If this is the opening time, store prediction
                if i == hours_to_opening - 1:
                    open_prediction = prediction
            
            # Now predict closing
            hours_to_closing = int((next_close - next_open).total_seconds() / 3600) + 1
            
            # Future dates for prediction
            future_dates = pd.date_range(
                start=next_open,
                periods=hours_to_closing,
                freq='1H'
            )
            
            for future_date in future_dates:
                # Generate features
                temp_features = create_features(temp_data)
                
                # Ensure feature consistency
                for feat in feature_names:
                    if feat not in temp_features.columns:
                        temp_features[feat] = 0
                
                temp_features = temp_features[feature_names]
                temp_features = temp_features.fillna(0)
                
                # Make prediction
                X_new = scaler.transform(temp_features.values[-1:])
                prediction = model.predict(X_new)[0]
                
                # Add to temp data for next iteration
                new_row = pd.DataFrame({
                    'Open': [prediction * 0.999],
                    'High': [prediction * 1.001],
                    'Low': [prediction * 0.998],
                    'Close': [prediction],
                    'Volume': [data['Volume'].mean()],
                    'MarketOpen': [is_market_open(market_name, future_date)]
                }, index=[future_date])
                
                temp_data = pd.concat([temp_data, new_row])
            
            # Get close prediction
            close_prediction = temp_data['Close'].iloc[-1]
            
            # Feature importance
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Return predictions and metadata
            return {
                'current_price': current_price,
                'next_open_time': next_open,
                'next_open_prediction': open_prediction,
                'next_close_time': next_close,
                'next_close_prediction': close_prediction,
                'feature_importance': importance
            }
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Plot market data
def plot_market_data(data, market_name, predictions=None):
    fig = go.Figure()
    
    # Current time
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show only last 14 days
    display_start = current_time - timedelta(days=14)
    display_data = data[data.index >= display_start]
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['Close'],
        mode='lines',
        name=f'{market_name} Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add market open shading
    if 'MarketOpen' in display_data.columns:
        # Find periods when market is open
        open_periods = []
        start_open = None
        
        for i in range(len(display_data)):
            if display_data['MarketOpen'].iloc[i]:
                if start_open is None:
                    start_open = display_data.index[i]
            else:
                if start_open is not None:
                    end_open = display_data.index[i-1]
                    open_periods.append((start_open, end_open))
                    start_open = None
        
        # If still open at the end
        if start_open is not None:
            open_periods.append((start_open, display_data.index[-1]))
        
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
                xref="x",
                yref="paper"
            )
    
    # Add current time line
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
    
    fig.add_annotation(
        x=current_time_str,
        y=1,
        text="Current",
        showarrow=False,
        xref="x",
        yref="paper",
        font=dict(color="green", size=12)
    )
    
    # Add predictions if available
    if predictions:
        # Add opening prediction
        if 'next_open_time' in predictions and 'next_open_prediction' in predictions:
            open_time = predictions['next_open_time']
            open_price = predictions['next_open_prediction']
            
            fig.add_trace(go.Scatter(
                x=[open_time],
                y=[open_price],
                mode='markers',
                name='Predicted Opening',
                marker=dict(
                    color='orange',
                    size=10,
                    symbol='circle'
                )
            ))
            
            fig.add_annotation(
                x=open_time,
                y=open_price,
                text="Predicted Open",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="orange", size=10)
            )
        
        # Add closing prediction
        if 'next_close_time' in predictions and 'next_close_prediction' in predictions:
            close_time = predictions['next_close_time']
            close_price = predictions['next_close_prediction']
            
            fig.add_trace(go.Scatter(
                x=[close_time],
                y=[close_price],
                mode='markers',
                name='Predicted Closing',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle'
                )
            ))
            
            fig.add_annotation(
                x=close_time,
                y=close_price,
                text="Predicted Close",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40,
                font=dict(color="red", size=10)
            )
    
    # Update layout
    fig.update_layout(
        title=f"{market_name} - Historical Data and Predictions",
        xaxis_title="Date",
        yaxis_title=f"Price ({MARKETS[market_name]['currency']})",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Fetch data for selected market
if data_source == "Live Data":
    market_data = fetch_historical_data(selected_market)
    if market_data is None or market_data.empty:
        st.warning(f"Could not fetch live data for {selected_market}. Using demo data instead.")
        market_data = create_demo_data(selected_market)
else:
    market_data = create_demo_data(selected_market)

# Generate predictions
predictions = predict_next_trading_day(market_data, selected_market)

# Display market information
st.subheader(f"{selected_market} Market Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    # Plot data and predictions
    market_plot = plot_market_data(market_data, selected_market, predictions)
    st.plotly_chart(market_plot, use_container_width=True)

with col2:
    # Display market status
    is_open = is_market_open(selected_market)
    status = "OPEN" if is_open else "CLOSED"
    status_color = "green" if is_open else "red"
    
    st.markdown(f"### Market Status: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    
    # Display current price
    if market_data is not None and not market_data.empty:
        current_price = market_data['Close'].iloc[-1]
        currency = MARKETS[selected_market]['currency']
        st.markdown(f"### Current Price: {current_price:.2f} {currency}")
    
    # Display predictions
    if predictions:
        st.markdown("## Next Trading Day Predictions")
        
        # Format next trading day
        next_open_time = predictions['next_open_time']
        next_trading_day = format_trading_day(next_open_time)
        st.markdown(f"### Trading Day: {next_trading_day}")
        
        # Calculate changes
        current = predictions['current_price']
        open_pred = predictions['next_open_prediction']
        close_pred = predictions['next_close_prediction']
        
        open_change = ((open_pred - current) / current) * 100
        open_color = "green" if open_change > 0 else "red"
        
        close_change = ((close_pred - open_pred) / open_pred) * 100
        close_color = "green" if close_change > 0 else "red"
        
        # Display opening prediction
        st.markdown(f"#### Opening at {next_open_time.strftime('%H:%M')}:")
        st.markdown(f"<span style='color:{open_color}'>{open_pred:.2f} {currency} ({open_change:+.2f}%)</span>", unsafe_allow_html=True)
        
        # Display closing prediction
        next_close_time = predictions['next_close_time']
        st.markdown(f"#### Closing at {next_close_time.strftime('%H:%M')}:")
        st.markdown(f"<span style='color:{close_color}'>{close_pred:.2f} {currency} ({close_change:+.2f}%)</span>", unsafe_allow_html=True)
        
        # Overall prediction summary
        day_change = ((close_pred - current) / current) * 100
        day_color = "green" if day_change > 0 else "red"
        
        direction = "RISE" if day_change > 0 else "FALL"
        st.markdown(f"### Overall Prediction: <span style='color:{day_color}'>{direction}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{day_color}'>Expected change: {day_change:+.2f}%</span>", unsafe_allow_html=True)

# Display prediction factors
if predictions and 'feature_importance' in predictions:
    st.subheader("Key Factors Influencing the Prediction")
    
    # Get top 10 features
    top_features = predictions['feature_importance'].head(10)
    
    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features['Feature'],
        y=top_features['Importance'],
        marker=dict(color='royalblue')
    ))
    
    fig.update_layout(
        title="Top 10 Factors by Importance",
        xaxis_title="Factor",
        yaxis_title="Importance",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the most important factors
    st.subheader("Factor Explanations")
    
    factor_explanations = {
        'lag_': "Previous price values from N hours ago",
        'ma_': "Moving average over N hours",
        'std_': "Price volatility over N hours",
        'momentum_': "Price change over N hours",
        'return_': "Percentage return over N hours",
        'volatility_': "Volatility measured over N hours",
        'hour': "Hour of the day",
        'day_of_week': "Day of the week (0=Monday, 6=Sunday)",
        'day_of_month': "Day of the month",
        'month': "Month of the year",
        'is_weekend': "Whether the prediction is for a weekend",
        'market_open': "Whether the market is open",
        'high_low_diff': "Range between high and low prices",
        'open_close_diff': "Difference between opening and closing prices"
    }
    
    for _, row in top_features.head(5).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Find matching explanation
        explanation = "Technical indicator"
        for key, value in factor_explanations.items():
            if key in feature:
                explanation = value
                break
        
        st.markdown(f"**{feature}** (Importance: {importance:.4f}): {explanation}")

# Display recent data in expandable section
with st.expander("View Recent Data"):
    if market_data is not None and not market_data.empty:
        display_start = current_time - timedelta(days=3)  # Show last 3 days
        display_data = market_data[market_data.index >= display_start].copy()
        
        # Format for display
        display_data = display_data.reset_index()
        display_data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Open']
        display_data['Datetime'] = display_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_data.sort_values('Datetime', ascending=False))

# Footer
st.markdown("---")
st.write("This dashboard shows historical data and provides predictions for the next trading day.")
st.write("The predictions are based on historical patterns and relevant market factors shown above.")
st.write("Market opening and closing times are shaded in green on the chart.")
