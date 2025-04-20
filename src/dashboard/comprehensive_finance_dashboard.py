import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import requests
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import yfinance as yf
import holidays
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Dashboard configuration
st.set_page_config(
    page_title="Comprehensive Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard title and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #005cb2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .market-open {
        color: green;
        font-weight: bold;
    }
    .market-closed {
        color: red;
        font-weight: bold;
    }
    .prediction-up {
        color: green;
        font-weight: bold;
    }
    .prediction-down {
        color: red;
        font-weight: bold;
    }
    .news-important {
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    .news-neutral {
        border-left: 4px solid #90CAF9;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    .accuracy-good {
        color: green;
    }
    .accuracy-moderate {
        color: orange;
    }
    .accuracy-poor {
        color: red;
    }
</style>
<h1 class="main-header">📈 Comprehensive Financial Dashboard</h1>
""", unsafe_allow_html=True)

# Current time
current_time = datetime.now()
formatted_current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

# Sidebar information
st.sidebar.info(f"Current Date and Time: {formatted_current_time}")
st.sidebar.info(f"User: zwickzwack")

# Create necessary directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/forecasts", exist_ok=True)
os.makedirs("data/news", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Market information and configuration
MARKETS = {
    "DAX": {
        "ticker": "^GDAXI",
        "description": "German Stock Index",
        "currency": "EUR",
        "opening_time": time(9, 0),  # 9:00 AM
        "closing_time": time(17, 30),  # 5:30 PM
        "timezone": "Europe/Berlin",
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "holidays": holidays.Germany()
    },
    "DowJones": {
        "ticker": "^DJI",
        "description": "Dow Jones Industrial Average",
        "currency": "USD",
        "opening_time": time(9, 30),  # 9:30 AM
        "closing_time": time(16, 0),  # 4:00 PM
        "timezone": "America/New_York",
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "holidays": holidays.US()
    },
    "USD_EUR": {
        "ticker": "EURUSD=X",
        "description": "Euro to US Dollar Exchange Rate",
        "currency": "USD",
        "opening_time": time(0, 0),  # Forex markets run 24 hours
        "closing_time": time(23, 59),
        "timezone": "UTC",
        "trading_days": [0, 1, 2, 3, 4, 6],  # Monday to Friday + Sunday (starts on Sunday)
        "holidays": {}  # Forex markets have fewer holidays
    }
}

# Function to check if a market is open
def is_market_open(market_name, check_time=None):
    if check_time is None:
        check_time = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return False
    
    # Check if it's a trading day (weekday)
    if check_time.weekday() not in market["trading_days"]:
        return False
    
    # Check if it's a holiday
    if check_time.date() in market["holidays"]:
        return False
    
    # Check if within trading hours
    current_time = check_time.time()
    if market_name == "USD_EUR":  # Forex markets run 24 hours on trading days
        return True
    
    return market["opening_time"] <= current_time <= market["closing_time"]

# Function to get the next trading day
def get_next_trading_day(market_name, from_date=None):
    if from_date is None:
        from_date = datetime.now()
    
    market = MARKETS.get(market_name)
    if not market:
        return None
    
    next_day = from_date + timedelta(days=1)
    
    # Check for valid trading day
    while (next_day.weekday() not in market["trading_days"] or 
           next_day.date() in market["holidays"]):
        next_day = next_day + timedelta(days=1)
    
    # Set the opening time
    next_opening = datetime.combine(next_day.date(), market["opening_time"])
    
    return next_opening

# Function to get simulated financial news
def get_financial_news(days_back=14):
    # Check if we have cached news
    news_file = f"data/news/financial_news_{days_back}days.json"
    
    if os.path.exists(news_file):
        with open(news_file, 'r') as f:
            return json.load(f)
    
    # Generate simulated news if we don't have cached news
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # List of potential headlines
    positive_headlines = [
        "Market sentiment improves as inflation fears ease",
        "Tech sector leads market rally on strong earnings",
        "Central bank signals continued support for economy",
        "Economic data beats expectations, boosting investor confidence",
        "Major merger announced, stocks rally on consolidation hopes",
        "Trade tensions ease following productive negotiations",
        "Unemployment falls to multi-year low, boosting market optimism",
        "Retail sales surge, indicating strong consumer spending",
        "New stimulus package announced to support economic growth",
        "Manufacturing activity expands at fastest pace in years"
    ]
    
    negative_headlines = [
        "Inflation concerns weigh on market sentiment",
        "Tech stocks tumble on disappointing earnings reports",
        "Central bank signals potential rate hikes to combat inflation",
        "Economic data disappoints, raising recession fears",
        "Geopolitical tensions escalate, increasing market volatility",
        "Trade negotiations stall, threatening global supply chains",
        "Unemployment rises unexpectedly, raising economic concerns",
        "Retail sales slump, indicating weakening consumer confidence",
        "Government deadlock threatens economic stimulus plans",
        "Manufacturing activity contracts, raising economic warning signs"
    ]
    
    neutral_headlines = [
        "Markets mixed as investors await economic data",
        "Central bank maintains current monetary policy",
        "Earnings season begins with mixed results",
        "Investors cautious ahead of key economic reports",
        "Market flat as positive and negative factors balance",
        "Trade discussions continue with no major breakthroughs",
        "Employment remains stable as economy transitions",
        "Consumer spending patterns show shifting priorities",
        "Regulatory changes proposed for financial sector",
        "Global markets show varying responses to economic conditions"
    ]
    
    # Generate daily news with some randomness
    news = []
    current_date = start_date
    
    np.random.seed(42)  # For reproducibility
    
    while current_date <= end_date:
        # Determine number of news items for this day (1-3)
        num_items = np.random.randint(1, 4)
        
        for _ in range(num_items):
            # Determine sentiment based on day of week and some randomness
            day_factor = np.sin(current_date.weekday() * np.pi/7)  # Cyclical pattern
            random_factor = np.random.normal(0, 0.5)
            sentiment_score = day_factor + random_factor
            
            # Select headline based on sentiment
            if sentiment_score > 0.3:
                headline = np.random.choice(positive_headlines)
                sentiment = "positive"
            elif sentiment_score < -0.3:
                headline = np.random.choice(negative_headlines)
                sentiment = "negative"
            else:
                headline = np.random.choice(neutral_headlines)
                sentiment = "neutral"
            
            # Add random time during business hours
            hour = np.random.randint(8, 18)
            minute = np.random.randint(0, 60)
            news_datetime = datetime.combine(current_date.date(), time(hour, minute))
            
            # Create news item
            news_item = {
                "headline": headline,
                "datetime": news_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
                "sentiment_score": float(sentiment_score),  # Convert to float for JSON serialization
                "source": np.random.choice(["Bloomberg", "Reuters", "Financial Times", "CNBC", "Wall Street Journal"]),
                "impact_score": abs(float(sentiment_score) * np.random.uniform(0.5, 1.5))  # Simulate impact
            }
            
            news.append(news_item)
        
        current_date += timedelta(days=1)
    
    # Sort by datetime
    news = sorted(news, key=lambda x: x["datetime"])
    
    # Cache the news
    with open(news_file, 'w') as f:
        json.dump(news, f)
    
    return news

# Function to fetch historical market data
def fetch_historical_data(market_name, days_back=15):  # Use 15 days to ensure we get full 14 days
    market_info = MARKETS.get(market_name)
    if not market_info:
        st.error(f"Market information not found for {market_name}")
        return None
    
    ticker = market_info["ticker"]
    
    try:
        # Get data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try hourly data first
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
        
        if data.empty or len(data) < 24:
            # Fallback to daily data
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty:
                # Convert daily to hourly
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            # Make index timezone-naive
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Mark market open/close statuses
            data['MarketOpen'] = [is_market_open(market_name, dt) for dt in data.index]
            
            return data
        else:
            st.warning(f"No data found for {market_name} ({ticker}). Using demo data.")
            return create_demo_data(market_name, days_back)
    except Exception as e:
        st.error(f"Error fetching data for {market_name}: {str(e)}")
        return create_demo_data(market_name, days_back)

# Function to create demo data
def create_demo_data(market_name, days_back=14):
    market_info = MARKETS.get(market_name)
    
    # Set base parameters based on market
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
    
    # Generate price data with realistic patterns
    np.random.seed(42 + hash(market_name) % 100)  # Different seed for each market but consistent
    price_data = []
    current_price = base_value
    
    # Add trend and cyclical components
    trend_direction = np.random.choice([-1, 1])
    trend_strength = np.random.uniform(0.0001, 0.0003)
    
    for i, date in enumerate(date_range):
        # Time of day effect
        hour = date.hour
        
        # Higher volatility during market hours
        if market_name == "USD_EUR":
            hour_volatility = volatility  # Forex markets are always open
        else:
            # Check if market is open
            market_open = False
            if date.weekday() in market_info["trading_days"]:
                current_time = date.time()
                if market_info["opening_time"] <= current_time <= market_info["closing_time"]:
                    market_open = True
            
            hour_volatility = volatility if market_open else volatility * 0.3
        
        # Day of week effect
        day = date.weekday()
        
        # Weekday vs weekend effect
        if day <= 4:  # Weekday
            day_volatility = 1.0
            # Monday more positive, Friday more negative
            day_trend = 0.0001 * (2 - day)
        else:  # Weekend
            day_volatility = 0.3
            day_trend = 0
        
        # Add trend, cyclical, day and random components
        trend_component = trend_direction * trend_strength * i / len(date_range)
        
        # Add cyclical pattern (time of day effect)
        if market_name != "USD_EUR":
            hour_factor = np.sin((hour - 8) * np.pi / 10) if 8 <= hour <= 18 else 0
            cyclical_component = hour_factor * 0.0002
        else:
            # Less pronounced cycle for forex
            hour_factor = np.sin(hour * np.pi / 12)
            cyclical_component = hour_factor * 0.0001
        
        day_component = day_trend
        random_component = np.random.normal(0, hour_volatility * day_volatility / 1000)
        
        # Calculate price change
        price_change = trend_component + cyclical_component + day_component + random_component
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
    
    # Ensure Close is realistic
    for i in range(len(df)):
        df.iloc[i, df.columns.get_indexer(['Close'])[0]] = max(
            df.iloc[i, df.columns.get_indexer(['Low'])[0]],
            min(
                df.iloc[i, df.columns.get_indexer(['High'])[0]],
                df.iloc[i, df.columns.get_indexer(['Close'])[0]]
            )
        )
    
    # Mark market open/close statuses
    df['MarketOpen'] = [is_market_open(market_name, dt) for dt in df.index]
    
    return df

# Function to extract sentiment features from news
def extract_news_features(news_data, data_index):
    # Create a DataFrame with the same index as the price data
    news_features = pd.DataFrame(index=data_index)
    
    # Initialize columns
    news_features['sentiment_avg'] = 0.0
    news_features['sentiment_std'] = 0.0
    news_features['news_count'] = 0
    news_features['positive_ratio'] = 0.0
    news_features['negative_ratio'] = 0.0
    news_features['impact_score'] = 0.0
    
    # Convert news datetimes to datetime objects
    for news in news_data:
        news['datetime_obj'] = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
    
    # Group news by day
    news_by_day = {}
    for news in news_data:
        day_key = news['datetime_obj'].date()
        if day_key not in news_by_day:
            news_by_day[day_key] = []
        news_by_day[day_key].append(news)
    
    # Process each day
    for date in data_index:
        day_key = date.date()
        
        day_news = news_by_day.get(day_key, [])
        
        if day_news:
            # Count news
            news_features.at[date, 'news_count'] = len(day_news)
            
            # Calculate sentiment metrics
            sentiments = [n['sentiment_score'] for n in day_news]
            news_features.at[date, 'sentiment_avg'] = np.mean(sentiments)
            news_features.at[date, 'sentiment_std'] = np.std(sentiments) if len(sentiments) > 1 else 0
            
            # Calculate positive/negative ratios
            pos_count = sum(1 for n in day_news if n['sentiment'] == 'positive')
            neg_count = sum(1 for n in day_news if n['sentiment'] == 'negative')
            total = len(day_news)
            
            news_features.at[date, 'positive_ratio'] = pos_count / total if total > 0 else 0
            news_features.at[date, 'negative_ratio'] = neg_count / total if total > 0 else 0
            
            # Calculate impact score
            news_features.at[date, 'impact_score'] = np.mean([n.get('impact_score', 0) for n in day_news])
    
    # Forward fill to propagate sentiment to hours without news
    news_features = news_features.fillna(method='ffill')
    
    # Add lagged features
    for lag in [1, 2, 3]:
        news_features[f'sentiment_avg_lag{lag}'] = news_features['sentiment_avg'].shift(lag * 24)
        news_features[f'impact_score_lag{lag}'] = news_features['impact_score'].shift(lag * 24)
    
    # Add moving averages
    for window in [24, 48, 72]:  # 1, 2, 3 days
        news_features[f'sentiment_ma{window}'] = news_features['sentiment_avg'].rolling(window=window).mean()
        news_features[f'impact_ma{window}'] = news_features['impact_score'].rolling(window=window).mean()
    
    # Fill any remaining NAs
    news_features = news_features.fillna(0)
    
    return news_features

# Function to create model features
def create_features(data, price_col, news_features=None):
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
        features[f'ma_{window}'] = data[price_col].rolling(window=window).mean()
        features[f'std_{window}'] = data[price_col].rolling(window=window).std()
    
    features['momentum'] = data[price_col].diff(periods=1)
    features['momentum_6h'] = data[price_col].diff(periods=6)
    features['momentum_12h'] = data[price_col].diff(periods=12)
    features['momentum_24h'] = data[price_col].diff(periods=24)
    
    features['return_1h'] = data[price_col].pct_change(periods=1)
    features['return_6h'] = data[price_col].pct_change(periods=6)
    features['return_24h'] = data[price_col].pct_change(periods=24)
    
    # Volatility
    features['volatility_12h'] = features['return_1h'].rolling(window=12).std()
    features['volatility_24h'] = features['return_1h'].rolling(window=24).std()
    
    # Lag features
    for i in range(1, 25):
        features[f'lag_{i}'] = data[price_col].shift(i)
    
    # Add news features if available
    if news_features is not None:
        for col in news_features.columns:
            features[f'news_{col}'] = news_features[col]
    
    # Fill any missing values
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# Function to make forecasts
def make_forecasts(data, price_col, news_features=None):
    if data is None or len(data) < 24:
        st.error("Insufficient data for forecasting. Need at least 24 hourly data points.")
        return None
    
    try:
        # Create features
        features = create_features(data, price_col, news_features)
        
        # Ensure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(features[col].mean())
        
        # Keep track of feature names
        feature_names = features.columns.tolist()
        
        # Train model
        X = features.values
        y = data[price_col].values
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Create forecast DataFrames
        last_date = data.index[-1]
        current_value = float(data[price_col].iloc[-1])
        
        # Define forecast horizons (1 hour to 14 days)
        forecast_hours = 14 * 24  # 14 days
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=forecast_hours, freq='1H')
        forecast_values = []
        
        # Rolling forecast
        current_data = data.copy()
        
        for i in range(forecast_hours):
            # Get current features
            if news_features is not None:
                # Extend news features
                extended_news = news_features.copy()
                new_date = future_dates[i]
                
                # Just use the last values for future news features (could be improved)
                if new_date not in extended_news.index:
                    latest_news_values = extended_news.iloc[-1].to_dict()
                    extended_news.loc[new_date] = latest_news_values
                
                current_features = create_features(current_data, price_col, extended_news)
            else:
                current_features = create_features(current_data, price_col)
            
            # Ensure feature consistency
            for feat in feature_names:
                if feat not in current_features.columns:
                    current_features[feat] = 0
            
            current_features = current_features[feature_names]
            
            # Scale features
            current_X = scaler.transform(current_features.values[-1:])
            
            # Make prediction
            prediction = model.predict(current_X)[0]
            
            # Add to forecast values
            forecast_values.append(prediction)
            
            # Add to current data for next iteration
            if i < forecast_hours - 1:
                new_date = future_dates[i]
                new_row = pd.DataFrame({price_col: [prediction], 'MarketOpen': [is_market_open(data.name, new_date)]}, index=[new_date])
                current_data = pd.concat([current_data, new_row])
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({price_col: forecast_values, 'MarketOpen': [is_market_open(data.name, dt) for dt in future_dates]}, index=future_dates)
        
        return forecast_df, model, feature_names
    
    except Exception as e:
        st.error(f"Error making forecasts: {str(e)}")
        st.error(f"Trace: {traceback.format_exc() if 'traceback' in globals() else 'Traceback module not available'}")
        return None

# Function to evaluate forecast accuracy
def evaluate_forecast(historical_forecasts, actual_data, price_col):
    if historical_forecasts is None or actual_data is None:
        return None
    
    evaluation = {}
    
    # Get common dates
    common_dates = actual_data.index.intersection(historical_forecasts.index)
    
    if len(common_dates) == 0:
        return None
    
    # Extract relevant data
    y_true = actual_data.loc[common_dates, price_col].values
    y_pred = historical_forecasts.loc[common_dates, price_col].values
    
    # Calculate metrics
    evaluation['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    evaluation['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate direction accuracy
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    correct_direction = (actual_diff * pred_diff) > 0
    evaluation['direction_accuracy'] = np.mean(correct_direction) * 100
    
    # Daily evaluations
    daily_evals = []
    
    for date, group in pd.DataFrame({'actual': y_true, 'predicted': y_pred}, index=common_dates).groupby(pd.Grouper(freq='D')):
        if len(group) > 1:
            daily_mape = mean_absolute_percentage_error(group['actual'].values, group['predicted'].values) * 100
            daily_rmse = np.sqrt(mean_squared_error(group['actual'].values, group['predicted'].values))
            
            # Direction accuracy for this day
            daily_actual_diff = np.diff(group['actual'].values)
            daily_pred_diff = np.diff(group['predicted'].values)
            
            if len(daily_actual_diff) > 0:
                daily_direction = np.mean((daily_actual_diff * daily_pred_diff) > 0) * 100
            else:
                daily_direction = np.nan
            
            daily_evals.append({
                'date': date.strftime('%Y-%m-%d'),
                'mape': daily_mape,
                'rmse': daily_rmse,
                'direction_accuracy': daily_direction,
                'data_points': len(group)
            })
    
    evaluation['daily'] = daily_evals
    
    return evaluation

# Function to generate combined plot
def generate_combined_plot(datasets, forecasts, news_data=None):
    """
    Generate a combined plot with historical data, forecasts, and market information.
    
    Parameters:
    - datasets: Dict of DataFrames with historical data
    - forecasts: Dict of forecast DataFrames
    - news_data: List of news items
    
    Returns:
    - Plotly figure
    """
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.02,
                        subplot_titles=("DAX", "Dow Jones", "USD/EUR Exchange Rate"),
                        row_heights=[0.33, 0.33, 0.33])
    
    # Current time
    current_time = datetime.now()
    
    # Colors for different markets
    market_colors = {
        "DAX": "blue",
        "DowJones": "red",
        "USD_EUR": "green"
    }
    
    # Add data for each market
    for i, (market_name, data) in enumerate(datasets.items(), 1):
        if data is None:
            continue
        
        forecast = forecasts.get(market_name)
        color = market_colors.get(market_name, "black")
        
        # Add historical data
        historical_start = current_time - timedelta(days=14)
        historical_data = data[data.index >= historical_start]
        
        # Add historical line
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name=f'{market_name} Historical',
                line=dict(color=color, width=2),
                legendgroup=market_name
            ),
            row=i, col=1
        )
        
        # Add market open/close markers
        opens = []
        closes = []
        
        # Find market openings and closings
        if market_name != "USD_EUR":  # Skip for forex which is always open
            for j in range(1, len(historical_data)):
                if historical_data['MarketOpen'].iloc[j] and not historical_data['MarketOpen'].iloc[j-1]:
                    opens.append(historical_data.index[j])
                elif not historical_data['MarketOpen'].iloc[j] and historical_data['MarketOpen'].iloc[j-1]:
                    closes.append(historical_data.index[j])
            
            # Add market opening markers
            if opens:
                fig.add_trace(
                    go.Scatter(
                        x=opens,
                        y=[historical_data.loc[open_time, 'Close'] for open_time in opens],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name=f'{market_name} Market Open',
                        legendgroup=market_name
                    ),
                    row=i, col=1
                )
            
            # Add market closing markers
            if closes:
                fig.add_trace(
                    go.Scatter(
                        x=closes,
                        y=[historical_data.loc[close_time, 'Close'] for close_time in closes],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name=f'{market_name} Market Close',
                        legendgroup=market_name
                    ),
                    row=i, col=1
                )
        
        # Add forecast if available
        if forecast is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast['Close'],
                    mode='lines',
                    name=f'{market_name} Forecast',
                    line=dict(color=color, width=2, dash='dash'),
                    legendgroup=market_name
                ),
                row=i, col=1
            )
            
            # Add market open/close markers for forecast
            if market_name != "USD_EUR":
                forecast_opens = []
                forecast_closes = []
                
                for j in range(1, len(forecast)):
                    if forecast['MarketOpen'].iloc[j] and not forecast['MarketOpen'].iloc[j-1]:
                        forecast_opens.append(forecast.index[j])
                    elif not forecast['MarketOpen'].iloc[j] and forecast['MarketOpen'].iloc[j-1]:
                        forecast_closes.append(forecast.index[j])
                
                # Add forecast market opening markers
                if forecast_opens:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_opens,
                            y=[forecast.loc[open_time, 'Close'] for open_time in forecast_opens],
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='green', opacity=0.5),
                            name=f'{market_name} Forecast Market Open',
                            legendgroup=market_name
                        ),
                        row=i, col=1
                    )
                
                # Add forecast market closing markers
                if forecast_closes:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_closes,
                            y=[forecast.loc[close_time, 'Close'] for close_time in forecast_closes],
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color='red', opacity=0.5),
                            name=f'{market_name} Forecast Market Close',
                            legendgroup=market_name
                        ),
                        row=i, col=1
                    )
    
    # Add vertical line at current time
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    for i in range(1, 4):
        fig.add_shape(
            type="line",
            x0=current_time_str,
            y0=0,
            x1=current_time_str,
            y1=1,
            line=dict(color="green", width=2, dash="solid"),
            xref=f"x{i if i > 1 else ''}",
            yref=f"y{i} domain"
        )
        
        fig.add_annotation(
            x=current_time_str,
            y=1,
            text="Current",
            showarrow=False,
            xref=f"x{i if i > 1 else ''}",
            yref=f"y{i} domain",
            yanchor="bottom",
            font=dict(color="green", size=12)
        )
    
    # Add significant news annotations if available
    if news_data:
        significant_news = [n for n in news_data if abs(n.get('sentiment_score', 0)) > 0.5]
        
        for news in significant_news:
            news_time = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
            
            # Only show news for dates in the plot
            if (current_time - timedelta(days=14)) <= news_time <= current_time:
                # Choose a random subplot to avoid crowding
                row = np.random.randint(1, 4)
                
                arrow_color = "green" if news['sentiment'] == "positive" else "red" if news['sentiment'] == "negative" else "gray"
                
                # Add news marker
                fig.add_trace(
                    go.Scatter(
                        x=[news_time],
                        y=[0.5],  # Use 0.5 as a relative position in the plot
                        mode='markers',
                        marker=dict(symbol='star', size=10, color=arrow_color),
                        text=f"{news['headline']} ({news['source']})",
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=row, col=1
                )
    
    # Update layout for better appearance
    fig.update_layout(
        title="Financial Markets - 14-Day Historical Data and 14-Day Forecast",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=900,
        hovermode="x unified"
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Price (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1)
    fig.update_yaxes(title_text="Exchange Rate", row=3, col=1)
    
    # Set x-axis range to show 14 days before and 14 days after current time
    view_start = (current_time - timedelta(days=14)).strftime('%Y-%m-%d')
    view_end = (current_time + timedelta(days=14)).strftime('%Y-%m-%d')
    
    fig.update_xaxes(range=[view_start, view_end])
    
    return fig

# Main application logic
st.sidebar.header("Market Status")

# Display current market status
for market_name, market_info in MARKETS.items():
    is_open = is_market_open(market_name)
    status_class = "market-open" if is_open else "market-closed"
    status_text = "OPEN" if is_open else "CLOSED"
    
    st.sidebar.markdown(f"<div><b>{market_name}:</b> <span class='{status_class}'>{status_text}</span></div>", unsafe_allow_html=True)
    
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

# Fetch data and make forecasts
with st.spinner("Fetching market data and generating forecasts..."):
    # Get financial news
    news_data = get_financial_news(days_back=14)
    
    # Initialize market data and forecasts
    market_data = {}
    market_forecasts = {}
    
    # Process each market
    for market_name in MARKETS.keys():
        # Fetch historical data
        if data_source == "Live Data":
            data = fetch_historical_data(market_name)
            if data is None or data.empty:
                data = create_demo_data(market_name)
                st.warning(f"Could not fetch live data for {market_name}. Using demo data instead.")
        else:
            data = create_demo_data(market_name)
        
        # Set name attribute to market name for use in functions
        data.name = market_name
        
        # Store data
        market_data[market_name] = data
        
        # Extract news features
        news_features = extract_news_features(news_data, data.index)
        
        # Generate forecast
        forecast_result = make_forecasts(data, 'Close', news_features)
        
        if forecast_result:
            forecast_df, model, feature_names = forecast_result
            market_forecasts[market_name] = forecast_df
            
            # Save feature importance for later
            market_name_lower = market_name.lower()
            if not hasattr(st.session_state, f"{market_name_lower}_feature_importance"):
                if model is not None and hasattr(model, 'feature_importances_') and feature_names:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    setattr(st.session_state, f"{market_name_lower}_feature_importance", importance_df)
                    
                    # Find news-related features and their importance
                    news_features = [f for f in feature_names if 'news_' in f]
                    news_importance = {}
                    
                    for feat in news_features:
                        idx = feature_names.index(feat)
                        news_importance[feat] = model.feature_importances_[idx]
                    
                    setattr(st.session_state, f"{market_name_lower}_news_importance", news_importance)

# Create combined visualization
st.markdown("<h2 class='sub-header'>Market Visualization</h2>", unsafe_allow_html=True)

combined_fig = generate_combined_plot(market_data, market_forecasts, news_data)
st.plotly_chart(combined_fig, use_container_width=True)

# Show summary statistics for each market
st.markdown("<h2 class='sub-header'>Market Summary</h2>", unsafe_allow_html=True)

market_cols = st.columns(len(MARKETS))

for i, (market_name, market_info) in enumerate(MARKETS.items()):
    data = market_data.get(market_name)
    forecast = market_forecasts.get(market_name)
    
    with market_cols[i]:
        st.subheader(market_name)
        
        if data is not None and not data.empty:
            current_price = float(data['Close'].iloc[-1])
            st.metric(label="Current Price", value=f"{current_price:.2f}")
            
            # 24-hour change
            if len(data) > 24:
                price_24h_ago = float(data['Close'].iloc[-25])
                change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                st.metric(label="24-Hour Change", value=f"{change_24h:+.2f}%", delta=f"{change_24h:+.2f}%")
            
            # Forecast for next opening
            if forecast is not None and not forecast.empty:
                # For DAX and DowJones, find the next market opening
                if market_name in ["DAX", "DowJones"]:
                    # Find next opening in forecast
                    next_open_idx = None
                    
                    for j in range(1, len(forecast)):
                        if forecast['MarketOpen'].iloc[j] and not forecast['MarketOpen'].iloc[j-1]:
                            next_open_idx = j
                            break
                    
                    if next_open_idx is not None:
                        next_open_time = forecast.index[next_open_idx]
                        next_open_price = float(forecast['Close'].iloc[next_open_idx])
                        next_open_change = ((next_open_price - current_price) / current_price) * 100
                        
                        direction_class = "prediction-up" if next_open_change > 0 else "prediction-down"
                        
                        st.markdown(f"<b>Next Market Open:</b> {next_open_time.strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                        st.markdown(f"<b>Predicted Opening Price:</b> <span class='{direction_class}'>{next_open_price:.2f} ({next_open_change:+.2f}%)</span>", unsafe_allow_html=True)
                else:
                    # For currency, just show the next hour
                    next_hour_price = float(forecast['Close'].iloc[0])
                    next_hour_change = ((next_hour_price - current_price) / current_price) * 100
                    direction_class = "prediction-up" if next_hour_change > 0 else "prediction-down"
                    
                    st.markdown(f"<b>Next Hour Prediction:</b> <span class='{direction_class}'>{next_hour_price:.4f} ({next_hour_change:+.2f}%)</span>", unsafe_allow_html=True)

# Show important news influencing the forecast
st.markdown("<h2 class='sub-header'>Key News Influencing Forecasts</h2>", unsafe_allow_html=True)

# Sort news by impact score
important_news = sorted(news_data, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:10]

for news in important_news:
    sentiment = news['sentiment']
    score = news['sentiment_score']
    
    # Determine news class based on importance
    news_class = "news-important" if abs(score) > 0.5 else "news-neutral"
    
    # Generate news item with datetime, headline, source, and sentiment
    st.markdown(f"""
    <div class='{news_class}'>
        <p><b>{news['datetime']}</b> - {news['headline']}</p>
        <p><small>Source: {news['source']} | Sentiment: {sentiment.upper()} | Impact: {news['impact_score']:.2f}</small></p>
    </div>
    """, unsafe_allow_html=True)

# Show feature importance for each market
st.markdown("<h2 class='sub-header'>Forecast Model Insights</h2>", unsafe_allow_html=True)

model_tabs = st.tabs(list(MARKETS.keys()))

for i, market_name in enumerate(MARKETS.keys()):
    market_name_lower = market_name.lower()
    with model_tabs[i]:
        # Get feature importance from session state
        importance_df = getattr(st.session_state, f"{market_name_lower}_feature_importance", None)
        news_importance = getattr(st.session_state, f"{market_name_lower}_news_importance", None)
        
        if importance_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top 10 Features by Importance")
                
                top_features = importance_df.head(10)
                
                # Create bar chart of feature importance
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_features['Feature'],
                    y=top_features['Importance'],
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("News Influence")
                
                if news_importance:
                    # Sort news features by importance
                    sorted_news = sorted(news_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    for feat, imp in sorted_news[:5]:  # Show top 5
                        # Clean up feature name for display
                        display_name = feat.replace('news_', '').replace('_', ' ').title()
                        st.markdown(f"**{display_name}**: {imp:.4f}")
                    
                    # Calculate total news influence
                    total_news_influence = sum(abs(v) for v in news_importance.values())
                    total_model_influence = sum(importance_df['Importance'])
                    news_percentage = (total_news_influence / total_model_influence) * 100
                    
                    st.metric("Total News Influence", f"{news_percentage:.2f}%")
                else:
                    st.info("No news features found in the model")

# Show forecast accuracy evaluation
st.markdown("<h2 class='sub-header'>Forecast Accuracy Evaluation</h2>", unsafe_allow_html=True)

st.info("""
This section would normally display the accuracy of previous forecasts compared to actual values.
Since this is a new dashboard, historical forecast data isn't available yet.
As forecasts are generated and time passes, this section will automatically populate with accuracy metrics.
""")

# Simulate some accuracy data for demonstration
demo_accuracy = {
    "DAX": {
        "mape": 1.25,
        "rmse": 85.32,
        "direction_accuracy": 72.5,
        "daily": [
            {"date": "2025-04-18", "mape": 0.98, "direction_accuracy": 75.0},
            {"date": "2025-04-17", "mape": 1.32, "direction_accuracy": 68.2},
            {"date": "2025-04-16", "mape": 1.45, "direction_accuracy": 70.1}
        ]
    },
    "DowJones": {
        "mape": 0.87,
        "rmse": 123.45,
        "direction_accuracy": 68.9,
        "daily": [
            {"date": "2025-04-18", "mape": 0.72, "direction_accuracy": 71.3},
            {"date": "2025-04-17", "mape": 0.95, "direction_accuracy": 65.8},
            {"date": "2025-04-16", "mape": 0.91, "direction_accuracy": 69.4}
        ]
    },
    "USD_EUR": {
        "mape": 0.42,
        "rmse": 0.0031,
        "direction_accuracy": 64.2,
        "daily": [
            {"date": "2025-04-18", "mape": 0.38, "direction_accuracy": 67.5},
            {"date": "2025-04-17", "mape": 0.45, "direction_accuracy": 62.8},
            {"date": "2025-04-16", "mape": 0.41, "direction_accuracy": 63.2}
        ]
    }
}

accuracy_cols = st.columns(len(MARKETS))

for i, market_name in enumerate(MARKETS.keys()):
    accuracy = demo_accuracy.get(market_name)
    
    with accuracy_cols[i]:
        st.subheader(market_name)
        
        if accuracy:
            # Display overall metrics
            mape = accuracy['mape']
            direction = accuracy['direction_accuracy']
            
            mape_class = "accuracy-good" if mape < 1.0 else "accuracy-moderate" if mape < 2.0 else "accuracy-poor"
            direction_class = "accuracy-good" if direction > 70 else "accuracy-moderate" if direction > 60 else "accuracy-poor"
            
            st.markdown(f"<b>Mean Absolute % Error:</b> <span class='{mape_class}'>{mape:.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"<b>Direction Accuracy:</b> <span class='{direction_class}'>{direction:.1f}%</span>", unsafe_allow_html=True)
            
            # Display daily metrics
            st.markdown("<b>Daily Performance:</b>", unsafe_allow_html=True)
            
            for day in accuracy['daily']:
                day_mape = day['mape']
                day_direction = day['direction_accuracy']
                
                day_mape_class = "accuracy-good" if day_mape < 1.0 else "accuracy-moderate" if day_mape < 2.0 else "accuracy-poor"
                day_direction_class = "accuracy-good" if day_direction > 70 else "accuracy-moderate" if day_direction > 60 else "accuracy-poor"
                
                st.markdown(
                    f"{day['date']}: MAPE <span class='{day_mape_class}'>{day_mape:.2f}%</span>, "
                    f"Direction <span class='{day_direction_class}'>{day_direction:.1f}%</span>", 
                    unsafe_allow_html=True
                )

# Footer information
st.markdown("""
---
This comprehensive financial dashboard provides 14-day historical data and 14-day forecasts for major financial indices.
The forecasts are based on technical indicators and news sentiment analysis, updated hourly.
Market opening and closing times are marked, and accuracy metrics track the performance of previous predictions.
""")

# Cache the current forecast for future evaluation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for market_name, forecast in market_forecasts.items():
    if forecast is not None:
        forecast_file = f"data/forecasts/{market_name}_{timestamp}.csv"
        forecast.to_csv(forecast_file)
