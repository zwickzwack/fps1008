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

# Dashboard title
st.set_page_config(page_title="Financial Index Backtesting", page_icon="📈", layout="wide")
st.title("📈 Financial Index Prediction with Backtesting")

# Data directories
DATA_DIR = "data/raw"
MODELS_DIR = "data/models"
LOG_DIR = "logs"

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Select index
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox("Select Index:", options=index_options)

# Create demo data
def create_demo_data(index_name):
    st.sidebar.warning(f"Creating demo data for {index_name}")
    
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
    
    # Generate dates for the past 30 days with hourly intervals
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
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

# Load data
@st.cache_data
def load_data(index_name):
    try:
        files = glob.glob(os.path.join(DATA_DIR, f"{index_name}_*.csv"))
        if not files:
            st.sidebar.warning(f"No data files found for {index_name}")
            return create_demo_data(index_name)
        
        latest_file = max(files, key=os.path.getctime)
        st.sidebar.info(f"Loading data from: {os.path.basename(latest_file)}")
        
        try:
            # Try to load the CSV file
            df = pd.read_csv(latest_file)
            
            # Check if there's a proper datetime index column
            if 'Unnamed: 0' in df.columns:
                # This is likely the datetime index
                df = df.rename(columns={'Unnamed: 0': 'date'})
                df = df.set_index('date')
            
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                st.sidebar.error(f"Failed to parse dates: {str(e)}")
                # If we can't parse the index, create a new one
                df = df.reset_index(drop=True)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(df) // 24)
                df.index = pd.date_range(start=start_date, end=end_date, periods=len(df))
            
            # Ensure numeric data types for all columns that should be numeric
            for col in df.columns:
                # Try to convert columns to numeric, coerce errors to NaN
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            
            # Check if we have any numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.sidebar.error("No numeric columns found in the data!")
                return create_demo_data(index_name)
            
            # Log the dtypes for debugging
            st.sidebar.info(f"Column data types: {df.dtypes}")
            
            return df
            
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")
            st.sidebar.error(traceback.format_exc())
            return create_demo_data(index_name)
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        return create_demo_data(index_name)

# Load the data
data = load_data(selected_index)

# Find price column
price_col = None
for col in ['Close', 'close', 'Adj Close', 'adj_close', 'Price', 'price']:
    if col in data.columns:
        price_col = col
        break

if price_col is None:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        price_col = numeric_cols[0]
        st.sidebar.info(f"Using {price_col} as price column")
    else:
        st.error("No numeric columns found in data")
        st.stop()

# Display data type information
st.sidebar.info(f"Price column: {price_col}")
st.sidebar.info(f"Price column type: {data[price_col].dtype}")
        
# Tabs
tab1, tab2 = st.tabs(["Historical Data", "Backtesting"])

with tab1:
    # Plot data
    st.subheader(f"{selected_index} Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[price_col],
        mode='lines',
        name=selected_index
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show latest data
    st.subheader("Latest Data")
    st.dataframe(data.tail(10))
    
    # Show data info
    with st.expander("Data Information"):
        st.write("Data Shape:", data.shape)
        st.write("Data Types:")
        st.write(data.dtypes)
        st.write("Data Statistics:")
        st.write(data.describe())

with tab2:
    st.subheader("Backtesting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.selectbox("Forecast Horizon (hours):", options=[1, 4, 8, 24], index=1)
    
    with col2:
        window_size = st.slider("Lookback Window (hours):", min_value=6, max_value=48, value=12)
        
    with col3:
        backtest_days = st.slider("Backtest Period (days):", min_value=1, max_value=14, value=3)
    
    if st.button("Run Backtesting"):
        with st.spinner("Running backtesting..."):
            # Functions for backtesting
            def prepare_features(df, price_col, window_size):
                try:
                    # Make sure we're working with a copy
                    result = df.copy()
                    
                    # Ensure the price column is numeric
                    result[price_col] = pd.to_numeric(result[price_col], errors='coerce')
                    
                    # Debug info
                    st.sidebar.info(f"Price column after conversion: {result[price_col].dtype}")
                    
                    # Create features
                    result['hour'] = result.index.hour
                    result['day_of_week'] = result.index.dayofweek
                    
                    # Calculate moving averages
                    result['ma_3'] = result[price_col].rolling(window=3).mean()
                    result['ma_6'] = result[price_col].rolling(window=6).mean()
                    result['ma_12'] = result[price_col].rolling(window=12).mean()
                    
                    # Calculate momentum indicators
                    result['momentum'] = result[price_col].diff(periods=1)
                    result['rate_of_change'] = result[price_col].pct_change(periods=1) * 100
                    result['volatility'] = result[price_col].rolling(window=window_size).std()
                    
                    # Create lag features
                    for i in range(1, window_size + 1):
                        result[f'lag_{i}'] = result[price_col].shift(i)
                    
                    # Fill missing values
                    result = result.fillna(method='bfill').fillna(method='ffill')
                    
                    return result
                except Exception as e:
                    st.error(f"Error preparing features: {str(e)}")
                    st.error(traceback.format_exc())
                    return None
            
            # Prepare for backtesting
            backtest_start_idx = max(0, len(data) - backtest_days * 24)
            
            # Initialize results
            predictions = []
            actuals = []
            timestamps = []
            
            progress_bar = st.progress(0)
            
            # Ensure we have numeric data
            try:
                # Convert to numeric again to be sure
                numeric_data = data.copy()
                numeric_data[price_col] = pd.to_numeric(numeric_data[price_col], errors='coerce')
                
                # Walk-forward validation
                total_steps = len(numeric_data) - backtest_start_idx - horizon
                for i, idx in enumerate(range(backtest_start_idx, len(numeric_data) - horizon)):
                    # Update progress
                    progress = min(100, int(i / total_steps * 100))
                    progress_bar.progress(progress)
                    
                    # Get training data
                    train_data = numeric_data.iloc[:idx]
                    
                    # Prepare features
                    df = prepare_features(train_data, price_col, window_size)
                    
                    if df is None or len(df) < window_size * 2:
                        continue
                    
                    # Split features and target for training
                    X = df.drop([price_col], axis=1)
                    y = df[price_col]
                    
                    # Train a simple model
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_scaled, y)
                    
                    # Prepare the current data point for prediction
                    current_data = numeric_data.iloc[idx:idx+1]
                    current_features = prepare_features(
                        pd.concat([train_data.iloc[-window_size:], current_data]), 
                        price_col, 
                        window_size
                    )
                    
                    if current_features is None:
                        continue
                        
                    X_current = current_features.drop([price_col], axis=1).iloc[-1:].values
                    X_current_scaled = scaler.transform(X_current)
                    
                    # Make prediction
                    prediction = model.predict(X_current_scaled)[0]
                    
                    # Save results
                    predictions.append(prediction)
                    
                    # Actual value (horizon steps ahead)
                    if idx + horizon < len(numeric_data):
                        actual = numeric_data.iloc[idx + horizon][price_col]
                        actuals.append(actual)
                        timestamps.append(numeric_data.index[idx])
                
                # Create results dataframe
                if len(predictions) > 0 and len(actuals) > 0:
                    results = pd.DataFrame({
                        'timestamp': timestamps,
                        'prediction': predictions[:len(actuals)],
                        'actual': actuals
                    })
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_absolute_error
                    mae = mean_absolute_error(results['actual'], results['prediction'])
                    mape = np.mean(np.abs((results['actual'] - results['prediction']) / results['actual'])) * 100
                    
                    # Calculate direction accuracy
                    correct_direction = ((results['actual'] > results['actual'].shift(1)) & 
                                      (results['prediction'] > results['prediction'].shift(1))) | \
                                      ((results['actual'] < results['actual'].shift(1)) & 
                                       (results['prediction'] < results['prediction'].shift(1)))
                    
                    direction_accuracy = correct_direction.mean() * 100
                    
                    # Display metrics
                    st.subheader("Backtesting Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mae:.4f}")
                    with col2:
                        st.metric("MAPE", f"{mape:.2f}%")
                    with col3:
                        st.metric("Direction Accuracy", f"{direction_accuracy:.2f}%")
                    
                    # Plot results
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=results['timestamp'],
                        y=results['actual'],
                        mode='lines',
                        name='Actual'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=results['timestamp'],
                        y=results['prediction'],
                        mode='lines',
                        name='Predicted'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_index} - {horizon}h Horizon Forecast Backtest",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot error
                    results['error'] = results['actual'] - results['prediction']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=results['timestamp'],
                        y=results['error'],
                        name='Prediction Error',
                        marker_color=np.where(results['error'] >= 0, 'green', 'red')
                    ))
                    
                    fig.update_layout(
                        title="Prediction Error (Actual - Predicted)",
                        xaxis_title="Date",
                        yaxis_title="Error",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.subheader("Feature Importance")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=feature_importance['Feature'],
                            y=feature_importance['Importance'],
                            marker_color='darkblue'
                        ))
                        
                        fig.update_layout(
                            title="Feature Importance in Prediction Model",
                            xaxis_title="Feature",
                            yaxis_title="Importance",
                            xaxis={'categoryorder': 'total descending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    with st.expander("View Detailed Results"):
                        st.dataframe(results)
                        
                    # Save results
                    try:
                        results_file = f"{LOG_DIR}/{selected_index}_backtest_{horizon}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        results.to_csv(results_file)
                        st.success(f"Results saved to {results_file}")
                    except Exception as e:
                        st.warning(f"Failed to save results: {e}")
                else:
                    st.error("Not enough data for backtesting with the selected parameters")
            except Exception as e:
                st.error(f"Error during backtesting: {str(e)}")
                st.error(traceback.format_exc())
    
    st.subheader("Backtesting Explanation")
    st.markdown("""
    **What is backtesting?**
    
    Backtesting evaluates how well a prediction model would have performed in the past. It simulates making predictions at historical points in time, then compares those predictions to what actually happened.
    
    **How it works:**
    
    1. **Window-based Training**: For each point in the backtest period, the model is trained using only data available up to that point.
    2. **Forward Prediction**: The model predicts the price at a future time (based on selected horizon).
    3. **Error Calculation**: Actual values are compared with predictions to calculate accuracy metrics.
    
    **Key Metrics:**
    
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage difference
    - **Direction Accuracy**: How often the model correctly predicts price movement direction (up/down)
    
    Adjust the parameters to see how they affect prediction accuracy.
    """)

    with st.expander("Tips for Better Predictions"):
        st.markdown("""
        - **Lookback Window**: 12-24 hours typically provides good results
        - **Forecast Horizon**: Shorter horizons (1-4 hours) are generally more accurate
        - **Backtest Period**: Longer periods (7+ days) give more robust accuracy measurements
        - **Feature Engineering**: The model automatically creates technical indicators like moving averages
        
        Remember that past performance doesn't guarantee future results!
        """)
