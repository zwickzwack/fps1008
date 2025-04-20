#!/bin/bash

# Financial Prediction System - Model Tools
# Created: 2025-04-19
# Author: zwickzwack

# Set default directories
PROJECT_DIR="$HOME/financial-prediction-system"
DATA_DIR="$PROJECT_DIR/data/raw"
MODELS_DIR="$PROJECT_DIR/data/models"
LOG_DIR="$PROJECT_DIR/logs"
SCRIPTS_DIR="$PROJECT_DIR/src"

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$LOG_DIR"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "[$timestamp] [$level] $message"
    echo "[$timestamp] [$level] $message" >> "$LOG_DIR/model_tools.log"
}

# Function to check if Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Setup environment
setup_environment() {
    log "Setting up environment..." "SETUP"
    
    # Check and install required Python packages
    for package in "pandas" "numpy" "scikit-learn" "yfinance" "streamlit" "plotly" "holidays"; do
        if ! check_package "$package"; then
            log "Installing $package..." "SETUP"
            pip3 install "$package"
        fi
    done
    
    log "Environment setup complete" "SETUP"
}

# Function to collect financial data
collect_data() {
    log "Starting data collection..." "DATA"
    
    # Indices to collect
    indices=("DAX:^GDAXI" "DowJones:^DJI" "USD_EUR:EURUSD=X")
    
    # Current timestamp for filename
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    for index_pair in "${indices[@]}"; do
        IFS=":" read -r index_name ticker <<< "$index_pair"
        
        log "Collecting data for $index_name ($ticker)..." "DATA"
        
        # Python script to download data
        python3 -c "
import yfinance as yf
import pandas as pd
from datetime import datetime

try:
    # Download data (1 month hourly)
    data = yf.download('$ticker', period='1mo', interval='1h')
    
    if not data.empty:
        # Save to file
        data.to_csv('$DATA_DIR/${index_name}_${timestamp}.csv')
        print(f'Successfully downloaded {len(data)} records for $index_name')
    else:
        print('No data found for $index_name')
except Exception as e:
    print(f'Error downloading data: {e}')
"
        
        # Check if file was created
        if [ -f "$DATA_DIR/${index_name}_${timestamp}.csv" ]; then
            log "Data for $index_name saved to ${index_name}_${timestamp}.csv" "DATA"
            
            # Cleanup old files (keep only 5 most recent)
            file_count=$(ls -1 "$DATA_DIR/${index_name}_"*.csv 2>/dev/null | wc -l)
            if [ "$file_count" -gt 5 ]; then
                log "Cleaning up old files for $index_name..." "DATA"
                ls -t "$DATA_DIR/${index_name}_"*.csv | tail -n +6 | xargs rm -f
            fi
        else
            log "Failed to collect data for $index_name" "ERROR"
        fi
    done
    
    log "Data collection completed" "DATA"
}

# Function to train model
train_model() {
    local index="$1"
    local horizon="$2"
    local model_type="${3:-randomforest}"
    
    log "Training $model_type model for $index with $horizon hour horizon..." "TRAIN"
    
    # Find latest data file for the index
    latest_file=$(ls -t "$DATA_DIR/${index}_"*.csv 2>/dev/null | head -n 1)
    
    if [ -z "$latest_file" ]; then
        log "No data file found for $index" "ERROR"
        return 1
    fi
    
    log "Using data file: $latest_file" "TRAIN"
    
    # Python script to train model
    python3 -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import joblib
from datetime import datetime
import os

# Load data
try:
    data = pd.read_csv('$latest_file', index_col=0)
    data.index = pd.to_datetime(data.index)
    
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
        else:
            print('No numeric columns found in data')
            exit(1)
    
    # Prepare features
    window_size = 12
    horizon = $horizon
    
    # Extract price
    df = data[[price_col]].copy()
    
    # Create features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['ma_3'] = df[price_col].rolling(window=3).mean()
    df['ma_6'] = df[price_col].rolling(window=6).mean()
    df['ma_12'] = df[price_col].rolling(window=12).mean()
    df['momentum'] = df[price_col].diff(periods=1)
    df['rate_of_change'] = df[price_col].pct_change(periods=1) * 100
    df['volatility'] = df[price_col].rolling(window=window_size).std()
    
    # Lag features
    for i in range(1, window_size + 1):
        df[f'lag_{i}'] = df[price_col].shift(i)
    
    # Target
    df['target'] = df[price_col].shift(-horizon)
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    df = df.dropna()
    
    # Split features and target
    X = df.drop(['target', price_col], axis=1)
    y = df['target']
    
    # Train-test split
    train_size = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if '$model_type' == 'randomforest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    metrics = {
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'mse_train': mean_squared_error(y_train, y_pred_train),
        'mse_test': mean_squared_error(y_test, y_pred_test),
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
    }
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('$MODELS_DIR', exist_ok=True)
    
    model_file = f'$MODELS_DIR/$index_{model_type}_{horizon}h_{timestamp}.pkl'
    scaler_file = f'$MODELS_DIR/$index_{model_type}_{horizon}h_{timestamp}_scaler.pkl'
    metrics_file = f'$MODELS_DIR/$index_{model_type}_{horizon}h_{timestamp}_metrics.pkl'
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f'Model saved to {model_file}')
    print(f'MAE (test): {metrics[\"mae_test\"]:.4f}')
    print(f'R² (test): {metrics[\"r2_test\"]:.4f}')
    
except Exception as e:
    print(f'Error training model: {e}')
    exit(1)
"
    
    log "Model training completed for $index" "TRAIN"
}

# Function to perform backtesting
backtest_model() {
    local index="$1"
    local horizon="$2"
    local days="${3:-3}"
    
    log "Performing backtesting for $index with $horizon hour horizon ($days days)..." "BACKTEST"
    
    # Find latest data file for the index
    latest_file=$(ls -t "$DATA_DIR/${index}_"*.csv 2>/dev/null | head -n 1)
    
    if [ -z "$latest_file" ]; then
        log "No data file found for $index" "ERROR"
        return 1
    fi
    
    log "Using data file: $latest_file" "BACKTEST"
    
    # Python script for backtesting
    python3 -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load data
try:
    data = pd.read_csv('$latest_file', index_col=0)
    data.index = pd.to_datetime(data.index)
    
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
        else:
            print('No numeric columns found in data')
            exit(1)
    
    # Backtesting parameters
    window_size = 12
    horizon = $horizon
    days = $days
    
    # Prepare for backtesting
    backtest_start_idx = max(0, len(data) - days * 24)
    
    def prepare_features(df, price_col, window_size):
        result = df[[price_col]].copy()
        result['hour'] = result.index.hour
        result['day_of_week'] = result.index.dayofweek
        result['ma_3'] = result[price_col].rolling(window=3).mean()
        result['ma_6'] = result[price_col].rolling(window=6).mean()
        result['ma_12'] = result[price_col].rolling(window=12).mean()
        result['momentum'] = result[price_col].diff(periods=1)
        result['rate_of_change'] = result[price_col].pct_change(periods=1) * 100
        result['volatility'] = result[price_col].rolling(window=window_size).std()
        
        for i in range(1, window_size + 1):
            result[f'lag_{i}'] = result[price_col].shift(i)
            
        return result.fillna(method='bfill').fillna(method='ffill')
    
    # Initialize results
    predictions = []
    actuals = []
    timestamps = []
    
    # Walk-forward validation
    for i in range(backtest_start_idx, len(data) - horizon):
        # Get training data
        train_data = data.iloc[:i]
        
        # Prepare features
        df = prepare_features(train_data, price_col, window_size)
        
        # Skip if not enough data
        if len(df) < window_size * 2:
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
        current_data = data.iloc[i:i+1]
        current_features = prepare_features(
            pd.concat([train_data.iloc[-window_size:], current_data]), 
            price_col, 
            window_size
        )
        X_current = current_features.drop([price_col], axis=1).iloc[-1:].values
        X_current_scaled = scaler.transform(X_current)
        
        # Make prediction
        prediction = model.predict(X_current_scaled)[0]
        
        # Save results
        predictions.append(prediction)
        
        # Actual value (horizon steps ahead)
        if i + horizon < len(data):
            actual = data.iloc[i + horizon][price_col]
            actuals.append(actual)
            timestamps.append(data.index[i])
    
    # Create results dataframe
    if len(predictions) > 0 and len(actuals) > 0:
        results = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions[:len(actuals)],
            'actual': actuals
        })
        
        # Calculate metrics
        mae = mean_absolute_error(results['actual'], results['prediction'])
        mape = np.mean(np.abs((results['actual'] - results['prediction']) / results['actual'])) * 100
        
        # Calculate direction accuracy
        correct_direction = ((results['actual'] > results['actual'].shift(1)) & 
                           (results['prediction'] > results['prediction'].shift(1))) | \
                          ((results['actual'] < results['actual'].shift(1)) & 
                           (results['prediction'] < results['prediction'].shift(1)))
        
        direction_accuracy = correct_direction.mean() * 100
        
        # Save results to file
        results_file = f'$LOG_DIR/{index}_backtest_{horizon}h_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.csv'
        results.to_csv(results_file)
        
        print(f'Backtesting completed. Results saved to {results_file}')
        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape:.2f}%')
        print(f'Direction Accuracy: {direction_accuracy:.2f}%')
        print(f'Backtest period: {results.iloc[0][\"timestamp\"]} to {results.iloc[-1][\"timestamp\"]}')
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(results['actual'], label='Actual')
        plt.plot(results['prediction'], label='Predicted')
        plt.title(f'{index} - {horizon}h Forecast Backtest')
        plt.legend()
        
        plot_file = f'$LOG_DIR/{index}_backtest_{horizon}h_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.png'
        plt.savefig(plot_file)
        print(f'Plot saved to {plot_file}')
    else:
        print('Not enough data for backtesting')
    
except Exception as e:
    print(f'Error during backtesting: {e}')
    exit(1)
"
    
    log "Backtesting completed for $index" "BACKTEST"
}

# Function to start dashboard
start_dashboard() {
    log "Starting dashboard..." "DASHBOARD"
    
    # Check if streamlit is installed
    if ! check_package "streamlit"; then
        log "Installing streamlit..." "SETUP"
        pip3 install streamlit
    fi
    
    # Check if dashboard file exists
    dashboard_file="$SCRIPTS_DIR/dashboard/backtesting_dashboard.py"
    
    if [ ! -f "$dashboard_file" ]; then
        log "Dashboard file not found. Creating backtesting dashboard..." "DASHBOARD"
        
        # Create dashboard directory if it doesn't exist
        mkdir -p "$SCRIPTS_DIR/dashboard"
        
        # Create a backtesting dashboard
        cat > "$dashboard_file" << 'EOL'
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Dashboard title
st.set_page_config(page_title="Financial Index Backtesting", page_icon="📈", layout="wide")
st.title("📈 Financial Index Prediction with Backtesting")

# Data directories
DATA_DIR = "data/raw"
MODELS_DIR = "data/models"
LOG_DIR = "logs"

# Select index
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox("Select Index:", options=index_options)

# Load data
@st.cache_data
def load_data(index_name):
    files = glob.glob(os.path.join(DATA_DIR, f"{index_name}_*.csv"))
    if not files:
        return None
    
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

data = load_data(selected_index)

if data is not None:
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
        else:
            st.error("No numeric columns found in data")
            st.stop()
            
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
                    result = df[[price_col]].copy()
                    result['hour'] = result.index.hour
                    result['day_of_week'] = result.index.dayofweek
                    result['ma_3'] = result[price_col].rolling(window=3).mean()
                    result['ma_6'] = result[price_col].rolling(window=6).mean()
                    result['ma_12'] = result[price_col].rolling(window=12).mean()
                    result['momentum'] = result[price_col].diff(periods=1)
                    result['rate_of_change'] = result[price_col].pct_change(periods=1) * 100
                    result['volatility'] = result[price_col].rolling(window=window_size).std()
                    
                    for i in range(1, window_size + 1):
                        result[f'lag_{i}'] = result[price_col].shift(i)
                        
                    return result.fillna(method='bfill').fillna(method='ffill')
                
                # Prepare for backtesting
                backtest_start_idx = max(0, len(data) - backtest_days * 24)
                
                # Initialize results
                predictions = []
                actuals = []
                timestamps = []
                
                progress_bar = st.progress(0)
                
                # Walk-forward validation
                total_steps = len(data) - backtest_start_idx - horizon
                for i, idx in enumerate(range(backtest_start_idx, len(data) - horizon)):
                    # Update progress
                    progress = min(100, int(i / total_steps * 100))
                    progress_bar.progress(progress)
                    
                    # Get training data
                    train_data = data.iloc[:idx]
                    
                    # Prepare features
                    df = prepare_features(train_data, price_col, window_size)
                    
                    # Skip if not enough data
                    if len(df) < window_size * 2:
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
                    current_data = data.iloc[idx:idx+1]
                    current_features = prepare_features(
                        pd.concat([train_data.iloc[-window_size:], current_data]), 
                        price_col, 
                        window_size
                    )
                    X_current = current_features.drop([price_col], axis=1).iloc[-1:].values
                    X_current_scaled = scaler.transform(X_current)
                    
                    # Make prediction
                    prediction = model.predict(X_current_scaled)[0]
                    
                    # Save results
                    predictions.append(prediction)
                    
                    # Actual value (horizon steps ahead)
                    if idx + horizon < len(data):
                        actual = data.iloc[idx + horizon][price_col]
                        actuals.append(actual)
                        timestamps.append(data.index[idx])
                
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
                    
                    # Show detailed results
                    with st.expander("View Detailed Results"):
                        st.dataframe(results)
                else:
                    st.error("Not enough data for backtesting with the selected parameters")
        
        # Check if existing backtest results
        backtest_files = glob.glob(os.path.join(LOG_DIR, f"{selected_index}_backtest_*.csv"))
        
        if backtest_files:
            st.subheader("Previous Backtest Results")
            
            backtest_file = max(backtest_files, key=os.path.getctime)
            backtest_results = pd.read_csv(backtest_file)
            backtest_results['timestamp'] = pd.to_datetime(backtest_results['timestamp'])
            
            # Display summary
            st.info(f"Showing results from: {os.path.basename(backtest_file)}")
            
            # Plot results
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=backtest_results['timestamp'],
                y=backtest_results['actual'],
                mode='lines',
                name='Actual'
            ))
            
            fig.add_trace(go.Scatter(
                x=backtest_results['timestamp'],
                y=backtest_results['prediction'],
                mode='lines',
                name='Predicted'
            ))
            
            fig.update_layout(
                title="Previous Backtest: Actual vs Predicted Values",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
else:
    st.error("No data available. Please run data collection first.")
    
    with st.expander("How to collect data"):
        st.code("""
        # Run data collection
        ./financial_model_tools.sh collect
        
        # Train a model
        ./financial_model_tools.sh train DAX 4 randomforest
        
        # Run backtesting
        ./financial_model_tools.sh backtest DAX 4 3
        """)
EOL
    fi
    
    log "Starting Streamlit dashboard in the current terminal..." "DASHBOARD"
    
    # Start streamlit directly in the current terminal
    cd "$PROJECT_DIR" && streamlit run "$dashboard_file"
    
    log "Dashboard started" "DASHBOARD"
}

# Show help message
show_help() {
    echo -e "${BLUE}Financial Prediction System - Model Tools${NC}"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  setup       Install dependencies and prepare environment"
    echo "  collect     Collect financial data"
    echo "  train       Train a prediction model"
    echo "  backtest    Run backtesting on historical data"
    echo "  dashboard   Start the prediction dashboard"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 setup                           # Setup environment"
    echo "  $0 collect                         # Collect data for all indices"
    echo "  $0 train DAX 4 randomforest       # Train model for DAX with 4h horizon"
    echo "  $0 backtest DowJones 8 5          # Backtest DowJones with 8h horizon, 5 days"
    echo "  $0 dashboard                       # Start dashboard"
    echo
}

# Main script logic
case "$1" in
    setup)
        setup_environment
        ;;
    collect)
        collect_data
        ;;
    train)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Missing index name${NC}"
            echo "Usage: $0 train INDEX HORIZON [MODEL_TYPE]"
            echo "Example: $0 train DAX 4 randomforest"
            exit 1
        fi
        
        if [ -z "$3" ]; then
            echo -e "${RED}Error: Missing horizon value${NC}"
            echo "Usage: $0 train INDEX HORIZON [MODEL_TYPE]"
            echo "Example: $0 train DAX 4 randomforest"
            exit 1
        fi
        
        train_model "$2" "$3" "${4:-randomforest}"
        ;;
    backtest)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Missing index name${NC}"
            echo "Usage: $0 backtest INDEX HORIZON [DAYS]"
            echo "Example: $0 backtest DAX 4 3"
            exit 1
        fi
        
        if [ -z "$3" ]; then
            echo -e "${RED}Error: Missing horizon value${NC}"
            echo "Usage: $0 backtest INDEX HORIZON [DAYS]"
            echo "Example: $0 backtest DAX 4 3"
            exit 1
        fi
        
        backtest_model "$2" "$3" "${4:-3}"
        ;;
    dashboard)
        start_dashboard
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        ;;
esac
