import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Dashboard title
st.set_page_config(page_title="Financial Index Prediction", page_icon="📈", layout="wide")
st.title("📈 Financial Index Prediction Dashboard")

# Data directories
DATA_DIR = "data/raw"
MODELS_DIR = "data/models"

# Select index
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox("Select Index:", options=index_options)

# Load data
def load_data(index_name):
    files = glob.glob(os.path.join(DATA_DIR, f"{index_name}_*.csv"))
    if not files:
        st.error(f"No data found for {index_name}")
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
    
    # Check for models
    model_files = glob.glob(os.path.join(MODELS_DIR, f"{selected_index}_*.pkl"))
    model_files = [f for f in model_files if "_scaler.pkl" not in f and "_metrics.pkl" not in f]
    
    if model_files:
        st.subheader("Available Models")
        
        model_info = []
        for model_file in model_files:
            file_name = os.path.basename(model_file)
            parts = file_name.split('_')
            if len(parts) >= 4:
                model_type = parts[1]
                horizon = parts[2].replace('h', '')
                timestamp = '_'.join(parts[3:]).replace('.pkl', '')
                
                model_info.append({
                    'Model Type': model_type,
                    'Horizon (hours)': horizon,
                    'Created': datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')
                })
        
        st.dataframe(pd.DataFrame(model_info))
    else:
        st.info("No trained models found. Use the script to train models first.")
    
else:
    st.error("No data available. Please run data collection first.")
