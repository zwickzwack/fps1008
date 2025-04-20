import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Seitenkonfiguration
st.set_page_config(
    page_title="Finanzindex-Prognose",
    page_icon="📈",
    layout="wide"
)

# Debugging-Informationen anzeigen
st.sidebar.title("Debug-Informationen")
st.sidebar.write(f"Aktuelles Verzeichnis: {os.getcwd()}")

data_path = "data/raw"
if os.path.exists(data_path):
    files = os.listdir(data_path)
    st.sidebar.write(f"Dateien in data/raw: {files}")
else:
    st.sidebar.error(f"Verzeichnis {data_path} existiert nicht!")

# Funktion zum Laden von Daten
def load_data(index_name):
    """Lädt Daten für den ausgewählten Index"""
    # Prüfe im data/raw Verzeichnis nach Dateien
    pattern = f"{index_name}_*.csv"
    files = glob.glob(os.path.join("data/raw", pattern))
    
    if files:
        # Sortiere nach Erstellungszeit und wähle die neueste Datei
        latest_file = max(files, key=os.path.getctime)
        st.sidebar.success(f"Gefundene Datei: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file, parse_dates=True, index_col=0)
            return df
        except Exception as e:
            st.sidebar.error(f"Fehler beim Laden der Datei: {e}")
            return create_demo_data(index_name)
    else:
        st.sidebar.warning(f"Keine Dateien gefunden für {index_name}, verwende Demo-Daten")
        return create_demo_data(index_name)

def create_demo_data(index_name):
    """Erstellt Demo-Daten wenn keine Daten gefunden werden"""
    st.sidebar.info("Generiere Demo-Daten")
    
    # Zeitraum für Demo-Daten
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Unterschiedliche Startwerte je nach Index
    if index_name == "DAX":
        base_value = 18000
        volatility = 250
    elif index_name == "DowJones":
        base_value = 38000
        volatility = 400
    else:  # USD_EUR
        base_value = 0.92
        volatility = 0.01
    
    # Generiere einen realistischen Kursverlauf
    np.random.seed(42)  # Für Reproduzierbarkeit
    
    # Zufällige Bewegung mit gewisser Tendenz
    random_walk = np.random.normal(0, volatility/20, size=len(dates)).cumsum()
    trend = np.linspace(0, volatility/5, len(dates))
    values = base_value + random_walk + trend
    
    # DataFrame erstellen
    df = pd.DataFrame(index=dates)
    df['Close'] = values
    df['Open'] = df['Close'] * np.random.uniform(0.998, 1.002, size=len(df))
    df['High'] = np.maximum(df['Open'], df['Close']) * np.random.uniform(1.001, 1.005, size=len(df))
    df['Low'] = np.minimum(df['Open'], df['Close']) * np.random.uniform(0.995, 0.999, size=len(df))
    df['Volume'] = np.random.randint(1000, 10000, size=len(df))
    
    return df

# Hauptinhalt
st.title("Finanzindex-Prognose Dashboard")
st.markdown("""
    Dieses Dashboard visualisiert Finanzdaten und Prognosen für ausgewählte Indizes.
    Wählen Sie einen Index und einen Prognosehorizont aus der Seitenleiste.
""")

# Seitenleiste für Einstellungen
st.sidebar.title("Einstellungen")

# Index auswählen
index_options = ["DAX", "DowJones", "USD_EUR"]
selected_index = st.sidebar.selectbox(
    "Index auswählen:",
    options=index_options
)

# Prognosehorizont auswählen
horizon_options = {
    "1 Stunde": 1,
    "4 Stunden": 4, 
    "Handelsschluss (8h)": 8,
    "Nächster Handelsstart (24h)": 24
}
selected_horizon_name = st.sidebar.selectbox(
    "Prognosehorizont:",
    options=list(horizon_options.keys())
)
selected_horizon = horizon_options[selected_horizon_name]

# Daten laden
data = load_data(selected_index)

# Daten anzeigen
if data is not None:
    st.subheader(f"{selected_index} Kursdaten")
    
    # Letzte 7 Tage anzeigen
    cutoff_date = datetime.now() - timedelta(days=7)
    data_7d = data[data.index >= cutoff_date]
    
    # Daten visualisieren mit Plotly
    fig = go.Figure()
    
    # Füge Kurs hinzu
    price_col = 'Close' if 'Close' in data.columns else 'close'
    
    fig.add_trace(go.Scatter(
        x=data_7d.index,
        y=data_7d[price_col],
        mode='lines',
        name=f'{selected_index} Kurs'
    ))
    
    # Einfache Prognose basierend auf letztem Wert
    last_date = data_7d.index[-1]
    last_value = data_7d[price_col].iloc[-1]
    
    # Zufällige Änderung mit leichter Tendenz
    if selected_index == "USD_EUR":
        change_pct = np.random.normal(0.001 * selected_horizon, 0.005 * np.sqrt(selected_horizon/4))
    else:
        change_pct = np.random.normal(0.001 * selected_horizon, 0.01 * np.sqrt(selected_horizon/4))
    
    forecast_value = last_value * (1 + change_pct)
    forecast_date = last_date + timedelta(hours=selected_horizon)
    
    # Prognose hinzufügen
    fig.add_trace(go.Scatter(
        x=[last_date, forecast_date],
        y=[last_value, forecast_value],
        mode='lines+markers',
        line=dict(dash='dash', color='green' if forecast_value > last_value else 'red'),
        name=f'Prognose ({selected_horizon_name})'
    ))
    
    # Layout anpassen
    fig.update_layout(
        title=f"{selected_index} mit {selected_horizon_name} Prognose",
        xaxis_title="Datum",
        yaxis_title="Wert",
        legend=dict(orientation="h")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Details zur Prognose
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Aktueller Wert")
        st.metric(
            label=f"{selected_index} am {last_date.strftime('%d.%m.%Y %H:%M')}",
            value=f"{last_value:.2f}"
        )
    
    with col2:
        st.subheader("Prognose")
        direction = "↑" if forecast_value > last_value else "↓"
        change = forecast_value - last_value
        change_pct_display = change / last_value * 100
        
        st.metric(
            label=f"Prognose für {forecast_date.strftime('%d.%m.%Y %H:%M')}",
            value=f"{forecast_value:.2f}",
            delta=f"{change_pct_display:+.2f}%"
        )
    
    # Tabelle mit den letzten Werten
    st.subheader("Letzte Datenpunkte")
    st.dataframe(data_7d.tail(10))
else:
    st.error("Keine Daten verfügbar. Bitte überprüfen Sie die Datendateien.")
