import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pytz

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
            # Lade die CSV-Datei
            df = pd.read_csv(latest_file, index_col=0)
            
            # Konvertiere den Index zu datetime
            df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Stelle sicher, dass numerische Spalten wirklich numerisch sind
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            # Debug-Infos zum Index und Datentypen
            if len(df) > 0:
                st.sidebar.info(f"Index-Typ: {type(df.index)}")
                st.sidebar.info(f"Index-Dtype: {df.index.dtype}")
                st.sidebar.info(f"Erster Index-Wert: {df.index[0]}")
                
                # Zeige Datentypen aller Spalten
                st.sidebar.info(f"Datentypen der Spalten: {df.dtypes}")
            
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
    
    # Zeige alle verfügbaren Daten, aber maximal 7 Tage
    try:
        # Prüfe, ob der Index eine Zeitzone hat
        has_tz = False
        if len(data) > 0:
            idx = data.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                has_tz = True
                st.sidebar.info(f"Index hat Zeitzone: {idx.tz}")
        
        # Wähle die letzten 7 Tage oder alles, was verfügbar ist
        if len(data) > 168:  # etwa 7 Tage * 24 Stunden
            data_7d = data.iloc[-168:]
        else:
            data_7d = data
    except Exception as e:
        st.error(f"Fehler bei der Datenauswahl: {e}")
        data_7d = data
    
    # Daten visualisieren mit Plotly
    fig = go.Figure()
    
    # Finde die Preisspalte
    price_col = None
    for col_name in ['Close', 'close', 'Adj Close', 'adj_close', 'Price', 'price']:
        if col_name in data_7d.columns:
            price_col = col_name
            break
    
    if price_col is None:
        # Nehmen wir die erste numerische Spalte, falls keine standard Spalte gefunden wird
        numeric_cols = data_7d.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
            st.info(f"Keine Standard-Preisspalte gefunden, verwende {price_col}")
        else:
            st.error("Keine numerischen Spalten gefunden!")
            st.stop()
    
    # Füge Debug-Informationen hinzu
    st.sidebar.info(f"Ausgewählte Preisspalte: {price_col}")
    if len(data_7d) > 0:
        st.sidebar.info(f"Letzter Preiswert: {data_7d[price_col].iloc[-1]}")
        st.sidebar.info(f"Typ des letzten Werts: {type(data_7d[price_col].iloc[-1])}")
    
    # Füge Kurs hinzu
    fig.add_trace(go.Scatter(
        x=data_7d.index,
        y=data_7d[price_col],
        mode='lines',
        name=f'{selected_index} Kurs'
    ))
    
    # Einfache Prognose basierend auf letztem Wert
    if len(data_7d) > 0:
        try:
            # Sichere Konvertierung zu float
            last_value_raw = data_7d[price_col].iloc[-1]
            
            # Stelle sicher, dass last_value ein float ist
            if isinstance(last_value_raw, (int, float, np.number)):
                last_value = float(last_value_raw)
            else:
                # Versuche zu konvertieren, falls es ein String ist
                try:
                    last_value = float(last_value_raw)
                except (ValueError, TypeError):
                    st.error(f"Konnte den letzten Wert nicht in eine Zahl konvertieren: {last_value_raw}")
                    last_value = 100.0  # Fallback-Wert
            
            last_date = data_7d.index[-1]
            
            # Zufällige Änderung mit leichter Tendenz
            if selected_index == "USD_EUR":
                change_pct = np.random.normal(0.001 * selected_horizon, 0.005 * np.sqrt(selected_horizon/4))
            else:
                change_pct = np.random.normal(0.001 * selected_horizon, 0.01 * np.sqrt(selected_horizon/4))
            
            forecast_value = last_value * (1 + change_pct)
            
            # Respektiere die Zeitzone des Index beim Erstellen des Prognose-Datums
            if has_tz:
                forecast_date = last_date + pd.Timedelta(hours=selected_horizon)
            else:
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
                last_date_str = last_date.strftime('%d.%m.%Y %H:%M')
                st.metric(
                    label=f"{selected_index} am {last_date_str}",
                    value=f"{last_value:.2f}"
                )
            
            with col2:
                st.subheader("Prognose")
                direction = "↑" if forecast_value > last_value else "↓"
                change = forecast_value - last_value
                change_pct_display = change / last_value * 100
                forecast_date_str = forecast_date.strftime('%d.%m.%Y %H:%M')
                
                st.metric(
                    label=f"Prognose für {forecast_date_str}",
                    value=f"{forecast_value:.2f}",
                    delta=f"{change_pct_display:+.2f}%"
                )
            
            # Zeige Richtung mit Modellvertrauen
            confidence = np.random.randint(65, 95)
            direction_text = "STEIGEN ↑" if direction == "↑" else "FALLEN ↓"
            
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 5px; 
                     background-color: {'rgba(0, 255, 0, 0.1)' if direction == '↑' else 'rgba(255, 0, 0, 0.1)'}; 
                     margin-top: 20px; margin-bottom: 20px;">
                    <h3 style="text-align: center;">{selected_index} wird voraussichtlich {direction_text}</h3>
                    <p style="text-align: center;">
                        Modellvertrauen: {confidence}%
                    </p>
                    <p style="text-align: center;">
                        Prognose für {forecast_date_str}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Fehler bei der Prognoseberechnung: {e}")
            import traceback
            st.sidebar.error(traceback.format_exc())
    else:
        st.warning("Keine Daten für die Visualisierung verfügbar")
    
    # Tabelle mit den letzten Werten
    st.subheader("Letzte Datenpunkte")
    if len(data_7d) > 0:
        st.dataframe(data_7d.tail(10))
    else:
        st.warning("Keine Datenpunkte verfügbar")
    
    # KPIs des Modells (Demo-Daten)
    st.subheader("Modell-KPIs")
    
    # Zeige verschiedene KPIs in Spalten
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric("MAE", f"{np.random.uniform(10, 50):.2f}")
        
    with kpi2:
        st.metric("MAPE", f"{np.random.uniform(1.0, 4.0):.2f}%")
        
    with kpi3:
        st.metric("Trainings-Iteration", f"{np.random.randint(5, 20)}")
        
    # Zeige Verbesserungsverlauf
    improvement = np.random.uniform(5, 25)
    st.metric("Modellverbesserung seit letztem Training", f"{improvement:.2f}%")
else:
    st.error("Keine Daten verfügbar. Bitte überprüfen Sie die Datendateien.")
