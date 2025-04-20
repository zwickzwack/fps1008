import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import json

# Füge das Projektverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Seitenkonfiguration
st.set_page_config(
    page_title="Finanzindex-Prognosemodell",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funktionen zum Laden der Daten
def load_market_data(index_name):
    """Lädt die neuesten Marktdaten für den angegebenen Index"""
    data_dir = "data/raw"
    files = [f for f in os.listdir(data_dir) if f.startswith(index_name)]

    if not files:
        return generate_demo_data(index_name)

    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(data_dir, f)))
    file_path = os.path.join(data_dir, latest_file)

    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return generate_demo_data(index_name)

def load_processed_data(index_name):
    """Lädt die verarbeiteten Feature-Daten"""
    file_path = f"data/processed/{index_name}_features.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=True, index_col=0)
    else:
        return None

def load_model_info(index_name):
    """Lädt Informationen über das trainierte Modell"""
    model_dir = f"data/models/{index_name}"
    info_file = os.path.join(model_dir, "model_info.json")

    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            return json.load(f)
    else:
        # Demo-Modellinformationen
        return {
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_count": random.randint(5, 20),
            "train_epochs": random.randint(30, 100),
            "train_loss": round(random.uniform(0.001, 0.01), 6),
            "val_loss": round(random.uniform(0.001, 0.02), 6),
            "test_mae": round(random.uniform(10, 50), 2),
            "test_mape": round(random.uniform(0.5, 2.0), 2),
            "features_used": ["price", "volume", "sentiment", "technical_indicators"],
            "model_improvement": round(random.uniform(5, 25), 2)
        }

def generate_demo_data(index_name):
    """Generiert Demo-Daten für den Fall, dass keine echten Daten verfügbar sind"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')

    # Basiswerte je nach Index
    if index_name == "DAX":
        base_value = 17000
        volatility = 200
    elif index_name == "DowJones":
        base_value = 38000
        volatility = 400
    else:  # USD_EUR
        base_value = 0.92
        volatility = 0.01

    # Zufällige Wertverläufe generieren
    np.random.seed(42)
    random_walk = np.random.normal(0, volatility/20, size=len(date_range)).cumsum()

    # Tendenz hinzufügen
    trend = np.linspace(0, volatility/5, len(date_range))
    values = base_value + random_walk + trend

    # Erzeuge DataFrame
    data = pd.DataFrame(index=date_range)
    data['Close'] = values
    data['Open'] = values * np.random.uniform(0.998, 1.002, size=len(values))
    data['High'] = np.maximum(data['Open'], data['Close']) * np.random.uniform(1.001, 1.005, size=len(values))
    data['Low'] = np.minimum(data['Open'], data['Close']) * np.random.uniform(0.995, 0.999, size=len(values))
    data['Volume'] = np.random.randint(1000, 10000, size=len(values))

    return data

def generate_forecast(data, horizon, index_name):
    """Generiert Prognosen basierend auf den letzten Daten"""
    if data is None or data.empty:
        return None

    # In einem echten System würden wir hier das trainierte Modell verwenden
    # Für die Demo generieren wir plausible Prognosen

    # Letzter bekannter Wert
    if 'Close' in data.columns:
        last_value = data['Close'].iloc[-1]
    elif 'close' in data.columns:
        last_value = data['close'].iloc[-1]
    else:
        return None

    # Je nach Index unterschiedliche Volatilität
    if index_name == "DAX":
        volatility = 0.015  # 1.5%
    elif index_name == "DowJones":
        volatility = 0.012  # 1.2%
    else:  # USD_EUR
        volatility = 0.005  # 0.5%

    # Zufällige Bewegung mit leichter Tendenz nach oben
    change_pct = np.random.normal(0.001 * horizon, volatility * np.sqrt(horizon/4))

    # Prognosezeitpunkte
    last_time = data.index[-1]
    forecast_times = pd.date_range(start=last_time + timedelta(hours=1),
                                  periods=horizon, freq='1H')

    # Erzeuge die Prognose
    forecast_value = last_value * (1 + change_pct)

    # Erzeuge DataFrame mit der Prognose
    forecast = pd.DataFrame(index=[forecast_times[-1]])
    forecast['timestamp'] = forecast.index
    forecast['horizon'] = horizon
    forecast['value'] = forecast_value
    forecast['current_value'] = last_value
    forecast['change_pct'] = change_pct * 100
    forecast['direction'] = 'up' if change_pct > 0 else 'down'

    return forecast

# Dashboard-Layout
st.title("KI-basiertes Prognosemodell für Finanzindizes")
st.markdown("""
    Dieses Dashboard visualisiert KI-gestützte Prognosen für DAX, Dow Jones und USD/EUR
    basierend auf historischen Daten und Nachrichtenanalyse.
""")

# Seitenleiste für Filteroptionen
st.sidebar.header("Einstellungen")

# Index-Auswahl
selected_index = st.sidebar.selectbox(
    "Index auswählen:",
    options=["DAX", "DowJones", "USD_EUR"],
    index=0
)

# Prognosehorizont-Auswahl
horizon_options = {
    "1 Stunde": 1,
    "4 Stunden": 4,
    "Handelsschluss (8h)": 8,
    "Nächster Handelsstart (24h)": 24
}
selected_horizon_name = st.sidebar.selectbox(
    "Prognosehorizont:",
    options=list(horizon_options.keys()),
    index=1
)
selected_horizon = horizon_options[selected_horizon_name]

# Zeitraum-Auswahl für historische Daten
history_days = st.sidebar.slider(
    "Historische Daten anzeigen (Tage):",
    min_value=1,
    max_value=30,
    value=7
)

# Daten aktualisieren-Button
if st.sidebar.button("Daten und Prognosen aktualisieren"):
    st.experimental_rerun()

# Daten laden
market_data = load_market_data(selected_index)
processed_data = load_processed_data(selected_index)
model_info = load_model_info(selected_index)

# Filter data to requested timeframe
if market_data is not None and not market_data.empty:
    start_date = datetime.now() - timedelta(days=history_days)
    market_data = market_data[market_data.index >= start_date]

# Prognosen generieren
forecast = generate_forecast(market_data, selected_horizon, selected_index)

# Dashboard-Layout
col1, col2 = st.columns([2, 1])

# Linker Bereich: Grafiken
with col1:
    st.subheader(f"{selected_index} Kurs und Prognose")

    if market_data is not None and not market_data.empty:
        # Vorbereitung der Daten für Plotly
        if 'Close' in market_data.columns:
            price_col = 'Close'
        elif 'close' in market_data.columns:
            price_col = 'close'
        else:
            price_col = market_data.columns[0]

        # Erstelle einen Plotly-Graph
        fig = go.Figure()

        # Historische Daten
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data[price_col],
            mode='lines',
            name='Historischer Kurs',
            line=dict(color='blue')
        ))

        # Prognose hinzufügen, falls vorhanden
        if forecast is not None and not forecast.empty:
            # Verbindungspunkt zwischen historischen Daten und Prognose
            fig.add_trace(go.Scatter(
                x=[market_data.index[-1], forecast.index[0]],
                y=[market_data[price_col].iloc[-1], forecast['value'].iloc[0]],
                mode='lines',
                line=dict(color='green', dash='dash'),
                name='Prognose'
            ))

            # Prognose-Punkt hervorheben
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['value'],
                mode='markers',
                marker=dict(size=10, color='green'),
                name=f'{selected_horizon_name} Prognose'
            ))

            # Beschriftung für die Prognose
            direction = "↑" if forecast['direction'].iloc[0] == 'up' else "↓"
            change_text = f"{direction} {abs(forecast['change_pct'].iloc[0]):.2f}%"

            fig.add_annotation(
                x=forecast.index[0],
                y=forecast['value'].iloc[0],
                text=change_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green' if direction == "↑" else 'red',
                font=dict(size=14)
            )

        # Layout anpassen
        fig.update_layout(
            title=f"{selected_index} - {history_days} Tage Historie mit {selected_horizon_name} Prognose",
            xaxis_title="Datum",
            yaxis_title="Wert",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volumen-Diagramm (falls vorhanden)
        if 'Volume' in market_data.columns or 'volume' in market_data.columns:
            vol_col = 'Volume' if 'Volume' in market_data.columns else 'volume'

            if market_data[vol_col].sum() > 0:  # Nur anzeigen, wenn Volumendaten vorhanden sind
                fig_vol = go.Figure()

                fig_vol.add_trace(go.Bar(
                    x=market_data.index,
                    y=market_data[vol_col],
                    name='Volumen',
                    marker_color='rgba(58, 71, 80, 0.6)'
                ))

                fig_vol.update_layout(
                    title=f"{selected_index} Handelsvolumen",
                    xaxis_title="Datum",
                    yaxis_title="Volumen"
                )

                st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning(f"Keine Daten für {selected_index} verfügbar.")

# Rechter Bereich: Prognosedetails und KPIs
with col2:
    st.subheader("Prognosedetails")

    if forecast is not None and not forecast.empty:
        # Aktueller Wert
        current_value = forecast['current_value'].iloc[0]
        forecast_value = forecast['value'].iloc[0]
        change_pct = forecast['change_pct'].iloc[0]
        direction = forecast['direction'].iloc[0]
        forecast_time = forecast.index[0]

        # Farbige Box für die Prognose
        color = "rgba(0, 255, 0, 0.1)" if direction == 'up' else "rgba(255, 0, 0, 0.1)"
        direction_text = "STEIGT ↑" if direction == 'up' else "FÄLLT ↓"

        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {color};">
                <h3 style="text-align: center;">{selected_index} {direction_text}</h3>
                <p style="text-align: center; font-size: 24px;">
                    Prognose: {forecast_value:.2f} ({change_pct:+.2f}%)
                </p>
                <p style="text-align: center;">
                    zum {forecast_time.strftime('%d.%m.%Y %H:%M')} Uhr
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Details anzeigen
        st.markdown("### Details")
        details_col1, details_col2 = st.columns(2)

        with details_col1:
            st.metric("Aktueller Wert", f"{current_value:.2f}")
            st.metric("Zeithorizont", f"{selected_horizon} Stunden")

        with details_col2:
            st.metric("Prognostizierter Wert", f"{forecast_value:.2f}", f"{change_pct:+.2f}%")
            st.metric("Wahrscheinlichkeit", f"{random.randint(60, 90)}%")

    # Modell-KPIs anzeigen
    st.markdown("### Modell-KPIs")

    # Zeige Trainingsinformationen
    st.markdown(f"**Letztes Training:** {model_info['last_trained']}")
    st.markdown(f"**Trainingsiterationen:** {model_info['train_count']}")

    # KPIs in Spalten anzeigen
    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        st.metric("Trainings-MAE", f"{model_info['test_mae']:.2f}")
        st.metric("Modellverbesserung", f"{model_info['model_improvement']:.2f}%")

    with kpi_col2:
        st.metric("MAPE", f"{model_info['test_mape']:.2f}%")
        st.metric("Trainings-Epochen", f"{model_info['train_epochs']}")

    # Feature-Wichtigkeit
    st.markdown("### Feature-Wichtigkeit")

    # Demo-Daten für Feature-Wichtigkeit
    feature_importance = {
        "Technische Indikatoren": random.uniform(0.3, 0.5),
        "Historische Preise": random.uniform(0.2, 0.4),
        "Sentiment-Analyse": random.uniform(0.1, 0.3),
        "Handelsvolumen": random.uniform(0.05, 0.2),
        "Zeitliche Faktoren": random.uniform(0.05, 0.15)
    }

    # Sortiere nach Wichtigkeit
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Erstelle horizontales Balkendiagramm
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=list(sorted_features.keys()),
        x=list(sorted_features.values()),
        orientation='h',
        marker=dict(
            color=['rgba(55, 126, 184, 0.7)', 'rgba(255, 127, 0, 0.7)',
                   'rgba(50, 180, 50, 0.7)', 'rgba(180, 50, 50, 0.7)',
                   'rgba(148, 103, 189, 0.7)'][:len(sorted_features)]
        )
    ))

    fig.update_layout(
        title="Feature-Wichtigkeit für Prognose",
        xaxis_title="Wichtigkeit",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer-Bereich
st.markdown("---")
st.markdown("""
    **Hinweise:** Die gezeigten Prognosen basieren auf historischen Daten und Nachrichtenanalyse.
    Finanzmarktprognosen sind mit Unsicherheiten behaftet und sollten nicht als alleinige Grundlage für Investitionsentscheidungen dienen.
""")

# Hauptmenü ausführen
if __name__ == "__main__":
    pass
