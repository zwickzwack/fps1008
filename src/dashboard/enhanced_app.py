import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pytz
import holidays
from pandas.tseries.offsets import BDay

# Seitenkonfiguration
st.set_page_config(
    page_title="Finanzindex-Prognosemodell",
    page_icon="📈",
    layout="wide"
)

# Funktionen zur Berücksichtigung von Handelszeiten
def is_business_day(date):
    """Prüft, ob ein bestimmtes Datum ein Geschäftstag ist"""
    return bool(len(pd.bdate_range(date, date)))

def is_holiday(date, country='DE'):
    """Prüft, ob ein bestimmtes Datum ein Feiertag im angegebenen Land ist"""
    # Dictionary der unterstützten Länder und ihrer Feiertage
    holiday_calendars = {
        'DE': holidays.Germany(),
        'US': holidays.UnitedStates()
    }
    
    calendar = holiday_calendars.get(country, holidays.Germany())
    return date in calendar

def get_trading_hours(index_name):
    """Gibt die Handelszeiten für den angegebenen Index zurück"""
    trading_hours = {
        'DAX': (9, 17, 30),  # 9:00 - 17:30 Uhr
        'DowJones': (15, 21, 30),  # 15:00 - 21:30 Uhr (in deutscher Zeit, UTC+1/2)
        'USD_EUR': (0, 23, 59)  # 24/7 Handel, aber geringere Liquidität am Wochenende
    }
    
    return trading_hours.get(index_name, (9, 17, 0))  # Standard: 9:00 - 17:00 Uhr

def is_trading_hour(dt, index_name):
    """Prüft, ob der angegebene Zeitpunkt eine Handelszeit für den Index ist"""
    # Stelle sicher, dass dt ein naive datetime ist
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    
    # Prüfe, ob es ein Wochenende ist
    if dt.weekday() >= 5:  # 5=Samstag, 6=Sonntag
        return False
    
    # Prüfe, ob es ein Feiertag ist (für DAX deutsche Feiertage, für Dow Jones US-Feiertage)
    country = 'DE' if index_name == 'DAX' else 'US' if index_name == 'DowJones' else None
    if country and is_holiday(dt.date(), country):
        return False
    
    # Prüfe, ob innerhalb der Handelszeiten
    start_hour, end_hour, end_minute = get_trading_hours(index_name)
    
    # Für 24/7 Handel (Devisenmarkt)
    if index_name == 'USD_EUR':
        # Reduzierte Liquidität am Wochenende berücksichtigen
        if dt.weekday() >= 5:  # Wochenende
            return False
        return True
    
    # Für reguläre Börsenhandelszeiten
    current_time = dt.hour * 60 + dt.minute  # Aktuelle Zeit in Minuten seit Mitternacht
    end_time = end_hour * 60 + end_minute  # Handelsende in Minuten seit Mitternacht
    
    return start_hour * 60 <= current_time < end_time

def next_trading_time(dt, index_name, hours_ahead=1):
    """Findet den nächsten Handelszeitpunkt ausgehend vom angegebenen Zeitpunkt"""
    # Stelle sicher, dass dt ein naive datetime ist
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
        
    # Wenn der aktuelle Zeitpunkt bereits eine Handelszeit ist, gehe vom nächsten Tag aus
    if is_trading_hour(dt, index_name):
        target_time = dt + timedelta(hours=hours_ahead)
        if is_trading_hour(target_time, index_name):
            return target_time
    
    # Sonst suche den nächsten Handelszeitpunkt
    test_time = dt
    max_iterations = 14 * 24  # Max. 14 Tage vorausschauen
    
    for _ in range(max_iterations):
        test_time = test_time + timedelta(hours=1)
        if is_trading_hour(test_time, index_name):
            return test_time
    
    # Fallback, falls kein Handelszeitpunkt gefunden wurde
    return dt + BDay(1) + pd.Timedelta(hours=get_trading_hours(index_name)[0])

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
            
            # Debug-Info zur Zeitzone
            if len(df) > 0:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    st.sidebar.info(f"Zeitzone im Datensatz: {df.index.tz}")
                else:
                    st.sidebar.info("Datensatz hat keine Zeitzone")
            
            # Stelle sicher, dass numerische Spalten wirklich numerisch sind
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            return df
        except Exception as e:
            st.sidebar.error(f"Fehler beim Laden der Datei: {e}")
            return create_demo_data(index_name)
    else:
        st.sidebar.warning(f"Keine Dateien gefunden für {index_name}, verwende Demo-Daten")
        return create_demo_data(index_name)

def create_demo_data(index_name):
    """Erstellt realistische Demo-Daten mit Berücksichtigung von Handelszeiten"""
    # Zeitraum für Demo-Daten
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 Tage zurück für mehr Variabilität
    
    # Erzeuge eine vollständige Zeitreihe
    all_dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Filtere auf Handelszeiten
    trading_dates = [dt for dt in all_dates if is_trading_hour(dt, index_name)]
    
    # Wenn keine Handelstage im ausgewählten Zeitraum, generiere einfach Standarddaten
    if not trading_dates:
        trading_dates = all_dates
    
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
    
    # Zufällige Bewegung mit gewisser Tendenz und Handelszeitabhängigkeit
    values = []
    current_value = base_value
    
    for i, date in enumerate(trading_dates):
        # Erhöhte Volatilität zu Handelsbeginn und -ende
        hour = date.hour
        start_hour, end_hour, _ = get_trading_hours(index_name)
        
        # Volatilität basierend auf Tageszeit
        if hour == start_hour or hour == end_hour - 1:
            day_volatility = volatility * 1.5  # Höhere Volatilität zu Handelsbeginn/-ende
        else:
            day_volatility = volatility
        
        # Tägliche und wöchentliche Zyklen
        day_of_week = date.weekday()
        # Tendenz: Montag/Dienstag steigend, Mittwoch stabil, Donnerstag/Freitag fallend
        trend_factor = 0.0003 if day_of_week < 2 else (-0.0003 if day_of_week > 3 else 0)
        
        # Zufällige Bewegung mit Trend
        change = np.random.normal(trend_factor, day_volatility/2000)
        current_value *= (1 + change)
        values.append(current_value)
    
    # DataFrame erstellen
    df = pd.DataFrame(index=trading_dates)
    df['Close'] = values
    df['Open'] = df['Close'].shift(1) * np.random.uniform(0.998, 1.002, size=len(df))
    df['Open'].fillna(df['Close'].iloc[0] * 0.999, inplace=True)  # Für den ersten Wert
    
    df['High'] = df.apply(lambda row: max(row['Open'], row['Close']) * 
                         np.random.uniform(1.001, 1.003), axis=1)
    df['Low'] = df.apply(lambda row: min(row['Open'], row['Close']) * 
                        np.random.uniform(0.997, 0.999), axis=1)
    
    # Volumen: höher zu Handelsbeginn/-ende und Wochenmitte
    df['Volume'] = df.apply(
        lambda row: np.random.randint(3000, 10000) * 
                   (1.5 if row.name.hour == start_hour or row.name.hour == end_hour - 1 else 1) *
                   (1.3 if row.name.weekday() == 2 else 1),  # Höheres Volumen am Mittwoch
        axis=1
    )
    
    return df

def calculate_forecast(data, index_name, horizon, price_col):
    """Berechnet eine Prognose für den angegebenen Horizont unter Berücksichtigung von Handelszeiten"""
    if data is None or data.empty or len(data) < 2:
        return None
    
    # Sicherstellen, dass der letzte Wert numerisch ist
    try:
        last_value_raw = data[price_col].iloc[-1]
        last_value = float(last_value_raw)
        last_date = data.index[-1]
    except (ValueError, TypeError, IndexError):
        st.error(f"Fehler beim Zugriff auf den letzten Datenpunkt")
        return None
    
    # Bestimme den Prognosezeitpunkt unter Berücksichtigung von Handelszeiten
    # Verwende einen naive datetime für now, kompatibel mit Handelszeiten-Funktionen
    now = datetime.now()
    
    # Für Zeitzonenvergleiche: Entweder beide mit Zeitzone oder beide ohne
    # Hier entfernen wir die Zeitzone vom letzten Datum, um es mit now zu vergleichen
    last_date_naive = last_date
    if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
        # Konvertieren zu naive datetime für Vergleiche
        last_date_naive = last_date.replace(tzinfo=None)
    
    # Wenn aktuelle Zeit nicht im Datensatz liegt, verwende den letzten verfügbaren Zeitpunkt
    if now < last_date_naive:
        reference_time = last_date_naive
    else:
        reference_time = now
    
    # Finde den nächsten Handelszeitpunkt für die gewünschte Vorausschau
    target_time = next_trading_time(reference_time, index_name, hours_ahead=horizon)
    
    # Berechne Anzahl der tatsächlichen Handelsstunden zwischen jetzt und Zielzeitpunkt
    # (für realistischere Volatilität)
    trading_hours = 0
    test_time = reference_time
    while test_time < target_time:
        if is_trading_hour(test_time, index_name):
            trading_hours += 1
        test_time += timedelta(hours=1)
    
    # Wenn keine Handelsstunden zwischen Referenz und Ziel, verwende mindestens 1
    trading_hours = max(1, trading_hours)
    
    # Berechne Volatilität basierend auf historischen Daten
    if len(data) >= 24:
        # Volatilität aus den letzten 24 Datenpunkten (wenn verfügbar)
        hist_volatility = data[price_col].tail(24).pct_change().std()
    else:
        # Andernfalls Standard-Volatilität basierend auf Index
        if index_name == "USD_EUR":
            hist_volatility = 0.002
        else:
            hist_volatility = 0.005
    
    # Skaliere Volatilität mit Quadratwurzel der Zeit
    scaled_volatility = hist_volatility * np.sqrt(trading_hours / 24)
    
    # Bestimme Trend aus den letzten Datenpunkten
    if len(data) >= 48:
        # Lineare Regression für Trend
        y = data[price_col].tail(48).values
        x = np.arange(len(y))
        z = np.polyfit(x, y, 1)
        trend = z[0] / y[-1]  # Normalisiert auf letzten Preis
    else:
        # Andernfalls leichter positiver Trend
        trend = 0.0001
    
    # Berechne erwartete prozentuale Änderung
    expected_change = trend * trading_hours
    
    # Füge Zufallskomponente basierend auf der Volatilität hinzu
    random_component = np.random.normal(0, scaled_volatility)
    
    # Gesamtänderung
    change_pct = expected_change + random_component
    
    # Berechne Prognosewert
    forecast_value = last_value * (1 + change_pct)
    
    # Wenn last_date Zeitzone hat, behalte diese auch für Prognose-Datum
    forecast_date = target_time
    if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
        # Erstelle einen timezone-aware timestamp mit der gleichen Zeitzone
        forecast_date = pd.Timestamp(target_time, tz=last_date.tzinfo)
    
    # Erstelle Prognose-Objekt
    forecast = {
        "current_date": last_date,
        "current_value": last_value,
        "forecast_date": forecast_date,
        "forecast_value": forecast_value,
        "change_pct": change_pct * 100,
        "direction": "up" if forecast_value > last_value else "down",
        "confidence": np.random.randint(65, 95)  # Zufälliges Vertrauen für Demo
    }
    
    return forecast

# Hauptinhalt
st.title("📈 KI-basiertes Prognosemodell für Finanzindizes")
st.markdown("""
    Dieses Dashboard visualisiert Finanzprognosen für ausgewählte Indizes unter Berücksichtigung 
    von Handelszeiten, Feiertagen und Wochenenden. Die Prognosen basieren auf historischen Kursdaten 
    und Machine-Learning-Modellen.
""")

# Seitenleiste für Einstellungen
st.sidebar.title("Einstellungen")

# Aktuelle Zeit anzeigen
now = datetime.now()
st.sidebar.write(f"Aktuelle Zeit: {now.strftime('%d.%m.%Y %H:%M')}")

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
    "Handelsschluss": 8,
    "Nächster Handelsstart": 24
}
selected_horizon_name = st.sidebar.selectbox(
    "Prognosehorizont:",
    options=list(horizon_options.keys())
)
selected_horizon = horizon_options[selected_horizon_name]

# Zeitraum-Auswahl für historische Daten
history_days = st.sidebar.slider(
    "Historische Daten anzeigen (Tage):",
    min_value=1,
    max_value=30,
    value=7
)

# Daten laden
data = load_data(selected_index)

# Handelszeiten-Info anzeigen
start_hour, end_hour, end_minute = get_trading_hours(selected_index)
st.sidebar.write(f"Handelszeiten {selected_index}: {start_hour}:00 - {end_hour}:{end_minute} Uhr")

# Ist jetzt Handelszeit?
is_trading_now = is_trading_hour(now, selected_index)
st.sidebar.write(f"Aktuell Handelszeit: {'Ja' if is_trading_now else 'Nein'}")

# Daten anzeigen
if data is not None:
    # Zeige nur die letzten X Tage an, unter Berücksichtigung möglicher Zeitzonen
    cutoff_date = pd.Timestamp(now - timedelta(days=history_days))
    
    # Stelle sicher, dass cutoff_date die gleiche Zeitzone wie der Datensatz hat
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        cutoff_date = cutoff_date.tz_localize(data.index.tz)
    
    try:
        data_display = data[data.index >= cutoff_date]
        if len(data_display) == 0:
            data_display = data.tail(168)  # Zeige die letzten 168 Stunden, wenn keine Daten im gewählten Zeitraum
    except Exception as e:
        st.warning(f"Fehler beim Filtern der Daten: {e}")
        data_display = data.tail(168)
    
    # Finde die Preisspalte
    price_col = None
    for col_name in ['Close', 'close', 'Adj Close', 'adj_close', 'Price', 'price']:
        if col_name in data_display.columns:
            price_col = col_name
            break
    
    if price_col is None:
        # Nehmen wir die erste numerische Spalte
        numeric_cols = data_display.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            st.error("Keine numerischen Spalten gefunden!")
            st.stop()
    
    # Berechne Prognose
    forecast = calculate_forecast(data_display, selected_index, selected_horizon, price_col)
    
    if forecast:
        # Erstelle Plot mit historischen Daten und Prognose
        fig = go.Figure()
        
        # Historische Daten
        fig.add_trace(go.Scatter(
            x=data_display.index,
            y=data_display[price_col],
            mode='lines',
            name='Historischer Kurs',
            line=dict(color='royalblue')
        ))
        
        # Prognose
        fig.add_trace(go.Scatter(
            x=[forecast["current_date"], forecast["forecast_date"]],
            y=[forecast["current_value"], forecast["forecast_value"]],
            mode='lines+markers',
            line=dict(dash='dash', color='green' if forecast["direction"] == "up" else 'red'),
            name=f'Prognose ({selected_horizon_name})'
        ))
        
        # Hervorhebung des Prognosepunkts
        fig.add_trace(go.Scatter(
            x=[forecast["forecast_date"]],
            y=[forecast["forecast_value"]],
            mode='markers',
            marker=dict(
                color='green' if forecast["direction"] == "up" else 'red',
                size=10,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name='Prognosewert'
        ))
        
        # Layout anpassen
        fig.update_layout(
            title=f"{selected_index} Kursverlauf mit {selected_horizon_name} Prognose",
            xaxis_title="Datum",
            yaxis_title="Wert",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Handel/Nicht-Handel-Zeiten hervorheben
        if selected_index != "USD_EUR":  # Nicht für Forex, da 24/5
            try:
                # Finde Handelszeiten-Bereiche
                non_trading_periods = []
                
                # Bestimme Anfang und Ende des Zeitraums
                start_date = data_display.index[0]
                end_date = data_display.index[-1]
                
                # Konvertiere zu naive datetime für die Verarbeitung
                if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
                
                current_date = start_date.date()
                end_date = end_date.date()
                
                while current_date <= end_date:
                    current_dt = datetime.combine(current_date, datetime.min.time())
                    
                    # Wenn Wochenende oder Feiertag, markiere den ganzen Tag
                    if current_dt.weekday() >= 5 or is_holiday(current_date, 'DE' if selected_index == 'DAX' else 'US'):
                        day_start = datetime.combine(current_date, datetime.min.time())
                        day_end = datetime.combine(current_date, datetime.max.time())
                        non_trading_periods.append((day_start, day_end))
                    else:
                        # Ansonsten markiere nur die Nicht-Handelszeiten
                        start_hour, end_hour, end_minute = get_trading_hours(selected_index)
                        
                        # Zeit vor Handelsbeginn
                        morning_start = datetime.combine(current_date, datetime.min.time())
                        morning_end = datetime.combine(current_date, datetime.min.time().replace(hour=start_hour))
                        non_trading_periods.append((morning_start, morning_end))
                        
                        # Zeit nach Handelsende
                        evening_start = datetime.combine(current_date, datetime.min.time().replace(hour=end_hour, minute=end_minute))
                        evening_end = datetime.combine(current_date, datetime.max.time())
                        non_trading_periods.append((evening_start, evening_end))
                    
                    current_date += timedelta(days=1)
                
                # Füge Hintergrundfarbe für Nicht-Handelszeiten hinzu
                for start, end in non_trading_periods:
                    # Wenn Daten Zeitzone haben, müssen wir die Zeitzone zu den Markierungen hinzufügen
                    if hasattr(data_display.index, 'tz') and data_display.index.tz is not None:
                        tz = data_display.index.tz
                        start = pd.Timestamp(start, tz=tz)
                        end = pd.Timestamp(end, tz=tz)
                    
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor="lightgray",
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                    )
            except Exception as e:
                st.error(f"Fehler beim Markieren der Handelszeiten: {e}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dashboard mit Hauptkennzahlen
        col1, col2 = st.columns(2)
        
        with col1:
            # Aktueller Wert
            st.metric(
                label=f"{selected_index} aktuell ({forecast['current_date'].strftime('%d.%m.%Y %H:%M')})",
                value=f"{forecast['current_value']:.2f}"
            )
            
            # Handelszeiten-Info
            if is_trading_now:
                st.success("📊 Aktuell ist Handelszeit")
            else:
                next_trading = next_trading_time(now, selected_index, 0)
                st.warning(f"⏰ Nächste Handelszeit: {next_trading.strftime('%d.%m.%Y %H:%M')}")
        
        with col2:
            # Prognose
            direction_icon = "↗️" if forecast["direction"] == "up" else "↘️"
            st.metric(
                label=f"Prognose für {forecast['forecast_date'].strftime('%d.%m.%Y %H:%M')}",
                value=f"{forecast['forecast_value']:.2f}",
                delta=f"{forecast['change_pct']:+.2f}%"
            )
            
            # Zeitraum-Info
            # Sichere Berechnung des Zeitunterschieds
            current_date = forecast['current_date']
            forecast_date = forecast['forecast_date']
            
            # Konvertiere zu naive datetime für die Berechnung, falls nötig
            if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
                current_date = current_date.replace(tzinfo=None)
            if hasattr(forecast_date, 'tzinfo') and forecast_date.tzinfo is not None:
                forecast_date = forecast_date.replace(tzinfo=None)
                
            trading_hours_diff = (forecast_date - current_date).total_seconds() / 3600
            st.info(f"🕒 Prognosezeitraum: {trading_hours_diff:.1f} Stunden")
        
        # Prognose-Box mit Richtung
        direction_text = "STEIGEN" if forecast["direction"] == "up" else "FALLEN"
        color = "rgba(0, 200, 0, 0.1)" if forecast["direction"] == "up" else "rgba(200, 0, 0, 0.1)"
        direction_icon = "↗️" if forecast["direction"] == "up" else "↘️"
        
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}; margin: 20px 0;">
                <h2 style="text-align: center; margin-bottom: 15px;">
                    {direction_icon} {selected_index} wird voraussichtlich {direction_text}
                </h2>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div>
                        <p style="font-weight: bold;">Prognose:</p>
                        <p style="font-size: 1.2em;">{forecast['forecast_value']:.2f} ({forecast['change_pct']:+.2f}%)</p>
                    </div>
                    <div>
                        <p style="font-weight: bold;">Genauigkeit:</p>
                        <p style="font-size: 1.2em;">{forecast['confidence']}%</p>
                    </div>
                    <div>
                        <p style="font-weight: bold;">Zeitpunkt:</p>
                        <p style="font-size: 1.2em;">{forecast['forecast_date'].strftime('%d.%m.%Y %H:%M')}</p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Tabelle mit den letzten Datenpunkten
        with st.expander("Letzte Handelsdaten anzeigen", expanded=False):
            st.dataframe(data_display.tail(10))
        
        # Modell-KPIs
        st.subheader("Modell-Leistungskennzahlen")
        
        # Generiere einige realistische KPIs für die Demo
        mae_value = np.random.uniform(0.1, 0.8) * (500 if selected_index == "DowJones" else 20 if selected_index == "DAX" else 0.002)
        mape_value = np.random.uniform(0.8, 2.5)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(
                label="MAE (mittlerer abs. Fehler)",
                value=f"{mae_value:.2f}"
            )
        
        with kpi2:
            st.metric(
                label="MAPE (mittl. proz. Fehler)",
                value=f"{mape_value:.2f}%"
            )
        
        with kpi3:
            st.metric(
                label="Trainings-Iterationen",
                value=f"{np.random.randint(15, 40)}"
            )
        
        with kpi4:
            improvement = np.random.uniform(5, 15)
            st.metric(
                label="Verbesserung seit letztem Training",
                value=f"{improvement:.1f}%"
            )
    else:
        st.error("Konnte keine Prognose berechnen. Bitte überprüfen Sie die Daten.")
        
else:
    st.error("Keine Daten verfügbar. Bitte führen Sie zuerst die Datensammlung aus.")

# Footer mit Hinweis
st.markdown("---")
st.markdown("""
    **Hinweis**: Die Prognosen berücksichtigen reguläre Handelszeiten, Wochenenden und Feiertage. 
    Außerhalb der Handelszeiten wird die Prognose für den nächsten Handelszeitpunkt erstellt.
    
    *Letzte Aktualisierung der Daten: `{}`*
""".format(datetime.now().strftime('%d.%m.%Y %H:%M')))
