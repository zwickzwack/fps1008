import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback
import re

# Stellen Sie sicher, dass NLTK-Ressourcen geladen sind
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    
# Sentiment-Analyzer initialisieren
sia = SentimentIntensityAnalyzer()

# Dashboard-Konfiguration
st.set_page_config(page_title="Financial News Analyzer", page_icon="📊", layout="wide")
st.title("📊 Financial News-Based Market Predictor")

# Aktuelles Datum (für Simulationsumgebung fest eingestellt)
current_time = datetime.strptime("2025-04-19 18:07:50", "%Y-%m-%d %H:%M:%S")
st.sidebar.info(f"Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info(f"User: zwickzwack")

# Markt-Konfiguration
MARKETS = {
    "DAX": {
        "ticker": "^GDAXI",
        "description": "German Stock Index",
        "currency": "EUR",
        "opening_time": time(9, 0),  # 9:00 AM
        "closing_time": time(17, 30),  # 5:30 PM
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "keywords": ["DAX", "German market", "Frankfurt", "German economy", "ECB", "European Central Bank", 
                    "Deutsche Börse", "German stocks", "German inflation", "German manufacturing",
                    "Volkswagen", "Siemens", "BMW", "Deutsche Bank", "Allianz", "SAP", "BASF"],
    },
    "DowJones": {
        "ticker": "^DJI",
        "description": "Dow Jones Industrial Average",
        "currency": "USD",
        "opening_time": time(9, 30),  # 9:30 AM
        "closing_time": time(16, 0),  # 4:00 PM
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "keywords": ["Dow Jones", "DJIA", "Wall Street", "US market", "Federal Reserve", "Fed", 
                    "US economy", "US inflation", "US manufacturing", "US employment", 
                    "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Walmart", "JPMorgan"],
    }
}

# Verzeichnisse erstellen
os.makedirs("data/news", exist_ok=True)
os.makedirs("data/market", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Markt auswählen
selected_market = st.sidebar.selectbox(
    "Select Market:",
    options=list(MARKETS.keys())
)

# Prüfen, ob Markt geöffnet ist
def is_market_open(market_name, check_time=None):
    if check_time is None:
        check_time = current_time
    
    market = MARKETS.get(market_name)
    if not market:
        return False
    
    # Prüfen, ob es ein Handelstag ist
    if check_time.weekday() not in market["trading_days"]:
        return False
    
    # Prüfen, ob innerhalb der Handelszeiten
    current_time_of_day = check_time.time()
    return market["opening_time"] <= current_time_of_day <= market["closing_time"]

# Nächsten Handelstag ermitteln
def get_next_trading_day(market_name, from_date=None):
    if from_date is None:
        from_date = current_time
    
    market = MARKETS.get(market_name)
    if not market:
        return None
    
    # Mit morgen beginnen
    next_day = from_date + timedelta(days=1)
    
    # Nächsten Handelstag finden
    while next_day.weekday() not in market["trading_days"]:
        next_day = next_day + timedelta(days=1)
    
    # Öffnungszeit an diesem Tag zurückgeben
    next_opening = datetime.combine(next_day.date(), market["opening_time"])
    return next_opening

# Finanznachrichten abrufen
def fetch_financial_news(days_back=14, market_name=None):
    # Cache-Datei überprüfen
    news_file = f"data/news/{market_name}_news_{days_back}days.json"
    
    if os.path.exists(news_file):
        with open(news_file, 'r') as f:
            return json.load(f)
    
    # In einer realen Anwendung würden wir hier eine echte Nachrichten-API verwenden
    # Zum Beispiel: NewsAPI, Alpha Vantage News API, Bloomberg API, etc.
    # Da wir in einer simulierten Umgebung sind, generieren wir Beispielnachrichten
    
    st.sidebar.info(f"Fetching financial news for {market_name}...")
    
    # Relevante Keywords für den ausgewählten Markt
    keywords = MARKETS.get(market_name, {}).get("keywords", [])
    
    # Nachrichtendatenbank generieren
    end_date = current_time
    start_date = end_date - timedelta(days=days_back)
    
    # Listen potenzieller Schlagzeilen
    positive_headlines = [
        "Strong economic data boosts investor confidence in {market}",
        "Corporate earnings exceed expectations, lifting {market}",
        "Central bank signals continued support for economy, {market} rises",
        "Inflation fears ease as {market} rebounds",
        "Trade negotiations show progress, {market} responds positively",
        "Consumer spending surges, driving {market} gains",
        "Tech sector leads rally in {market}",
        "Manufacturing activity expands, {market} reaches new highs",
        "Unemployment falls to multi-year low, boosting {market}",
        "New stimulus package announced, {market} jumps"
    ]
    
    negative_headlines = [
        "Economic slowdown concerns weigh on {market}",
        "Corporate earnings disappoint, {market} tumbles",
        "Central bank hints at tightening policy, pressuring {market}",
        "Inflation data exceeds expectations, {market} falls",
        "Trade tensions escalate, pushing {market} lower",
        "Consumer confidence declines, impacting {market}",
        "Tech sector sell-off leads {market} downturn",
        "Manufacturing activity contracts, {market} slides",
        "Unemployment rises unexpectedly, {market} drops",
        "Political uncertainty affects {market} performance"
    ]
    
    neutral_headlines = [
        "{market} trades sideways as investors await economic data",
        "Mixed corporate results leave {market} searching for direction",
        "Central bank maintains current policy, {market} little changed",
        "Inflation in line with expectations, {market} steady",
        "Trade discussions continue without breakthrough, {market} stable",
        "Consumer spending matches forecasts, {market} holds ground",
        "Tech sector shows mixed performance in {market}",
        "Manufacturing data meets expectations, {market} flat",
        "Labor market remains stable, {market} unchanged",
        "Investors cautious as {market} awaits policy developments"
    ]
    
    # Nachrichten generieren
    news = []
    current_date = start_date
    np.random.seed(hash(market_name + str(days_back)) % 10000)
    
    while current_date <= end_date:
        # Anzahl der Nachrichten für diesen Tag bestimmen (0-5)
        num_items = np.random.randint(0, 6)
        
        for _ in range(num_items):
            # Stimmung basierend auf Wochentag und Zufallsfaktor bestimmen
            day_factor = np.sin(current_date.weekday() * np.pi/7) * 0.5  # Zyklisches Muster
            random_factor = np.random.normal(0, 0.7)
            sentiment_score = day_factor + random_factor
            
            # Schlagzeile basierend auf Stimmung auswählen
            if sentiment_score > 0.3:
                headline_template = np.random.choice(positive_headlines)
                sentiment = "positive"
            elif sentiment_score < -0.3:
                headline_template = np.random.choice(negative_headlines)
                sentiment = "negative"
            else:
                headline_template = np.random.choice(neutral_headlines)
                sentiment = "neutral"
            
            # Marktname in Schlagzeile einfügen
            headline = headline_template.format(market=market_name)
            
            # Zufällige Uhrzeit während Geschäftszeiten
            hour = np.random.randint(7, 20)
            minute = np.random.randint(0, 60)
            news_datetime = datetime.combine(current_date.date(), time(hour, minute))
            
            # Zufälliges relevantes Keyword auswählen
            relevant_keywords = np.random.choice(keywords, size=min(3, len(keywords)), replace=False).tolist()
            
            # Nachrichtenelement erstellen
            news_item = {
                "headline": headline,
                "datetime": news_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
                "sentiment_score": float(sentiment_score),
                "source": np.random.choice(["Bloomberg", "Reuters", "Financial Times", "CNBC", "Wall Street Journal"]),
                "impact_score": abs(float(sentiment_score) * np.random.uniform(0.5, 1.5)),
                "keywords": relevant_keywords,
                "content": generate_news_content(headline, sentiment, relevant_keywords)
            }
            
            news.append(news_item)
        
        current_date += timedelta(days=1)
    
    # Nach Datum sortieren
    news = sorted(news, key=lambda x: x["datetime"])
    
    # Nachrichten speichern
    with open(news_file, 'w') as f:
        json.dump(news, f)
    
    return news

# Nachrichteninhalt generieren
def generate_news_content(headline, sentiment, keywords):
    # Templated Nachrichtentext basierend auf Sentiment
    if sentiment == "positive":
        templates = [
            "Analysts are optimistic about the outlook for {keyword1} as {keyword2} shows strong performance. {headline} Investors are responding positively to the news, with many increasing their positions in anticipation of continued growth. Market strategists point to improving fundamentals and technical indicators that suggest further upside potential.",
            "The recent developments in {keyword1} have created a positive environment for investors. {headline} This comes after a period of uncertainty, but the latest data suggests a robust recovery is underway. {keyword2} is particularly well-positioned to benefit from these trends, according to market experts.",
            "{headline} This represents a significant shift in market sentiment regarding {keyword1}. The improved outlook is attributed to strong earnings reports and favorable economic conditions. Analysts at several major banks have upgraded their forecasts for {keyword2}, citing positive momentum."
        ]
    elif sentiment == "negative":
        templates = [
            "Concerns are growing among investors about the prospects for {keyword1} as {keyword2} faces challenges. {headline} Market participants are reassessing their positions in light of these developments, with some reducing exposure to affected sectors. Analysts warn that volatility may increase in the near term.",
            "The situation surrounding {keyword1} has created headwinds for the market. {headline} This follows earlier indications of potential issues, which have now materialized into concrete challenges. {keyword2} is particularly vulnerable to these trends, according to industry observers.",
            "{headline} This marks a troubling development for {keyword1}, which had previously shown signs of stability. The deterioration is linked to broader economic concerns and specific issues affecting {keyword2}. Several analysts have downgraded their outlooks in response."
        ]
    else:  # neutral
        templates = [
            "Market observers are taking a measured approach to recent developments in {keyword1}. {headline} This comes amid mixed signals from economic indicators and corporate reports. {keyword2} shows both positive and challenging aspects, leading to a balanced outlook among analysts.",
            "Investors are carefully monitoring the situation with {keyword1} as the market digests recent information. {headline} The implications remain unclear, with potential for movement in either direction depending on forthcoming data. {keyword2} remains a focal point for those following this sector.",
            "{headline} Analysts are divided in their interpretation of what this means for {keyword1}. Some see potential opportunities emerging, while others maintain a cautious stance. {keyword2} exemplifies this split in opinion, with varied forecasts from different research teams."
        ]
    
    # Zufällige Vorlage auswählen
    template = np.random.choice(templates)
    
    # Keywords einfügen
    keyword1 = keywords[0] if keywords else "the market"
    keyword2 = keywords[1] if len(keywords) > 1 else "the sector"
    
    # Nachrichtentext erzeugen
    content = template.format(headline=headline, keyword1=keyword1, keyword2=keyword2)
    
    return content

# Historische Marktdaten abrufen
def fetch_historical_data(market_name):
    try:
        market_info = MARKETS.get(market_name)
        if not market_info:
            return None
        
        ticker = market_info["ticker"]
        
        # Daten von Yahoo Finance abrufen - 30 Tage für besseres Modelltraining
        end_date = current_time
        start_date = end_date - timedelta(days=30)
        
        st.sidebar.info(f"Fetching market data for {market_name} ({ticker})...")
        
        # Bei stündlichen Daten versuchen
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
        
        if data.empty or len(data) < 24:
            # Auf tägliche Daten zurückfallen
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty:
                # In stündliche Daten konvertieren
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='1H')
                data = data.reindex(new_index, method='ffill')
        
        if not data.empty:
            # MultiIndex-Spalten behandeln, falls vorhanden
            if isinstance(data.columns, pd.MultiIndex):
                st.sidebar.info("MultiIndex columns detected")
                # Flache Version der Daten mit einfachen Spaltennamen erstellen
                flat_data = pd.DataFrame(index=data.index)
                
                # Daten für unser Ticker extrahieren
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns.get_level_values(0):
                        flat_data[col] = data[col, ticker] if (col, ticker) in data.columns else data[col][0]
                data = flat_data
            
            # Zeitzonenfreien Index erstellen
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Markt geöffnet/geschlossen markieren
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

# Demo-Daten erstellen
def create_demo_data(market_name):
    # Basiswerte setzen
    if market_name == "DAX":
        base_value = 18500
        volatility = 100
    else:  # DowJones
        base_value = 39000
        volatility = 200
    
    # Datumsreihe generieren
    end_date = current_time
    start_date = end_date - timedelta(days=30)  # 30 Tage für besseres Modelltraining
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Preisdaten generieren
    np.random.seed(42 + hash(market_name) % 100)
    price_data = []
    current_price = base_value
    
    for date in date_range:
        # Zeit- und Tageseffekte
        hour = date.hour
        day = date.weekday()
        
        # Marktöffnungseffekt
        market_open = is_market_open(market_name, date)
        hour_volatility = volatility if market_open else volatility * 0.3
        
        # Tageseffekt
        day_factor = 1.0 if day <= 4 else 0.3
        day_trend = 0.0001 * (2 - day) if day <= 4 else 0
        
        # Preisänderungskomponenten
        random_component = np.random.normal(0, hour_volatility * day_factor / 1000)
        price_change = day_trend + random_component
        current_price *= (1 + price_change)
        
        price_data.append(current_price)
    
    # DataFrame erstellen
    df = pd.DataFrame({
        'Open': price_data,
        'High': [p * (1 + np.random.uniform(0.001, 0.003)) for p in price_data],
        'Low': [p * (1 - np.random.uniform(0.001, 0.003)) for p in price_data],
        'Close': [p * (1 + np.random.normal(0, 0.001)) for p in price_data],
        'Volume': np.random.randint(1000, 10000, size=len(date_range))
    }, index=date_range)
    
    # Markt geöffnet/geschlossen markieren
    df['MarketOpen'] = [is_market_open(market_name, dt) for dt in df.index]
    
    return df

# Nachrichtenfeatures erstellen
def extract_news_features(news_data, market_data_index):
    """Extrahiert Features aus Nachrichtendaten für die Vorhersagemodellierung"""
    
    # DataFrame mit demselben Index wie die Marktdaten erstellen
    news_features = pd.DataFrame(index=market_data_index)
    
    # Spalten initialisieren
    news_features['sentiment_avg'] = 0.0
    news_features['sentiment_std'] = 0.0
    news_features['news_count'] = 0
    news_features['positive_ratio'] = 0.0
    news_features['negative_ratio'] = 0.0
    news_features['impact_score'] = 0.0
    news_features['keyword_relevance'] = 0.0
    
    # Nachrichtendaten-Datetime in Datetime-Objekte konvertieren
    for news in news_data:
        news['datetime_obj'] = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
    
    # Nachrichten nach Tag gruppieren
    news_by_day = {}
    for news in news_data:
        day_key = news['datetime_obj'].date()
        if day_key not in news_by_day:
            news_by_day[day_key] = []
        news_by_day[day_key].append(news)
    
    # Jeden Tag verarbeiten
    for date in market_data_index:
        day_key = date.date()
        
        day_news = news_by_day.get(day_key, [])
        
        if day_news:
            # Nachrichten zählen
            news_features.at[date, 'news_count'] = len(day_news)
            
            # Stimmungsmetriken berechnen
            sentiments = [n['sentiment_score'] for n in day_news]
            news_features.at[date, 'sentiment_avg'] = np.mean(sentiments)
            news_features.at[date, 'sentiment_std'] = np.std(sentiments) if len(sentiments) > 1 else 0
            
            # Positiv/Negativ-Verhältnisse berechnen
            pos_count = sum(1 for n in day_news if n['sentiment'] == 'positive')
            neg_count = sum(1 for n in day_news if n['sentiment'] == 'negative')
            total = len(day_news)
            
            news_features.at[date, 'positive_ratio'] = pos_count / total if total > 0 else 0
            news_features.at[date, 'negative_ratio'] = neg_count / total if total > 0 else 0
            
            # Impact-Score berechnen
            news_features.at[date, 'impact_score'] = np.mean([n.get('impact_score', 0) for n in day_news])
            
            # Keyword-Relevanz berechnen (basierend auf Anzahl der Schlüsselwörter)
            keyword_count = sum(len(n.get('keywords', [])) for n in day_news)
            news_features.at[date, 'keyword_relevance'] = keyword_count / total if total > 0 else 0
    
    # Forward fill für Stunden ohne Nachrichten
    news_features = news_features.fillna(method='ffill')
    
    # Verzögerte Features hinzufügen
    for lag in [1, 2, 3]:
        news_features[f'sentiment_avg_lag{lag}'] = news_features['sentiment_avg'].shift(lag * 24)
        news_features[f'impact_score_lag{lag}'] = news_features['impact_score'].shift(lag * 24)
    
    # Gleitende Durchschnitte hinzufügen
    for window in [24, 48, 72]:  # 1, 2, 3 Tage
        news_features[f'sentiment_ma{window}'] = news_features['sentiment_avg'].rolling(window=window).mean()
        news_features[f'impact_ma{window}'] = news_features['impact_score'].rolling(window=window).mean()
    
    # Verbleibende NAs füllen
    news_features = news_features.fillna(0)
    
    return news_features

# Technische Features erstellen
def create_technical_features(data):
    """Erstellt technische Indikatoren aus Marktdaten"""
    features = pd.DataFrame(index=data.index)
    
    # Zeitfeatures
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['day_of_month'] = data.index.day
    features['month'] = data.index.month
    features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Marktöffnungs-Feature
    if 'MarketOpen' in data.columns:
        features['market_open'] = data['MarketOpen'].astype(int)
    
    # Technische Indikatoren
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
    
    # Volatilität
    features['volatility_12h'] = features['return_1h'].rolling(window=12).std()
    features['volatility_24h'] = features['return_1h'].rolling(window=24).std()
    
    # Preisdifferenz aus OHLC
    features['high_low_diff'] = (data['High'] - data['Low']) / data['Low']
    features['open_close_diff'] = (data['Close'] - data['Open']) / data['Open']
    
    # Lag-Features
    for i in range(1, 25):
        features[f'lag_{i}'] = data['Close'].shift(i)
    
    # Fehlende Werte füllen
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# Kombinierte Features für das Modell erstellen
def create_combined_features(market_data, news_features):
    """Kombiniert technische und Nachrichtenfeatures für das Vorhersagemodell"""
    
    # Technische Features erstellen
    tech_features = create_technical_features(market_data)
    
    # Alle Features kombinieren
    combined = tech_features.copy()
    
    # Nachrichtenfeatures hinzufügen
    if news_features is not None:
        for col in news_features.columns:
            combined[f'news_{col}'] = news_features[col]
    
    return combined

# Nächsten Handelstag vorhersagen (basierend auf Nachrichten und technischen Indikatoren)
def predict_next_trading_day(market_data, market_name, news_data):
    if market_data is None or len(market_data) < 48:  # Genügend Daten für das Training benötigen
        return None
    
    try:
        with st.spinner("Training predictive model..."):
            # Nachrichten-Features extrahieren
            news_features = extract_news_features(news_data, market_data.index)
            
            # Kombinierte Features erstellen
            features = create_combined_features(market_data, news_features)
            feature_names = features.columns.tolist()
            
            # Sicherstellen, dass alle Features numerisch sind
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].fillna(0)
            
            # Modell trainieren
            X = features.values
            y = market_data['Close'].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Gradient Boosting für bessere Performance mit Nachrichtendaten
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            model.fit(X_scaled, y)
            
            # Nächsten Handelstag-Zeitstempel abrufen
            next_open = get_next_trading_day(market_name)
            
            if next_open is None:
                return None
            
            # Aktueller Preis
            current_price = market_data['Close'].iloc[-1]
            
            # Stunden bis zur Eröffnung berechnen
            hours_to_opening = int((next_open - current_time).total_seconds() / 3600) + 1
            
            # Zukünftige Daten für die Vorhersage
            future_dates = pd.date_range(
                start=market_data.index[-1] + timedelta(hours=1),
                periods=hours_to_opening,
                freq='1H'
            )
            
            # Rollende Vorhersagen bis zur Eröffnung
            temp_data = market_data.copy()
            temp_news_features = news_features.copy()
            open_prediction = None
            
            for i, future_date in enumerate(future_dates):
                # Nachrichtenfeatures in die Zukunft erweitern
                if future_date not in temp_news_features.index:
                    # Letzte Nachrichtenwerte für zukünftige Nachrichtenfeatures verwenden
                    latest_news_values = temp_news_features.iloc[-1].to_dict()
                    temp_news_features.loc[future_date] = latest_news_values
                
                # Kombinierte Features generieren
                temp_combined = create_combined_features(temp_data, temp_news_features)
                
                # Featurekonsistenz sicherstellen
                for feat in feature_names:
                    if feat not in temp_combined.columns:
                        temp_combined[feat] = 0
                
                temp_combined = temp_combined[feature_names]
                temp_combined = temp_combined.fillna(0)
                
                # Vorhersage treffen
                X_new = scaler.transform(temp_combined.values[-1:])
                prediction = model.predict(X_new)[0]
                
                # Zu temp_data für die nächste Iteration hinzufügen
                new_row = pd.DataFrame({
                    'Open': [prediction],
                    'High': [prediction * 1.001],
                    'Low': [prediction * 0.999],
                    'Close': [prediction],
                    'Volume': [market_data['Volume'].mean()],
                    'MarketOpen': [is_market_open(market_name, future_date)]
                }, index=[future_date])
                
                temp_data = pd.concat([temp_data, new_row])
                
                # Wenn dies die Eröffnungszeit ist, Vorhersage speichern
                if i == hours_to_opening - 1:
                    open_prediction = prediction
            
            # Wichtige Nachrichten identifizieren
            important_news = identify_important_news(news_data, market_name, model, feature_names)
            
            # Feature-Wichtigkeit
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Nachrichteneinfluss auf die Vorhersage berechnen
            news_features_importance = calculate_news_influence(importance)
            
            # Vorhersagen und Metadaten zurückgeben
            return {
                'current_price': current_price,
                'next_open_time': next_open,
                'next_open_prediction': open_prediction,
                'feature_importance': importance,
                'important_news': important_news,
                'news_influence': news_features_importance
            }
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Wichtige Nachrichten identifizieren
def identify_important_news(news_data, market_name, model, feature_names):
    """Identifiziert Nachrichten, die für die Vorhersage relevant sind"""
    
    # Nachrichten der letzten 3 Tage filtern
    recent_news = []
    cutoff_date = current_time - timedelta(days=3)
    
    for news in news_data:
        news_date = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
        if news_date >= cutoff_date:
            recent_news.append(news)
    
    # Nachrichten nach Sentiment und Impact-Score sortieren
    sorted_news = sorted(recent_news, key=lambda x: abs(x.get('sentiment_score', 0) * x.get('impact_score', 0)), reverse=True)
    
    # Top-Nachrichten zurückgeben
    return sorted_news[:5]

# Nachrichteneinfluss berechnen
def calculate_news_influence(feature_importance):
    """Berechnet den Einfluss von Nachrichtenfeatures auf die Vorhersage"""
    
    # Nachrichtenfeatures filtern
    news_features = feature_importance[feature_importance['Feature'].str.contains('news_')]
    
    # Gesamteinfluss berechnen
    total_importance = feature_importance['Importance'].sum()
    news_importance = news_features['Importance'].sum()
    
    # Prozentsatz des Einflusses
    influence_percentage = (news_importance / total_importance) * 100
    
    # Top-Nachrichtenfeatures
    top_news_features = news_features.head(5)
    
    return {
        'total_influence_pct': influence_percentage,
        'top_features': top_news_features
    }

# Marktdaten und Vorhersagen plotten
def plot_market_with_news(data, market_name, predictions=None, news_data=None):
    fig = go.Figure()
    
    # Aktuellen Zeitpunkt formatieren
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Nur die letzten 14 Tage anzeigen
    display_start = current_time - timedelta(days=14)
    display_data = data[data.index >= display_start]
    
    # Preislinie hinzufügen
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['Close'],
        mode='lines',
        name=f'{market_name} Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Handelszeiten-Schattierung hinzufügen
    if 'MarketOpen' in display_data.columns:
        # Zeiträume finden, wenn der Markt geöffnet ist
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
        
        # Falls am Ende noch geöffnet
        if start_open is not None:
            open_periods.append((start_open, display_data.index[-1]))
        
        # Schattierung für offene Perioden hinzufügen
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
    
    # Aktuelle Zeitlinie hinzufügen
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
    
    # Vorhersagen hinzufügen, falls verfügbar
    if predictions:
        # Eröffnungsvorhersage hinzufügen
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
    
    # Wichtige Nachrichten hinzufügen
    if news_data:
        # Nur Nachrichten der letzten 14 Tage
        recent_news = []
        for news in news_data:
            news_date = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
            if news_date >= display_start and news_date <= current_time:
                recent_news.append(news)
        
        # Nach Impact sortieren
        sorted_news = sorted(recent_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)
        
        # Top-10-Nachrichten hinzufügen
        for news in sorted_news[:10]:
            news_time = datetime.strptime(news['datetime'], "%Y-%m-%d %H:%M:%S")
            
            # Farbe basierend auf Sentiment
            if news['sentiment'] == 'positive':
                color = 'green'
                symbol = 'triangle-up'
            elif news['sentiment'] == 'negative':
                color = 'red'
                symbol = 'triangle-down'
            else:
                color = 'gray'
                symbol = 'circle'
            
            # Nachrichtenmarker für wichtige Nachrichten
            fig.add_trace(go.Scatter(
                x=[news_time],
                y=[data.loc[data.index[data.index.get_indexer([news_time], method='nearest')[0]], 'Close']],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    symbol=symbol
                ),
                name=news['headline'],
                hoverinfo='text',
                hovertext=f"{news['datetime']}: {news['headline']}<br>Source: {news['source']}<br>Sentiment: {news['sentiment']}"
            ))
    
    # Layout aktualisieren
    fig.update_layout(
        title=f"{market_name} - Market Data with News Influence",
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

# Datenquellenauswahl
data_source = st.sidebar.radio(
    "Data Source:",
    options=["Live Data", "Demo Data"],
    index=0
)

# Daten für ausgewählten Markt abrufen
if data_source == "Live Data":
    market_data = fetch_historical_data(selected_market)
    if market_data is None or market_data.empty:
        st.warning(f"Could not fetch live data for {selected_market}. Using demo data instead.")
        market_data = create_demo_data(selected_market)
else:
    market_data = create_demo_data(selected_market)

# Finanznachrichten abrufen
news_data = fetch_financial_news(days_back=14, market_name=selected_market)

# Vorhersagen generieren
predictions = predict_next_trading_day(market_data, selected_market, news_data)

# Dashboard-Layout
st.subheader(f"{selected_market} Market Analysis with News Influence")

col1, col2 = st.columns([3, 1])

with col1:
    # Marktdaten und Nachrichten plotten
    market_plot = plot_market_with_news(market_data, selected_market, predictions, news_data)
    st.plotly_chart(market_plot, use_container_width=True)

with col2:
    # Marktstatus anzeigen
    is_open = is_market_open(selected_market)
    status = "OPEN" if is_open else "CLOSED"
    status_color = "green" if is_open else "red"
    
    st.markdown(f"### Market Status: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    
    # Aktuellen Preis anzeigen
    if market_data is not None and not market_data.empty:
        current_price = market_data['Close'].iloc[-1]
        currency = MARKETS[selected_market]['currency']
        st.markdown(f"### Current Price: {current_price:.2f} {currency}")
    
    # Vorhersagen anzeigen
    if predictions:
        st.markdown("## Next Trading Day Prediction")
        
        # Nächsten Handelstag formatieren
        next_open_time = predictions['next_open_time']
        next_trading_day = next_open_time.strftime("%A, %Y-%m-%d")
        st.markdown(f"### Trading Day: {next_trading_day}")
        
        # Änderungen berechnen
        current = predictions['current_price']
        open_pred = predictions['next_open_prediction']
        
        open_change = ((open_pred - current) / current) * 100
        open_color = "green" if open_change > 0 else "red"
        
        # Eröffnungsvorhersage anzeigen
        st.markdown(f"#### Opening at {next_open_time.strftime('%H:%M')}:")
        st.markdown(f"<span style='color:{open_color}'>{open_pred:.2f} {currency} ({open_change:+.2f}%)</span>", unsafe_allow_html=True)
        
        # Gesamtvorhersagezusammenfassung
        direction = "RISE" if open_change > 0 else "FALL"
        st.markdown(f"### Overall Prediction: <span style='color:{open_color}'>{direction}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{open_color}'>Expected change: {open_change:+.2f}%</span>", unsafe_allow_html=True)
        
        # Nachrichteneinfluss anzeigen
        if 'news_influence' in predictions:
            news_influence = predictions['news_influence']
            st.markdown(f"### News Impact on Prediction: {news_influence['total_influence_pct']:.1f}%")

# Wichtige Nachrichten anzeigen
if predictions and 'important_news' in predictions:
    st.subheader("Key News Influencing the Prediction")
    
    important_news = predictions['important_news']
    
    for news in important_news:
        # Farbe basierend auf Sentiment
        if news['sentiment'] == 'positive':
            sentiment_color = "green"
            sentiment_icon = "📈"
        elif news['sentiment'] == 'negative':
            sentiment_color = "red"
            sentiment_icon = "📉"
        else:
            sentiment_color = "gray"
            sentiment_icon = "📊"
        
        # Nachricht mit Styling anzeigen
        st.markdown(f"""
        <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin-bottom: 15px;">
            <p><b>{news['datetime']}</b> {sentiment_icon} {news['headline']}</p>
            <p style="color: #666; font-size: 0.9em;">Source: {news['source']} | 
            Sentiment: <span style="color: {sentiment_color};">{news['sentiment'].upper()}</span> | 
            Impact: {news['impact_score']:.2f}</p>
            <p style="font-size: 0.9em;">{news['content'][:150]}...</p>
        </div>
        """, unsafe_allow_html=True)

# Faktoren anzeigen, die die Vorhersage beeinflussen
if predictions and 'feature_importance' in predictions:
    st.subheader("Factors Influencing the Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top-10-Features
        top_features = predictions['feature_importance'].head(10)
        
        # Balkendiagramm erstellen
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features['Feature'],
            y=top_features['Importance'],
            marker=dict(color=['royalblue' if not f.startswith('news_') else 'orange' for f in top_features['Feature']])
        ))
        
        fig.update_layout(
            title="Top 10 Factors by Importance",
            xaxis_title="Factor",
            yaxis_title="Importance",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Nachrichtenbasierte vs. technische Features
        if 'news_influence' in predictions:
            news_pct = predictions['news_influence']['total_influence_pct']
            technical_pct = 100 - news_pct
            
            # Kuchendiagramm erstellen
            fig = go.Figure(data=[go.Pie(
                labels=['News-Based Features', 'Technical Features'],
                values=[news_pct, technical_pct],
                hole=.3,
                marker_colors=['orange', 'royalblue']
            )])
            
            fig.update_layout(
                title="Prediction Influence Breakdown",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Die wichtigsten Faktoren erklären
    st.subheader("Key Factor Explanations")
    
    factor_explanations = {
        'news_sentiment_avg': "Average sentiment of recent news articles (positive/negative)",
        'news_impact_score': "Estimated market impact of relevant news",
        'news_positive_ratio': "Proportion of positive news articles",
        'news_negative_ratio': "Proportion of negative news articles",
        'news_sentiment_ma': "Moving average of news sentiment over time",
        'lag_': "Previous price values from N hours ago",
        'ma_': "Moving average price over N hours",
        'std_': "Price volatility over N hours",
        'momentum_': "Price change over N hours",
        'return_': "Percentage return over N hours",
        'volatility_': "Volatility measured over N hours",
        'hour': "Hour of the day",
        'day_of_week': "Day of the week (0=Monday, 6=Sunday)",
        'market_open': "Whether the market is open",
        'high_low_diff': "Range between high and low prices",
        'open_close_diff': "Difference between opening and closing prices"
    }
    
    for _, row in top_features.head(5).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Passende Erklärung finden
        explanation = "Technical or news indicator"
        for key, value in factor_explanations.items():
            if key in feature:
                explanation = value
                break
        
        # Feature-Typ bestimmen
        feature_type = "📰 News-Based" if feature.startswith('news_') else "📊 Technical"
        
        st.markdown(f"**{feature_type}: {feature}** (Importance: {importance:.4f})")
        st.markdown(f"> {explanation}")

# Jüngste Nachrichten in einem ausklappbaren Bereich anzeigen
with st.expander("View Recent News"):
    # Nachrichten nach Datum sortieren
    sorted_news = sorted(news_data, key=lambda x: datetime.strptime(x['datetime'], "%Y-%m-%d %H:%M:%S"), reverse=True)
    
    # Nur Nachrichten der letzten 3 Tage anzeigen
    cutoff_date = current_time - timedelta(days=3)
    recent_news = [n for n in sorted_news if datetime.strptime(n['datetime'], "%Y-%m-%d %H:%M:%S") >= cutoff_date]
    
    if recent_news:
        for news in recent_news[:10]:  # Top 10 anzeigen
            # Icon basierend auf Sentiment
            if news['sentiment'] == 'positive':
                icon = "📈"
            elif news['sentiment'] == 'negative':
                icon = "📉"
            else:
                icon = "📊"
            
            st.markdown(f"**{news['datetime']}** {icon} **{news['headline']}**")
            st.markdown(f"Source: {news['source']} | Sentiment: {news['sentiment']} | Keywords: {', '.join(news['keywords'])}")
            st.markdown("---")
    else:
        st.write("No recent news available.")

# Footer
st.markdown("---")
st.markdown("""
This dashboard uses financial news and market data to predict market movements. 
The predictions are based on both technical analysis and news sentiment analysis.
The system identifies key news stories that are likely to influence the market and incorporates 
their sentiment and relevance into the prediction model.
""")
