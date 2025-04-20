import pandas as pd
import numpy as np
import os
import sqlite3
import re
from datetime import datetime, timedelta
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# NLTK-Ressourcen herunterladen
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

class NewsTextProcessor:
    def __init__(self, db_path="data/financial_data.db", news_dir="data/raw"):
        self.db_path = db_path
        self.news_dir = news_dir
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Erstelle die Datenbank-Verbindung
        self._init_database()

    def _init_database(self):
        """Initialisiert die Datenbank mit den erforderlichen Tabellen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Erstelle die Sentiment-Tabelle, falls sie noch nicht existiert
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            sentiment_mean REAL,
            sentiment_weighted REAL,
            news_count INTEGER,
            top_topics TEXT,
            processed_at TEXT
        )
        ''')

        # Erstelle Index für schnellere Abfragen
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_index_timestamp ON sentiment_data (index_name, timestamp)')

        conn.commit()
        conn.close()

    def _clean_text(self, text):
        """Bereinigt den Text für die Verarbeitung"""
        if not isinstance(text, str):
            return ""

        # Entferne HTML-Tags
        text = re.sub(r'<.*?>', '', text)

        # Entferne URLs
        text = re.sub(r'http\S+', '', text)

        # Entferne Sonderzeichen und konvertiere zu Kleinbuchstaben
        text = re.sub(r'[^\w\s]', '', text).lower()

        return text

    def _analyze_sentiment(self, text):
        """Analysiert das Sentiment eines Textes"""
        text = self._clean_text(text)
        if not text:
            return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}

        return self.sentiment_analyzer.polarity_scores(text)

    def _calculate_weighted_sentiment(self, df):
        """Berechnet gewichtetes Sentiment basierend auf Titel und Inhalt"""
        if df.empty:
            return pd.DataFrame()

        # Füge Spalten für Sentiment hinzu
        df['title_sentiment'] = df['title'].apply(
            lambda x: self._analyze_sentiment(x)['compound'] if isinstance(x, str) else 0
        )

        # Verwende Beschreibung, wenn Inhalt nicht verfügbar ist
        df['content_text'] = df.apply(
            lambda row: row['content'] if pd.notnull(row['content']) else
                       (row['description'] if pd.notnull(row['description']) else ""),
            axis=1
        )

        df['content_sentiment'] = df['content_text'].apply(
            lambda x: self._analyze_sentiment(x)['compound'] if isinstance(x, str) else 0
        )

        # Berechne gewichtetes Sentiment (Titel hat höheres Gewicht)
        df['weighted_sentiment'] = df['title_sentiment'] * 0.6 + df['content_sentiment'] * 0.4

        return df

    def _extract_topics(self, texts, n_topics=3, n_words=5):
        """Extrahiert Hauptthemen aus einer Sammlung von Texten"""
        if not texts or len(texts) < 5:  # Zu wenige Texte für vernünftige Themenextraktion
            return []

        # Bereinige Texte
        cleaned_texts = [self._clean_text(text) for text in texts if isinstance(text, str)]

        if not cleaned_texts:
            return []

        try:
            # Vectorizer für Bag-of-Words
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(cleaned_texts)

            # LDA-Modell zur Themenextraktion
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(dtm)

            # Feature-Namen (Wörter)
            words = vectorizer.get_feature_names_out()

            # Extrahiere Top-Wörter für jedes Thema
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-n_words:]
                top_words = [words[i] for i in top_words_idx]
                topics.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")

            return topics
        except Exception as e:
            print(f"Fehler bei der Themenextraktion: {e}")
            return []

    def process_news_files(self, index_name=None):
        """Verarbeitet News-Dateien und extrahiert Sentiment-Daten"""
        # Bestimme Dateipfadmuster für die Suche
        if index_name:
            file_pattern = os.path.join(self.news_dir, f"news_{index_name}_*.csv")
        else:
            file_pattern = os.path.join(self.news_dir, "news_*.csv")

        # Finde passende Dateien
        news_files = glob.glob(file_pattern)

        if not news_files:
            print(f"Keine News-Dateien gefunden für Muster: {file_pattern}")
            return pd.DataFrame()

        processed_data = []

        for file in news_files:
            try:
                # Extrahiere Index-Namen aus dem Dateinamen
                filename = os.path.basename(file)
                if "_" in filename:
                    parts = filename.split("_")
                    if len(parts) > 1:
                        extracted_index = parts[1].split(".")[0]

                        # Überprüfe, ob es sich um einen der unterstützten Indizes handelt
                        if extracted_index in ["DAX", "DowJones", "USD_EUR"]:
                            current_index = extracted_index
                        else:
                            # Wenn nicht erkannt, verwende den angegebenen Index oder Standard
                            current_index = index_name or "unknown"
                    else:
                        current_index = index_name or "unknown"
                else:
                    current_index = index_name or "unknown"

                # Lade die News-Daten
                df = pd.read_csv(file)

                if df.empty:
                    print(f"Leere Datei übersprungen: {file}")
                    continue

                # Konvertiere Timestamp
                if 'published_at' in df.columns:
                    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

                # Füge Sentiment hinzu
                df = self._calculate_weighted_sentiment(df)

                # Gruppiere nach Stunden und berechne durchschnittliches Sentiment
                if 'published_at' in df.columns and not df['published_at'].isnull().all():
                    df['hour'] = df['published_at'].dt.floor('H')

                    hourly_data = df.groupby('hour').agg({
                        'weighted_sentiment': 'mean',
                        'title': 'count'
                    }).reset_index()

                    hourly_data.columns = ['timestamp', 'sentiment_mean', 'news_count']

                    # Extrahiere Themen für jede Stunde
                    for idx, row in hourly_data.iterrows():
                        timestamp = row['timestamp']
                        hour_news = df[df['hour'] == timestamp]

                        # Sammle Texte für Themenextraktion
                        texts = []
                        for _, news_row in hour_news.iterrows():
                            if isinstance(news_row.get('title'), str):
                                texts.append(news_row['title'])
                            if isinstance(news_row.get('content_text'), str):
                                texts.append(news_row['content_text'])

                        topics = self._extract_topics(texts)

                        processed_data.append({
                            'index_name': current_index,
                            'timestamp': timestamp,
                            'sentiment_mean': row['sentiment_mean'],
                            'sentiment_weighted': row['sentiment_mean'],  # Gleiches Sentiment für jetzt
                            'news_count': row['news_count'],
                            'top_topics': '; '.join(topics) if topics else '',
                            'processed_at': datetime.now().isoformat()
                        })

            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {file}: {e}")
                import traceback
                traceback.print_exc()

        return pd.DataFrame(processed_data)

    def save_sentiment_to_db(self, sentiment_df):
        """Speichert Sentiment-Daten in der Datenbank"""
        if sentiment_df.empty:
            print("Keine Sentiment-Daten zum Speichern vorhanden")
            return

        conn = sqlite3.connect(self.db_path)

        try:
            # Füge Daten in die Datenbank ein
            sentiment_df.to_sql('sentiment_data', conn, if_exists='append', index=False)
            print(f"{len(sentiment_df)} Sentiment-Datensätze in die Datenbank geschrieben")
        except Exception as e:
            print(f"Fehler beim Speichern in die Datenbank: {e}")
        finally:
            conn.close()

    def run_sentiment_processing(self):
        """Führt die gesamte Sentiment-Verarbeitung aus"""
        indices = ["DAX", "DowJones", "USD_EUR"]

        for index_name in indices:
            print(f"\nVerarbeite Nachrichten für {index_name}...")
            sentiment_df = self.process_news_files(index_name)

            if not sentiment_df.empty:
                self.save_sentiment_to_db(sentiment_df)
                print(f"Sentiment für {index_name} erfolgreich verarbeitet und gespeichert")
            else:
                print(f"Keine Sentiment-Daten für {index_name} gefunden")

if __name__ == "__main__":
    processor = NewsTextProcessor()
    processor.run_sentiment_processing()
