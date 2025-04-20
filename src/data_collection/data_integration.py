import os
import pandas as pd
import glob
from datetime import datetime, timedelta
import sqlite3

class DataIntegrator:
    def __init__(self, base_path="../data"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        self.db_path = os.path.join(base_path, "financial_data.db")

        # Erstelle Verzeichnisse, falls nicht vorhanden
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

        # Initialisiere die Datenbank, falls nicht vorhanden
        self._init_database()

    def _init_database(self):
        """Initialisiert die SQLite-Datenbank mit Tabellen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Erstelle Tabelle für Finanzmarktdaten
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            index_name TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            collected_at DATETIME,
            UNIQUE(timestamp, index_name)
        )
        ''')

        # Erstelle Tabelle für Nachrichtendaten
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            content TEXT,
            source TEXT,
            url TEXT UNIQUE,
            published_at DATETIME,
            collected_at DATETIME,
            index_relevance TEXT,
            sentiment_score REAL
        )
        ''')

        # Erstelle Tabelle für aggregierte Sentiment-Daten
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            index_name TEXT NOT NULL,
            sentiment_mean REAL,
            sentiment_weighted REAL,
            news_count INTEGER,
            UNIQUE(timestamp, index_name)
        )
        ''')

        conn.commit()
        conn.close()

    def import_latest_financial_data(self):
        """Importiert die neuesten Finanzmarktdaten in die Datenbank"""
        conn = sqlite3.connect(self.db_path)

        # Finde die neuesten CSV-Dateien für jeden Index
        indices = ["DAX", "DowJones", "USD_EUR"]
        imported_count = 0

        for index_name in indices:
            pattern = os.path.join(self.raw_path, f"{index_name}_*.csv")
            files = glob.glob(pattern)

            if not files:
                print(f"Keine Daten gefunden für {index_name}")
                continue

            # Sortiere nach Dateierstellungsdatum, neueste zuerst
            latest_file = max(files, key=os.path.getctime)
            print(f"Importiere {index_name} Daten aus {os.path.basename(latest_file)}")

            try:
                # Lade Daten
                df = pd.read_csv(latest_file, parse_dates=True, index_col=0)
                df.reset_index(inplace=True)
                df.rename(columns={"index": "timestamp"}, inplace=True)

                # Füge Index-Name hinzu, falls nicht vorhanden
                if "index_name" not in df.columns:
                    df["index_name"] = index_name

                # Wähle relevante Spalten aus
                relevant_columns = ["timestamp", "index_name", "Open", "High", "Low", "Close", "Volume", "collected_at"]
                available_columns = [col for col in relevant_columns if col in df.columns]

                if len(available_columns) < 5:  # Mindestanzahl erforderlicher Spalten
                    print(f"Nicht genügend relevante Spalten in {latest_file}")
                    continue

                df_to_import = df[available_columns]

                # Importiere in die Datenbank
                df_to_import.to_sql("market_data", conn, if_exists="append", index=False)

                print(f"Erfolgreich importiert: {len(df_to_import)} Datenpunkte für {index_name}")
                imported_count += len(df_to_import)

            except Exception as e:
                print(f"Fehler beim Importieren von {latest_file}: {e}")

        conn.close()
        return imported_count

    def import_latest_news_data(self):
        """Importiert die neuesten Nachrichtendaten in die Datenbank"""
        conn = sqlite3.connect(self.db_path)

        # Finde die neuesten News-CSV-Dateien
        pattern = os.path.join(self.raw_path, "news_*.csv")
        files = glob.glob(pattern)

        if not files:
            print("Keine Nachrichtendaten gefunden")
            conn.close()
            return 0

        imported_count = 0

        for file in files:
            filename = os.path.basename(file)
            # Extrahiere den Indexnamen aus dem Dateinamen (Format: news_INDEX_TIMESTAMP.csv)
            parts = filename.split("_")
            if len(parts) >= 3:
                index_relevance = parts[1]
            else:
                index_relevance = "general"

            print(f"Importiere Nachrichtendaten mit Relevanz für {index_relevance} aus {filename}")

            try:
                # Lade Nachrichtendaten
                df = pd.read_csv(file)

                # Konvertiere Datumsangaben
                for date_col in ["published_at", "collected_at"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])

                # Füge Index-Relevanz hinzu
                df["index_relevance"] = index_relevance

                # Initialisiere Sentiment-Score mit 0
                if "sentiment_score" not in df.columns:
                    df["sentiment_score"] = 0.0

                # Importiere in die Datenbank
                df.to_sql("news_data", conn, if_exists="append", index=False)

                print(f"Erfolgreich importiert: {len(df)} Nachrichten für {index_relevance}")
                imported_count += len(df)

            except Exception as e:
                print(f"Fehler beim Importieren von {file}: {e}")

        conn.close()
        return imported_count

    def get_latest_market_data(self, index_name, days=7):
        """Holt die neuesten Marktdaten aus der Datenbank"""
        conn = sqlite3.connect(self.db_path)

        # Berechne Startdatum
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE index_name = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(index_name, start_date.isoformat(), end_date.isoformat()),
            parse_dates=["timestamp"]
        )

        conn.close()

        # Setze Timestamp als Index
        if not df.empty and "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)

        return df

    def get_relevant_news(self, index_name, days=7):
        """Holt relevante Nachrichten für einen Index aus der Datenbank"""
        conn = sqlite3.connect(self.db_path)

        # Berechne Startdatum
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = """
        SELECT title, description, content, source, url, published_at, sentiment_score
        FROM news_data
        WHERE (index_relevance = ? OR index_relevance = 'general')
              AND published_at BETWEEN ? AND ?
        ORDER BY published_at DESC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(index_name, start_date.isoformat(), end_date.isoformat()),
            parse_dates=["published_at"]
        )

        conn.close()
        return df

    def export_integrated_dataset(self, index_name, days=30):
        """
        Erstellt einen integrierten Datensatz mit Markt- und Sentiment-Daten
        für einen bestimmten Index und speichert ihn als CSV.
        """
        # Hole Marktdaten
        market_data = self.get_latest_market_data(index_name, days)

        if market_data.empty:
            print(f"Keine Marktdaten für {index_name} gefunden.")
            return None

        # Resample auf stündliche Intervalle, falls nötig
        if market_data.index.inferred_freq != 'H':
            market_data = market_data.resample('H').mean()

        # Hole relevante Nachrichten
        news_data = self.get_relevant_news(index_name, days)

        # Erstelle ein integriertes DataFrame
        integrated_df = market_data.copy()

        # Berechne aggregierte Sentiment-Daten für jeden Zeitpunkt
        if not news_data.empty:
            for timestamp in integrated_df.index:
                # 24-Stunden-Fenster vor dem aktuellen Zeitpunkt
                window_start = timestamp - timedelta(hours=24)
                relevant_news = news_data[
                    (news_data["published_at"] >= window_start) &
                    (news_data["published_at"] <= timestamp)
                ]

                if not relevant_news.empty:
                    integrated_df.at[timestamp, "sentiment_mean"] = relevant_news["sentiment_score"].mean()

                    # Gewichte neuere Nachrichten stärker
                    time_diffs = (timestamp - relevant_news["published_at"]).dt.total_seconds() / 3600  # in Stunden
                    weights = 1 / (1 + time_diffs)  # Höheres Gewicht für neuere Nachrichten

                    integrated_df.at[timestamp, "sentiment_weighted"] = (
                        relevant_news["sentiment_score"] * weights
                    ).sum() / weights.sum()

                    integrated_df.at[timestamp, "news_count"] = len(relevant_news)
                else:
                    integrated_df.at[timestamp, "sentiment_mean"] = 0
                    integrated_df.at[timestamp, "sentiment_weighted"] = 0
                    integrated_df.at[timestamp, "news_count"] = 0
        else:
            # Keine Nachrichtendaten verfügbar
            integrated_df["sentiment_mean"] = 0
            integrated_df["sentiment_weighted"] = 0
            integrated_df["news_count"] = 0

        # Fülle fehlende Werte mit Vorwärtsfüllung
        integrated_df = integrated_df.fillna(method="ffill")

        # Berechne technische Indikatoren
        # 1. Gleitende Durchschnitte
        integrated_df["MA_5"] = integrated_df["close"].rolling(window=5).mean()
        integrated_df["MA_20"] = integrated_df["close"].rolling(window=20).mean()

        # 2. Prozentuale Veränderungen
        integrated_df["return_1h"] = integrated_df["close"].pct_change(1)
        integrated_df["return_4h"] = integrated_df["close"].pct_change(4)

        # 3. Volatilität (Standardabweichung über 5 Perioden)
        integrated_df["volatility"] = integrated_df["close"].rolling(window=5).std()

        # Speichere als CSV
        output_file = os.path.join(self.processed_path, f"{index_name}_integrated.csv")
        integrated_df.to_csv(output_file)
        print(f"Integrierter Datensatz für {index_name} gespeichert unter {output_file}")

        return integrated_df

if __name__ == "__main__":
    integrator = DataIntegrator()

    # Importiere die neuesten Daten
    financial_count = integrator.import_latest_financial_data()
    news_count = integrator.import_latest_news_data()

    print(f"Importiert: {financial_count} Finanzmarktdaten, {news_count} Nachrichtendaten")

    # Erstelle integrierte Datensätze für alle Indizes
    for index in ["DAX", "DowJones", "USD_EUR"]:
        integrator.export_integrated_dataset(index)
