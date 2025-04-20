import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import traceback

class FinancialFeatureEngineer:
    def __init__(self, db_path="data/financial_data.db", output_dir="data/processed"):
        self.db_path = db_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_complete_features(self, index_name, days=30):
        """
        Erstellt einen vollständigen Feature-Datensatz für das Modelltraining

        Args:
            index_name: Name des Index ('DAX', 'DowJones', 'USD_EUR')
            days: Anzahl der Tage in der Vergangenheit für die Daten

        Returns:
            DataFrame mit allen Features
        """
        # Benutze die Daten direkt aus den CSV-Dateien, da wir noch keine Datenbank haben
        data_dir = "data/raw"

        # Finde die neueste Datei für den angegebenen Index
        file_pattern = f"{index_name}_*.csv"
        files = [f for f in os.listdir(data_dir) if f.startswith(index_name)]

        if not files:
            print(f"Keine Daten für {index_name} gefunden")
            return None

        # Sortiere nach Erstellungsdatum, neueste zuerst
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(data_dir, f)))
        file_path = os.path.join(data_dir, latest_file)

        # Lade die Marktdaten
        try:
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            print(f"Daten geladen aus {file_path}: {len(df)} Datenpunkte")
        except Exception as e:
            print(f"Fehler beim Laden der Daten aus {file_path}: {e}")
            return None

        # Bereinige die Daten
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
        for col in required_cols:
            if col not in df.columns:
                # Bei Devisendaten wie USD/EUR fehlt oft das Volumen
                if col == 'volume' and index_name == 'USD_EUR':
                    df['volume'] = 0  # Platzhalter für fehlende Volume-Daten
                else:
                    print(f"Warnung: Spalte {col} fehlt in den Daten")
                    return None

        # Beschränke auf die gewünschte Anzahl von Tagen
        if len(df) > days * 24:  # Ungefähr 24 Stunden pro Tag
            df = df.tail(days * 24)

        # Technische Indikatoren hinzufügen
        try:
            self._add_technical_indicators(df)
        except Exception as e:
            print(f"Fehler beim Hinzufügen technischer Indikatoren: {e}")
            traceback.print_exc()

        # Zeitliche Features hinzufügen
        try:
            self._add_time_features(df)
        except Exception as e:
            print(f"Fehler beim Hinzufügen zeitlicher Features: {e}")
            traceback.print_exc()

        # Lagging/Target Features hinzufügen (für Prognosen)
        try:
            self._add_target_features(df)
        except Exception as e:
            print(f"Fehler beim Hinzufügen von Ziel-Features: {e}")
            traceback.print_exc()

        # Entferne die ersten 20 Zeilen, da einige technische Indikatoren
        # einen bestimmten Verlauf benötigen und anfangs NaN sein können
        if len(df) > 20:
            df = df.iloc[20:]

        # Fülle verbleibende fehlende Werte
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # Speichere verarbeitete Daten
        output_file = os.path.join(self.output_dir, f"{index_name}_features.csv")
        df.to_csv(output_file)
        print(f"Feature-Datensatz für {index_name} gespeichert unter {output_file}")

        return df

    def _add_technical_indicators(self, df):
        """Fügt technische Indikatoren zu den Daten hinzu"""
        # Stelle sicher, dass alle benötigten Spalten vorhanden sind
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warnung: Spalte {col} fehlt, kann einige Indikatoren nicht berechnen")
                return

        try:
            # --- Trendfolge-Indikatoren ---
            # Gleitende Durchschnitte
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()

            # Exponentiell gleitende Durchschnitte
            df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

            # MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # RSI (vereinfachte Implementierung)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

            # --- Eigene Features ---
            # Preisänderungen
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)

            # Volatilität (rollierende Standardabweichung)
            df['volatility_5'] = df['close'].rolling(window=5).std() / df['close']
            df['volatility_10'] = df['close'].rolling(window=10).std() / df['close']

            # Relative Position des aktuellen Preises zum gleitenden Durchschnitt
            df['price_sma_ratio_5'] = df['close'] / df['sma_5']
            df['price_sma_ratio_20'] = df['close'] / df['sma_20']

            # Richtungsänderungsindikatoren
            df['direction_1'] = np.sign(df['price_change_1'])
            df['direction_5'] = np.sign(df['price_change_5'])

            # Volumen-Features (für Indizes mit Volumen)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_change'] = df['volume'].pct_change(1)
                df['volume_ratio'] = df['volume'] / df['volume_sma_5']
                df['pv_ratio'] = df['close'] * df['volume']
                df['pv_ratio_change'] = df['pv_ratio'].pct_change(1)

        except Exception as e:
            print(f"Fehler bei der Berechnung technischer Indikatoren: {e}")
            traceback.print_exc()

    def _add_time_features(self, df):
        """Fügt zeitbasierte Features hinzu"""
        # Extrahiere verschiedene Zeitkomponenten
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek  # 0=Montag, 6=Sonntag
        df['month'] = df.index.month
        df['year'] = df.index.year

        # Sinus- und Kosinus-Transformation für zyklische Features
        # Stunden des Tages (24-Stunden-Zyklus)
        hours_in_day = 24
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / hours_in_day))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / hours_in_day))

        # Tage der Woche (7-Tage-Zyklus)
        days_in_week = 7
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / days_in_week))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / days_in_week))

        # Monate des Jahres (12-Monats-Zyklus)
        months_in_year = 12
        df['month_sin'] = np.sin(df['month'] * (2 * np.pi / months_in_year))
        df['month_cos'] = np.cos(df['month'] * (2 * np.pi / months_in_year))

        # Handelszeiten-Indikatoren (vereinfacht)
        # Typische Handelszeiten: 9:00-17:30 für europäische Märkte
        df['is_trading_hour'] = ((df.index.hour >= 9) & (df.index.hour < 18)).astype(int)

        # Unterscheide zwischen Handelstagen (Mo-Fr) und Wochenenden
        df['is_trading_day'] = (df.index.dayofweek < 5).astype(int)

        # Kombination: Handelszeit während eines Handelstages
        df['is_active_trading'] = df['is_trading_hour'] * df['is_trading_day']

    def _add_target_features(self, df):
        """Fügt Ziel-Features für die Vorhersage hinzu"""
        # Zukünftige Preisänderungen als Targets
        df['target_1h'] = df['close'].shift(-1)  # Preis in 1 Stunde
        df['target_4h'] = df['close'].shift(-4)  # Preis in 4 Stunden
        df['target_8h'] = df['close'].shift(-8)  # Preis in 8 Stunden (Handelsschluss)
        df['target_24h'] = df['close'].shift(-24)  # Preis in 24 Stunden (nächster Handelsstart)

        # Prozentuale Änderungen
        df['target_return_1h'] = df['close'].pct_change(-1)  # Rendite in 1 Stunde
        df['target_return_4h'] = df['close'].pct_change(-4)  # Rendite in 4 Stunden
        df['target_return_8h'] = df['close'].pct_change(-8)  # Rendite in 8 Stunden
        df['target_return_24h'] = df['close'].pct_change(-24)  # Rendite in 24 Stunden

        # Richtung (binär: steigen oder fallen)
        df['target_direction_1h'] = np.where(df['target_return_1h'] > 0, 1, 0)
        df['target_direction_4h'] = np.where(df['target_return_4h'] > 0, 1, 0)
        df['target_direction_8h'] = np.where(df['target_return_8h'] > 0, 1, 0)
        df['target_direction_24h'] = np.where(df['target_return_24h'] > 0, 1, 0)

    def scale_features(self, df, index_name, exclude_cols=None):
        """
        Skaliert die Features für das Modelltraining

        Args:
            df: DataFrame mit Features
            index_name: Name des Index für die Skalierung
            exclude_cols: Liste von Spalten, die nicht skaliert werden sollen

        Returns:
            DataFrame mit skalierten Features und Liste der skalierten Spalten
        """
        if exclude_cols is None:
            exclude_cols = []

        # Identifiziere Zielspalten und füge sie zu exclude_cols hinzu
        target_cols = [col for col in df.columns if col.startswith('target_')]
        exclude_cols.extend(target_cols)

        # Identifiziere kategorische Spalten und füge sie zu exclude_cols hinzu
        categorical_cols = ['hour', 'day', 'day_of_week', 'month', 'year',
                           'is_trading_hour', 'is_trading_day', 'is_active_trading']
        exclude_cols.extend(categorical_cols)

        # Identifiziere Zeitstempelspalten und füge sie zu exclude_cols hinzu
        date_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        exclude_cols.extend(date_cols)

        # Entferne Duplikate aus exclude_cols
        exclude_cols = list(set(exclude_cols))

        # Identifiziere Spalten, die skaliert werden sollen
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Erstelle Modellverzeichnis, falls nicht vorhanden
        model_dir = os.path.join("data/models", index_name)
        os.makedirs(model_dir, exist_ok=True)

        # Erstelle und trainiere den Scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_scaled = df.copy()

        if feature_cols:
            df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

            # Speichere den Scaler für spätere Verwendung
            scaler_file = os.path.join(model_dir, "feature_scaler.pkl")
            joblib.dump(scaler, scaler_file)
            print(f"Feature-Scaler gespeichert unter {scaler_file}")

        # Speichere skalierte Daten
        scaled_file = os.path.join(self.output_dir, f"{index_name}_scaled_features.csv")
        df_scaled.to_csv(scaled_file)
        print(f"Skalierte Features für {index_name} gespeichert unter {scaled_file}")

        return df_scaled, feature_cols

    def process_all_indices(self, days=30):
        """Verarbeitet alle Indizes"""
        indices = ["DAX", "DowJones", "USD_EUR"]
        results = {}

        for index_name in indices:
            print(f"\nVerarbeite Features für {index_name}...")
            df = self.create_complete_features(index_name, days=days)

            if df is not None and not df.empty:
                # Skaliere Features
                scaled_df, _ = self.scale_features(df, index_name)
                results[index_name] = scaled_df
            else:
                print(f"Keine Daten für {index_name} verfügbar")

        return results

if __name__ == "__main__":
    engineer = FinancialFeatureEngineer()
    engineer.process_all_indices(days=30)
