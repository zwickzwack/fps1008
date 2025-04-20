import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

class FinancialDataCollector:
    def __init__(self):
        # Mapping der Indizes zu den entsprechenden Tickers
        self.index_mapping = {
            "DAX": "^GDAXI",
            "DowJones": "^DJI",
            "USD_EUR": "EURUSD=X"
        }
        
        self.alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        
    def collect_historical_data(self, index_name, period="1mo", interval="1h"):
        """
        Sammelt historische Daten für den angegebenen Index mit Yahoo Finance
        
        Args:
            index_name: Name des Index ("DAX", "DowJones", "USD_EUR")
            period: Zeitraum (z.B. "1d", "1mo", "1y")
            interval: Intervall (z.B. "1m", "1h", "1d")
            
        Returns:
            DataFrame mit historischen Daten
        """
        if index_name not in self.index_mapping:
            raise ValueError(f"Unbekannter Index: {index_name}. Verfügbare Indizes: {list(self.index_mapping.keys())}")
            
        ticker = self.index_mapping[index_name]
        print(f"Sammle {index_name} ({ticker}) Daten mit Intervall {interval} für Zeitraum {period}")
        
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval)
            
            # Speichere Metadaten
            data["index_name"] = index_name
            data["ticker"] = ticker
            data["collected_at"] = datetime.now().isoformat()
            
            print(f"Erfolgreich gesammelt: {len(data)} Datenpunkte")
            return data
        except Exception as e:
            print(f"Fehler beim Sammeln von Daten für {index_name}: {e}")
            return pd.DataFrame()
    
    def collect_alpha_vantage_data(self, index_name, interval="60min", output_size="full"):
        """
        Sammelt Daten von Alpha Vantage als alternative Quelle
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API-Schlüssel fehlt. Überspringen.")
            return None
            
        base_url = "https://www.alphavantage.co/query"
        
        # Mapping für Alpha Vantage Funktionen und Symbole
        av_mapping = {
            "DAX": {"function": "TIME_SERIES_INTRADAY", "symbol": "DAX"},
            "DowJones": {"function": "TIME_SERIES_INTRADAY", "symbol": "DJI"},
            "USD_EUR": {"function": "FX_INTRADAY", "from_symbol": "USD", "to_symbol": "EUR"}
        }
        
        if index_name not in av_mapping:
            print(f"Index {index_name} nicht unterstützt für Alpha Vantage")
            return None
            
        params = {
            "apikey": self.alpha_vantage_key,
            "interval": interval,
            "outputsize": output_size,
            "datatype": "json"
        }
        
        if index_name == "USD_EUR":
            params["function"] = av_mapping[index_name]["function"]
            params["from_symbol"] = av_mapping[index_name]["from_symbol"]
            params["to_symbol"] = av_mapping[index_name]["to_symbol"]
            time_series_key = f"Time Series FX ({interval})"
        else:
            params["function"] = av_mapping[index_name]["function"]
            params["symbol"] = av_mapping[index_name]["symbol"]
            time_series_key = f"Time Series ({interval})"
            
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if "Error Message" in data:
                print(f"Alpha Vantage Fehler: {data['Error Message']}")
                return None
                
            if time_series_key not in data:
                print(f"Keine Zeitreihendaten gefunden in Alpha Vantage Antwort: {data.keys()}")
                return None
                
            # Konvertiere in DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns for consistency
            column_mapping = {
                "1. open": "Open", 
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert string values to float
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                    
            # Set datetime index
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Add metadata
            df["index_name"] = index_name
            df["source"] = "Alpha Vantage"
            df["collected_at"] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            print(f"Fehler bei Alpha Vantage Anfrage für {index_name}: {e}")
            return None
    
    def collect_all_indices(self, period="1mo", interval="1h"):
        """Sammelt Daten für alle definierten Indizes"""
        all_data = {}
        
        for index_name in self.index_mapping:
            try:
                # Versuche erst Yahoo Finance
                data = self.collect_historical_data(index_name, period, interval)
                
                # Wenn Yahoo Finance versagt, versuche Alpha Vantage als Fallback
                if data.empty and self.alpha_vantage_key:
                    print(f"Versuche Alpha Vantage als Fallback für {index_name}")
                    av_interval = interval.replace("m", "min")
                    data = self.collect_alpha_vantage_data(index_name, interval=av_interval)
                
                if data is not None and not data.empty:
                    all_data[index_name] = data
                    print(f"Daten für {index_name} erfolgreich gesammelt")
                else:
                    print(f"Keine Daten für {index_name} verfügbar")
            except Exception as e:
                print(f"Fehler beim Sammeln von Daten für {index_name}: {e}")
                
        return all_data
    
    def save_data(self, data, output_dir="../../data/raw"):
        """Speichert gesammelte Daten als CSV-Dateien"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(data, dict):
            for index_name, df in data.items():
                if not df.empty:
                    file_path = os.path.join(output_dir, f"{index_name}_{timestamp}.csv")
                    df.to_csv(file_path)
                    print(f"Daten für {index_name} gespeichert unter {file_path}")
        else:
            # Einzelnes DataFrame
            index_name = data.get("index_name", "")[0] if "index_name" in data.columns else "financial_data"
            file_path = os.path.join(output_dir, f"{index_name}_{timestamp}.csv")
            data.to_csv(file_path)
            print(f"Daten gespeichert unter {file_path}")

if __name__ == "__main__":
    collector = FinancialDataCollector()
    # Sammle einen Monat stündlicher Daten
    all_data = collector.collect_all_indices(period="1mo", interval="1h")
    collector.save_data(all_data)
    
    # Zusätzlich: Sammle eine Woche 15-minütiger Daten für detailliertere Analysen
    detailed_data = collector.collect_all_indices(period="1wk", interval="15m")
    collector.save_data(detailed_data)
