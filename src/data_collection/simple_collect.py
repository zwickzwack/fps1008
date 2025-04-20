import requests
import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# API-Schlüssel direkt im Skript definieren (in der Praxis sollte man das vermeiden)
NEWS_API_KEY = "3305d13414674fca802ff42416233a10"
ALPHA_VANTAGE_KEY = "7D5ST456FJKG0YXN"

def collect_financial_data():
    """Sammelt Finanzdaten für die wichtigsten Indizes"""
    
    # Indizes definieren
    indices = {
        "DAX": "^GDAXI",
        "DowJones": "^DJI",
        "USD_EUR": "EURUSD=X"
    }
    
    # Berechne absolute Pfade basierend auf dem Skript-Standort
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    output_dir = os.path.join(project_dir, "data/raw")
    
    print(f"Ausgabeverzeichnis: {output_dir}")
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # Zeitstempel für Dateibenennung
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Daten für jeden Index sammeln
    for name, ticker in indices.items():
        print(f"Sammle Daten für {name} ({ticker})...")
        
        try:
            # Daten herunterladen (1 Monat, stündliche Intervalle)
            data = yf.download(ticker, period="1mo", interval="1h")
            
            if not data.empty:
                # Datei speichern
                file_path = os.path.join(output_dir, f"{name}_{timestamp}.csv")
                data.to_csv(file_path)
                print(f"Daten für {name} gespeichert unter {file_path} ({len(data)} Datenpunkte)")
            else:
                print(f"Keine Daten für {name} gefunden")
                
        except Exception as e:
            print(f"Fehler beim Sammeln von Daten für {name}: {e}")
    
    print("Datensammlung abgeschlossen!")

if __name__ == "__main__":
    collect_financial_data()
