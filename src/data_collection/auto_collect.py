#!/usr/bin/env python3
import os
import time
import schedule
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_collector')

# Verzeichnis für Daten
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Definition der zu sammelnden Indizes
INDICES = {
    "DAX": "^GDAXI",
    "DowJones": "^DJI",
    "USD_EUR": "EURUSD=X"
}

def collect_financial_data():
    """Sammelt aktuelle Finanzdaten und speichert sie im Datenverzeichnis"""
    logger.info("Starte Datensammlung...")
    
    # Aktuelles Datum und Zeit für Dateibenennung
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, ticker in INDICES.items():
        try:
            logger.info(f"Sammle Daten für {name} ({ticker})...")
            
            # Herunterladen der Daten (1 Monat, stündliche Intervalle)
            # Reicht für 7 Tage Anzeige im Dashboard und historische Analysen
            data = yf.download(ticker, period="1mo", interval="1h")
            
            if not data.empty:
                # Datei speichern
                file_path = os.path.join(DATA_DIR, f"{name}_{timestamp}.csv")
                data.to_csv(file_path)
                logger.info(f"Daten für {name} gespeichert: {file_path} ({len(data)} Datenpunkte)")
                
                # Alte Dateien bereinigen (optional)
                cleanup_old_files(name)
            else:
                logger.warning(f"Keine Daten für {name} gefunden")
        except Exception as e:
            logger.error(f"Fehler beim Sammeln der Daten für {name}: {e}")
    
    logger.info("Datensammlung abgeschlossen")

def cleanup_old_files(index_name, keep_days=5):
    """Löscht alte Dateien, um Speicherplatz zu sparen"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{index_name}_")]
        if len(files) <= 5:  # Behalte mindestens 5 Dateien
            return
            
        # Sortiere nach Erstellungsdatum (älteste zuerst)
        files.sort(key=lambda f: os.path.getctime(os.path.join(DATA_DIR, f)))
        
        # Lösche ältere Dateien und behalte die neuesten
        for file in files[:-5]:
            file_path = os.path.join(DATA_DIR, file)
            os.remove(file_path)
            logger.info(f"Alte Datei gelöscht: {file_path}")
    except Exception as e:
        logger.error(f"Fehler beim Bereinigen alter Dateien: {e}")

def run_scheduled_collection():
    """Führt die Datensammlung aus und plant die nächste"""
    try:
        collect_financial_data()
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei der geplanten Datensammlung: {e}")

# Planung der regelmäßigen Datensammlung
def setup_schedule():
    """Richtet den Zeitplan für die Datensammlung ein"""
    # Sammle während der Handelszeiten häufiger Daten
    schedule.every(30).minutes.do(run_scheduled_collection)
    
    # Führe die erste Sammlung sofort aus
    run_scheduled_collection()
    
    logger.info("Zeitplan für Datensammlung eingerichtet. Sammle alle 30 Minuten.")
    
    # Endlosschleife für den Scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Prüfe jede Minute auf ausstehende Aufgaben

if __name__ == "__main__":
    setup_schedule()
