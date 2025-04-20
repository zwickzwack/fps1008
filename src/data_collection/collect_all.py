from news_api import NewsAPICollector
from financial_data import FinancialDataCollector
import os
from datetime import datetime

def collect_all_data():
    """Sammelt alle Daten und speichert sie"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starte Datensammlung: {timestamp}")
    
    # Sammle Nachrichtendaten
    try:
        print("\n=== Sammle Nachrichtendaten ===")
        news_collector = NewsAPICollector()
        all_news = news_collector.get_news_for_indices(days_back=7)
        news_collector.save_news_data(all_news)
    except Exception as e:
        print(f"Fehler bei der Nachrichtensammlung: {e}")
    
    # Sammle Finanzdaten
    try:
        print("\n=== Sammle Finanzdaten ===")
        finance_collector = FinancialDataCollector()
        
        # Historische Daten (1 Monat, stündlich)
        print("\nSammle historische Daten (1 Monat, stündlich):")
        historical_data = finance_collector.collect_all_indices(period="1mo", interval="1h")
        finance_collector.save_data(historical_data)
        
        # Detaillierte Daten (1 Woche, 15min)
        print("\nSammle detaillierte Daten (1 Woche, 15min):")
        detailed_data = finance_collector.collect_all_indices(period="1wk", interval="15m")
        finance_collector.save_data(detailed_data)
    except Exception as e:
        print(f"Fehler bei der Finanzdatensammlung: {e}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nDatensammlung abgeschlossen: {timestamp}")

if __name__ == "__main__":
    collect_all_data()
