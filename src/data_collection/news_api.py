import requests
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

class NewsAPICollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("News API-Schlüssel fehlt. Bitte in .env-Datei oder als Parameter angeben.")
        self.base_url = "https://newsapi.org/v2"
        
    def get_financial_news(self, keywords=None, from_date=None, to_date=None, language="de,en"):
        """Sammelt Finanznachrichten basierend auf Schlüsselwörtern und Zeitraum"""
        if keywords is None:
            keywords = ["DAX", "Dow Jones", "USD/EUR", "Finanzen", "Wirtschaft", 
                         "Aktienmarkt", "Inflation", "Zinsen", "EZB", "Fed"]
            
        # Anfrageparameter
        params = {
            "apiKey": self.api_key,
            "q": " OR ".join(keywords),
            "language": language,
            "sortBy": "publishedAt"
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        print(f"Sammle Nachrichten mit Keywords: {keywords}")
        response = requests.get(f"{self.base_url}/everything", params=params)
        
        if response.status_code == 200:
            news_data = response.json()
            
            # Umwandlung in DataFrame
            articles = []
            for article in news_data.get("articles", []):
                articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "source": article.get("source", {}).get("name"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "collected_at": datetime.now().isoformat()
                })
            
            df = pd.DataFrame(articles)
            print(f"Gesammelt: {len(df)} Artikeln")
            return df
        else:
            print(f"Fehler bei der Nachrichtenabfrage: {response.status_code}, {response.text}")
            return pd.DataFrame()
    
    def get_news_for_indices(self, days_back=7):
        """Sammelt Nachrichten speziell für DAX, Dow Jones und USD/EUR"""
        from_date = (datetime.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        indices_news = {}
        
        # DAX-spezifische Nachrichten
        dax_keywords = ["DAX", "Deutscher Aktienindex", "Frankfurt Börse", "Deutsche Wirtschaft"]
        indices_news["DAX"] = self.get_financial_news(dax_keywords, from_date=from_date)
        
        # Dow Jones-spezifische Nachrichten
        dow_keywords = ["Dow Jones", "DJIA", "US Wirtschaft", "Wall Street", "US Aktienmarkt"]
        indices_news["DowJones"] = self.get_financial_news(dow_keywords, from_date=from_date)
        
        # USD/EUR-spezifische Nachrichten
        forex_keywords = ["USD/EUR", "Euro Dollar", "Wechselkurs", "Devisenmarkt", "Währung"]
        indices_news["USD_EUR"] = self.get_financial_news(forex_keywords, from_date=from_date)
        
        return indices_news
    
    def save_news_data(self, news_data, output_dir="../../data/raw"):
        """Speichert Nachrichtendaten als CSV-Dateien"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(news_data, dict):
            for index_name, df in news_data.items():
                if not df.empty:
                    file_path = os.path.join(output_dir, f"news_{index_name}_{timestamp}.csv")
                    df.to_csv(file_path, index=False)
                    print(f"Nachrichten für {index_name} gespeichert unter {file_path}")
        else:
            # Einzelnes DataFrame
            file_path = os.path.join(output_dir, f"news_general_{timestamp}.csv")
            news_data.to_csv(file_path, index=False)
            print(f"Allgemeine Nachrichten gespeichert unter {file_path}")

if __name__ == "__main__":
    collector = NewsAPICollector()
    all_news = collector.get_news_for_indices()
    collector.save_news_data(all_news)
