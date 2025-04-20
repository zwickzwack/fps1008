import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Seitentitel
st.title("Einfaches Finanzindex-Dashboard")

# Datum und Beschreibung
st.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write("Ein einfaches Dashboard zur Anzeige von Finanzdaten")

# Dummy-Daten generieren
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
data = np.random.normal(loc=100, scale=10, size=30).cumsum() + 1000

# Einfaches Diagramm erstellen
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, data)
ax.set_title('Einfacher Finanzindex')
ax.set_xlabel('Datum')
ax.set_ylabel('Wert')
st.pyplot(fig)

# Tabelle mit den letzten 5 Datenpunkten
st.write("Letzte Datenpunkte:")
df = pd.DataFrame({'Datum': dates[-5:], 'Wert': data[-5:]})
st.table(df)
