#!/bin/bash

# Installiere benötigte Pakete, falls noch nicht vorhanden
pip3 install streamlit pandas plotly holidays pytz

# Starte das Dashboard
cd "$(dirname "$0")"
streamlit run src/dashboard/enhanced_app.py
