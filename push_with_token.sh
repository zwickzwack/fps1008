#!/bin/bash

# Aktuelles Datum einfügen
CURRENT_DATE="2025-04-19 18:07:50"
sed -i '' "s/current_time = datetime.strptime(\".*\", \"%Y-%m-%d %H:%M:%S\")/current_time = datetime.strptime(\"$CURRENT_DATE\", \"%Y-%m-%d %H:%M:%S\")/" src/dashboard/news_based_predictor.py

# Alle Änderungen committen
git add .
git commit -m "Update dashboard with current timestamp: $CURRENT_DATE"

echo "Bitte geben Sie Ihr GitHub Personal Access Token ein (wird nicht angezeigt):"
read -s TOKEN

# Mit Token pushen
git push https://zwickzwack:$TOKEN@github.com/zwickzwack/financial-prediction-system.git main

# Token aus Speicher löschen
TOKEN=""
