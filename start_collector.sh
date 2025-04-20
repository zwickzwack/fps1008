#!/bin/bash

# Installiere benötigte Pakete, falls noch nicht vorhanden
pip3 install yfinance pandas schedule

# Erstelle das Datenverzeichnis, falls nicht vorhanden
mkdir -p data/raw

# Starte die automatische Datensammlung im Hintergrund
cd "$(dirname "$0")"
python3 src/data_collection/auto_collect.py &

# Schreibe die Prozess-ID in eine Datei für späteres Beenden
echo $! > collector.pid

echo "Automatische Datensammlung gestartet (Prozess-ID: $(cat collector.pid))"
echo "Zum Beenden führen Sie './stop_collector.sh' aus"
