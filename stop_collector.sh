#!/bin/bash

if [ -f collector.pid ]; then
    PID=$(cat collector.pid)
    if ps -p $PID > /dev/null; then
        echo "Beende Datensammler (Prozess-ID: $PID)..."
        kill $PID
        rm collector.pid
        echo "Datensammler beendet."
    else
        echo "Datensammler läuft nicht mehr."
        rm collector.pid
    fi
else
    echo "Keine Datensammler-PID gefunden. Überprüfen Sie, ob er läuft, mit 'ps aux | grep auto_collect'."
fi
