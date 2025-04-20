#!/bin/bash

# Docker-Image bauen
docker build -t financial-dashboard:latest .

# Alten Container stoppen und entfernen (falls vorhanden)
docker stop financial-dashboard 2>/dev/null
docker rm financial-dashboard 2>/dev/null

# Neuen Container starten
docker run -d --name financial-dashboard -p 8501:8501 financial-dashboard:latest

echo "Dashboard wurde erfolgreich als Docker-Container gestartet"
echo "Sie können es unter http://localhost:8501 erreichen"
