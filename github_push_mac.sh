#!/bin/bash

# Konfiguration
USERNAME="zwickzwack"
CURRENT_DATE="2025-04-19 17:13:37"
REPO_URL="https://github.com/zwickzwack/financial-prediction-system.git"
DASHBOARD_PATH="src/dashboard/news_based_predictor.py"

echo "=== GitHub Push für macOS ==="
echo "Benutzername: $USERNAME"
echo "Datum: $CURRENT_DATE"
echo "Repository: $REPO_URL"
echo "============================"

# Aktualisiere den Zeitstempel im Dashboard (macOS-Version)
if [ -f "$DASHBOARD_PATH" ]; then
    sed -i '' "s/current_time = datetime.strptime(\".*\", \"%Y-%m-%d %H:%M:%S\")/current_time = datetime.strptime(\"$CURRENT_DATE\", \"%Y-%m-%d %H:%M:%S\")/" $DASHBOARD_PATH
    echo "✅ Zeitstempel aktualisiert"
else
    echo "⚠️ Dashboard-Datei nicht gefunden: $DASHBOARD_PATH"
fi

# Git-Konfiguration prüfen
if ! command -v git &> /dev/null; then
    echo "❌ Git ist nicht installiert. Bitte installieren Sie Git mit:"
    echo "   brew install git"
    exit 1
fi

# Prüfe, ob es ein Git-Repository ist
if [ ! -d .git ]; then
    echo "Initialisiere Git-Repository..."
    git init
    echo "✅ Git-Repository initialisiert"
fi

# Git-Konfiguration
git config user.name "$USERNAME"
git config user.email "$USERNAME@users.noreply.github.com"
echo "✅ Git-Benutzer konfiguriert"

# Prüfe, ob alle wichtigen Verzeichnisse existieren
mkdir -p data/news
mkdir -p data/market
mkdir -p models
echo "✅ Verzeichnisstruktur überprüft"

# Erstelle .gitignore, falls nicht vorhanden
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'GITIGNOREEOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# macOS spezifisch
.DS_Store
.AppleDouble
.LSOverride
._*

# Jupyter Notebook
.ipynb_checkpoints

# Streamlit
.streamlit/

# VS Code
.vscode/

# PyCharm
.idea/
GITIGNOREEOF
    echo "✅ .gitignore erstellt"
fi

# Alle Dateien zum Commit hinzufügen
git add .
git commit -m "Dashboard deployment: $CURRENT_DATE"
echo "✅ Änderungen committed"

# Remote konfigurieren
if git remote | grep -q "origin"; then
    git remote set-url origin "$REPO_URL"
    echo "✅ Remote-URL aktualisiert"
else
    git remote add origin "$REPO_URL"
    echo "✅ Remote-URL hinzugefügt"
fi

# Branch erstellen
git branch -M main
echo "✅ Branch 'main' konfiguriert"

# GitHub-Authentifizierung einrichten
echo "Konfiguriere GitHub-Authentifizierung..."
git config --global credential.helper osxkeychain
echo "✅ Keychain als Credential Helper konfiguriert"

echo ""
echo "🔐 GitHub-Authentifizierung:"
echo "1. Bei der Passwortabfrage geben Sie bitte Ihr GitHub Personal Access Token ein (nicht Ihr Passwort)."
echo "2. Falls Sie noch kein Token haben, erstellen Sie eines unter: https://github.com/settings/tokens"
echo "3. Wählen Sie mindestens die 'repo'-Berechtigungen"
echo ""

# Push vorbereiten
echo "Bereit zum Push. Drücken Sie Enter zum Fortfahren..."
read

# Push durchführen
git push -u origin main
PUSH_RESULT=$?

if [ $PUSH_RESULT -eq 0 ]; then
    echo "✅ Push erfolgreich! Ihr Code ist jetzt auf GitHub: $REPO_URL"
else
    echo "❌ Push fehlgeschlagen. Bitte überprüfen Sie die Fehlermeldungen."
    echo "Stellen Sie sicher, dass:"
    echo "1. Ihr GitHub-Repository existiert: $REPO_URL"
    echo "2. Sie ein gültiges Personal Access Token verwenden"
    echo "3. Das Repository leer ist (ohne README, LICENSE, etc.)"
fi
