#!/bin/bash

# Konfigurationsvariablen
GITHUB_USERNAME="zwickzwack"
CURRENT_DATE="2025-04-19 16:52:26"
APP_NAME="financial-prediction-system"
DASHBOARD_PATH="src/dashboard/news_based_predictor.py"

echo "=== Financial News Market Predictor Deployment ==="
echo "Benutzername: zwickzwack"
echo "Datum: $CURRENT_DATE"
echo "GitHub: $GITHUB_USERNAME"
echo "==============================================="

# Sicherstellen, dass Verzeichnisse existieren
mkdir -p data/news
mkdir -p data/market
mkdir -p models

# Abhängigkeiten installieren
echo "Installiere erforderliche Abhängigkeiten..."
pip install -q streamlit pandas numpy plotly scikit-learn yfinance nltk requests

# NLTK-Daten herunterladen
python -m nltk.downloader -q vader_lexicon

# Aktuelles Datum in das Dashboard einfügen
echo "Aktualisiere Zeitstempel im Dashboard..."
sed -i "s/current_time = datetime.strptime(\".*\", \"%Y-%m-%d %H:%M:%S\")/current_time = datetime.strptime(\"$CURRENT_DATE\", \"%Y-%m-%d %H:%M:%S\")/" $DASHBOARD_PATH

# Falls das Repository noch nicht initialisiert wurde, initialisieren
if [ ! -d .git ]; then
    echo "Initialisiere Git-Repository..."
    git init
    git config user.name "$GITHUB_USERNAME"
    git config user.email "$GITHUB_USERNAME@users.noreply.github.com"
    
    # Erstellen einer .gitignore-Datei
    cat > .gitignore << 'GITIGNOREEOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
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

# Jupyter Notebook
.ipynb_checkpoints

# Streamlit
.streamlit/

# VS Code
.vscode/

# Data files (optional - entfernen, wenn Daten im Repository sein sollen)
# data/

# Environment
.env
.venv
venv/
ENV/
GITIGNOREEOF

    echo "Repository initialisiert."
fi

# Vorbereiten für GitHub-Push
echo "Bereite Repository für GitHub vor..."

# Commit aller Änderungen
git add .
git commit -m "Update dashboard for deployment - $CURRENT_DATE"

# Remote hinzufügen oder aktualisieren
if git remote | grep -q "origin"; then
    git remote set-url origin https://github.com/$GITHUB_USERNAME/$APP_NAME.git
else
    git remote add origin https://github.com/$GITHUB_USERNAME/$APP_NAME.git
fi

echo ""
echo "======================= HINWEISE ============================="
echo "1. Bitte stellen Sie sicher, dass Sie ein Repository namens"
echo "   '$APP_NAME' in Ihrem GitHub-Account haben."
echo ""
echo "2. Führen Sie den folgenden Befehl aus, um Ihre Änderungen zu pushen:"
echo "   git push -u origin main"
echo ""
echo "3. Nach dem Push können Sie Ihr Dashboard auf Streamlit Cloud deployen:"
echo "   - Gehen Sie zu https://streamlit.io/cloud"
echo "   - Melden Sie sich an und klicken Sie auf 'New app'"
echo "   - Wählen Sie Ihr Repository, Branch 'main' und den Pfad:"
echo "     $DASHBOARD_PATH"
echo ""
echo "4. Alternativ können Sie das Dashboard lokal starten mit:"
echo "   streamlit run $DASHBOARD_PATH"
echo "=============================================================="

# Starten des Dashboards
echo "Möchten Sie das Dashboard jetzt lokal starten? (j/n)"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    streamlit run $DASHBOARD_PATH
fi
