import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FinancialTimeSeriesModel:
    def __init__(self, index_name, model_dir="../data/models"):
        """
        Initialisiert das Zeitreihenmodell

        Args:
            index_name: Name des Index, für den das Modell trainiert wird
            model_dir: Verzeichnis zum Speichern der trainierten Modelle
        """
        self.index_name = index_name
        self.model_dir = os.path.join(model_dir, index_name)
        self.sequence_length = 24  # 24 Stunden Lookback
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.target_column = 'close'  # Standardmäßig schließen wir den Preis

        # Erstelle Verzeichnis, falls nicht vorhanden
        os.makedirs(self.model_dir, exist_ok=True)

    def _create_sequences(self, data, target_horizon=1):
        """
        Erstellt Sequenzen für das LSTM-Training

        Args:
            data: DataFrame mit Features und Targets
            target_horizon: Prognosehorizont in Stunden (1, 4, 8, oder 24)

        Returns:
            X: Sequenzen der Features, y: Zielwerte
        """
        # Bestimme den Target-Namen basierend auf dem Horizont
        target_col = f'target_{target_horizon}h'

        if target_col not in data.columns:
            raise ValueError(f"Ziel-Spalte {target_col} nicht in Daten gefunden")

        self.target_column = target_col

        # Entferne Zeilen mit fehlenden Zielwerten
        data = data.dropna(subset=[target_col])

        # Bestimme Features (alle Spalten außer Targets)
        feature_cols
