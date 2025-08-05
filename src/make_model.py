import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
import pickle

sys.path.append(os.path.abspath('..'))
from src import config
import logging
from src import preprocess 


"""
def load_data():
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  
"""
"""
def load_data_training():
    conn = sqlite3.connect(config.DATABASE_PATH_training)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  

def load_data_validation():
    conn = sqlite3.connect(config.DATABASE_PATH_validation)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  

def load_data_MS():
    conn = sqlite3.connect(config.DATABASE_PATH_MS)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  

"""

def carica_dati_training_validation_MS(target_col: str):
    """
    Legge i file CSV salvati dopo lo split e restituisce:
    X_train, X_val, y_train, y_val.

    Si aspetta che i file `dati_training.csv` e `dati_validation.csv`
    si trovino nel percorso configurato in `config.RAW_DATA_PATH`.
    """
    train_path = os.path.join(config.RAW_DATA_PATH, "training_MS.csv")
    val_path = os.path.join(config.RAW_DATA_PATH, "validation_MS.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    return X_train, X_val, y_train, y_val

def carica_dati_training_validation(target_col: str):
    """
    Legge i file CSV salvati dopo lo split e restituisce:
    X_train, X_val, y_train, y_val.

    Si aspetta che i file `dati_training.csv` e `dati_validation.csv`
    si trovino nel percorso configurato in `config.RAW_DATA_PATH`.
    """
    train_path = os.path.join(config.RAW_DATA_PATH, "dati_training.csv")
    val_path = os.path.join(config.RAW_DATA_PATH, "dati_validation.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    return X_train, X_val, y_train, y_val

"""# --- Esempio d'uso ---
TARGET_COL = "Median Salary (USD)"  # Cambia con il nome reale della tua variabile target

X_train, X_val, y_train, y_val = carica_dati_training_validation(TARGET_COL)
"""


"""

def train_model():
    logging.info("Loading data for training...")
    df = load_data()

    # Rinomina le colonne per coerenza con l'interfaccia utente
    df = df.rename(columns={
        "X5 latitude": "latitude",
        "X6 longitude": "longitude",
        "Y house price of unit area": "price"
    })

    X = df[["latitude", "longitude"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=2)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model evaluation - MSE: {mse}, R2: {r2}")

    # Salva il modello e la scalatura
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    with open(os.path.join(config.MODELS_PATH, "knn_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(config.MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Salvare i predittori
    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred

    conn = sqlite3.connect(config.DATABASE_PATH)
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    logging.info("KNN model training and saving completed.")
"""


def train_model_neural_network():
    """
    Funzione di training per la rete neurale.

    1. Carica training e validation set tramite `carica_dati_training_validation`.
    2. Applica i preprocessamenti definiti in `src.preprocess`:
       - encode_and_scale_features per le feature.
       - scale_target per il target.
    3. Allena un MLPRegressor sui dati preprocessati.
    4. Calcola MAE su train e validation per verificare l'overfitting.
    5. Salva il modello completo nella cartella `config.MODELS_PATH`.
    6. Ritorna i valori di MAE su train e validation.
    """
    logging.info("Caricamento dei dati di training e validation…")
    target_col = "Median Salary (USD)"  # Sostituisci con il nome corretto della tua variabile target
    X_train, X_val, y_train, y_val = carica_dati_training_validation_MS(target_col)

    # Applica encoding e scaling delle feature
    # Questo utilizza le funzioni definite in `src.preprocess` per riutilizzare
    # lo stesso transformer sul validation.
    X_train_processed, X_val_processed, feature_transformer = preprocess.encode_and_scale_features(X_train, X_val)

    # Scaling del target (MinMax)
    y_train_processed, y_val_processed, target_scaler = preprocess.scale_target(y_train, y_val)

    # Definisci e allena il modello di rete neurale
    """model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        alpha=0.001,  # penalità L2
        activation='relu',
        random_state=42,
        max_iter=500
    ) MAE=0.2466"""
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        alpha=0.001,
        activation='relu',
        random_state=42,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1
    ) # MAE=0.0066


    model.fit(X_train_processed, y_train_processed)

    # Predizioni su train e validation per controllare l'overfitting
    y_pred_train = model.predict(X_train_processed)
    y_pred_val = model.predict(X_val_processed)

    mae_train = mean_absolute_error(y_train_processed, y_pred_train)
    mae_val = mean_absolute_error(y_val_processed, y_pred_val)

    logging.info(f"MAE train: {mae_train:.4f}")
    logging.info(f"MAE validation: {mae_val:.4f}")
    logging.info(f"Differenza MAE (overfitting check): {mae_val - mae_train:.4f}")

    # Prepara un oggetto da salvare che include modello e trasformazioni
    model_bundle = {
        'model': model,
        'feature_transformer': feature_transformer,
        'target_scaler': target_scaler,
        'target_col': target_col
    }

    # Salva il modello in config.MODELS_PATH
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    model_path = os.path.join(config.MODELS_PATH, "neural_network_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logging.info(f"Modello salvato in {model_path}")

    # Stampa il valore MAE per confronto tra modelli
    print(f"MAE (validation): {mae_val:.4f}")

    return mae_train, mae_val


if __name__ == "__main__":
    # Esegui il training se lanciato direttamente
    mae_train, mae_val = train_model_neural_network()
    print(f"MAE train: {mae_train:.4f}, MAE val: {mae_val:.4f}")