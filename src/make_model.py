import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import pickle

sys.path.append(os.path.abspath('..'))
from src import config
import logging
from src import preprocess 

def load_data():
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  



def carica_dati_training_validation(target_col: str):
    """
    Legge i CSV salvati dopo lo split e restituisce:
    X_train, X_val, y_train, y_val
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

# --- Esempio d'uso ---
TARGET_COL = "nome_colonna_target"  # Cambia con il nome reale della tua variabile target

X_train, X_val, y_train, y_val = carica_dati_training_validation(TARGET_COL)





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
