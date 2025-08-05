import os
import sys
import pandas as pd
import sqlite3
import numpy as np
sys.path.append(os.path.abspath('..'))
from src import config
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
"""
def preprocess_data():
    # Connessione al database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Carica i dati grezzi
    df = pd.read_sql_query(f"SELECT * FROM {config.RAW_TABLE}", conn)

    # Filtra solo le colonne necessarie
    df = df[['X5 latitude', 'X6 longitude', 'Y house price of unit area']]

    # Salva i dati pre-processati nel database
    df.to_sql(config.PROCESSED_TABLE, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    print(f"Dati preprocessati salvati nella tabella {config.PROCESSED_TABLE}.")
"""



def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Divide il DataFrame in X (features) e y (target).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def encode_and_scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    Esegue il one-hot encoding sulle colonne categoriche e lo scaling
    (standardizzazione) sulle colonne numeriche. Ritorna i dati trasformati
    e l'oggetto transformer per riutilizzare la stessa trasformazione anche su test o nuovi dati.
    """
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    transformer = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('numeric', StandardScaler(), num_cols)
        ],
        remainder='drop'  # scarta eventuali colonne non specificate
    )
    
    X_train_processed = transformer.fit_transform(X_train)
    X_val_processed = transformer.transform(X_val)
    
    return X_train_processed, X_val_processed, transformer

def scale_target(y_train: pd.Series, y_val: pd.Series):
    """
    Esegue lo scaling (Min-Max) sulla variabile target. Questo può essere utile
    per problemi di regressione dove i valori target hanno scale molto diverse.
    Restituisce i target scalati e lo scaler da riutilizzare per invertire la scala.
    """
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    y_val_scaled = scaler.transform(y_val.to_numpy().reshape(-1, 1)).ravel()
    return y_train_scaled, y_val_scaled, scaler

def prepare_data_for_nn(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    """
    Esegue lo split train/val, poi chiama la codifica e lo scaling delle features,
    e (opzionale) lo scaling del target. Restituisce X_train_processed, X_val_processed,
    y_train_processed, y_val_processed, il trasformer delle feature e lo scaler del target.
    """
    # Divisione X e y
    X, y = split_features_target(df, target_col)
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                      random_state=random_state)
    
    # Encoding e scaling delle feature
    X_train_processed, X_val_processed, feature_transformer = encode_and_scale_features(X_train, X_val)
    
    # Scaling del target
    y_train_processed, y_val_processed, target_scaler = scale_target(y_train, y_val)
    
    return (X_train_processed, X_val_processed,
            y_train_processed, y_val_processed,
            feature_transformer, target_scaler)




