"""
Questo modulo definisce una funzione di training per una rete neurale
utilizzando i dataset di training e validation già suddivisi. Viene
utilizzato il preprocess importato dal pacchetto `src.preprocess` per
eseguire la codifica e lo scaling delle feature e del target. Il
modello finale viene salvato nella cartella specificata da
`config.MODELS_PATH` e stampa a video il Mean Absolute Error (MAE) per
facilitare il confronto tra modelli.
"""

import os
import pickle
import logging

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

from src import config
from src import preprocess


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
    X_train, X_val, y_train, y_val = carica_dati_training_validation(target_col)

    # Applica encoding e scaling delle feature
    # Questo utilizza le funzioni definite in `src.preprocess` per riutilizzare
    # lo stesso transformer sul validation.
    X_train_processed, X_val_processed, feature_transformer = preprocess.encode_and_scale_features(X_train, X_val)

    # Scaling del target (MinMax)
    y_train_processed, y_val_processed, target_scaler = preprocess.scale_target(y_train, y_val)

    # Definisci e allena il modello di rete neurale
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        random_state=42,
        max_iter=300
    )

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