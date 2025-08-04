import os
import sys
import pandas as pd
import sqlite3
import numpy as np
sys.path.append(os.path.abspath('..'))
from src import config
from sklearn.model_selection import train_test_split

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




