import sqlite3
import pandas as pd
import sys
import os
import importlib
from src import config

import logging
# Set up logging



def load_data_prova():

    logging.info('Apertuira file cvs...')
    path = os.path.join(config.RAW_DATA_PATH, "ai_job_trends_dataset.csv")

    print("PATH CSV:", os.path.join(config.RAW_DATA_PATH, "ai_job_trends_dataset.csv"))

    df = pd.read_csv(path, sep=',')
   
    print(f"Righe iniziali: {df.shape[0]}")
    df = df.dropna()
    print(f"Righe dopo rimozione NaN: {df.shape[0]}")

    df = df.drop_duplicates()
 
    df.reset_index(drop=True, inplace=True)
    

    # Create a connection to the SQLite database (or create if it doesn't exist)
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Write the DataFrame to a table (replace 'my_table' with your desired table name)
    df.to_sql(config.RAW_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    logging.info(f"Data successfully written to {config.RAW_TABLE} table.")