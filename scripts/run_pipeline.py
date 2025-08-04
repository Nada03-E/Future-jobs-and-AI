import os
import sys
sys.path.append(os.path.abspath('..')) #Utile per poter tornare indietro di una cartella nel path in qui ci troviamo in modo da poter accedere ad un altra cartella. ( in questo caso è utilizzato per entrare nella cartella src in modo da poter accedere alle varie funzioni in modo più comodo e pratrico)

import logging
from src import config
from src.load_data import load_data

from src.make_model import train_model


#creare la pagina di log per vedere i passaggi.
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    logging.info("Inizio elaborazione Valutazione Prezzo case...")

    # Caricare il file excell
    logging.info("Caricare i dati iniziali...")
    load_data()

    # Creazione del modelo Migliore
    logging.info("Eseguire il preprocessing e Fare il training del modello...")
    train_model()


if __name__ == "__main__":
    main()

