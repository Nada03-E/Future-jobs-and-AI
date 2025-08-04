import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for raw data (Excel files)
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/")

# SQLite Database Path
DATABASE_PATH = os.path.join(BASE_DIR, "database/ai_jobs.db")

# Preprocessed Data Table Name
PROCESSED_TABLE = "Dati_Processati"

# Raw Data Table Name
RAW_TABLE = "Dati_Non_preprocesati"

# Predictions Table Name
PREDICTIONS_TABLE = "predictions"

# model evaluation
EVALUATION_TABLE = "grid_search_results"

# Logging Configuration
LOGGING_LEVEL = "INFO"

#save the model
MODELS_PATH = os.path.join(BASE_DIR, "models/")
