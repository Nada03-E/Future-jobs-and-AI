import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from src import config

def load_all_metrics(metrics_dir):
    """
    Carica tutte le metriche dei modelli salvati come .pkl in una cartella.
    Restituisce un DataFrame con le metriche aggregate.
    """
    records = []

    for filename in os.listdir(metrics_dir):
        if filename.endswith("_metrics.pkl"):
            filepath = os.path.join(metrics_dir, filename)
            with open(filepath, "rb") as f:
                metrics = pickle.load(f)
                model_name = filename.replace("_metrics.pkl", "")
                metrics["model_name"] = model_name
                records.append(metrics)

    df = pd.DataFrame(records)
    return df

def plot_model_comparison(df, sort_by="r2_val"):
    """
    Visualizza un grafico a barre dei modelli ordinati per una metrica (default: R² validation).
    """
    df_sorted = df.sort_values(by=sort_by, ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_sorted["model_name"], df_sorted[sort_by], color="skyblue")
    plt.ylabel(sort_by.replace("_", " ").upper())
    plt.title(f"Confronto modelli - {sort_by.replace('_', ' ').upper()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔍 Caricamento metriche dei modelli...")
    df_metrics = load_all_metrics(config.MODELS_PATH)

    if df_metrics.empty:
        print("❌ Nessuna metrica trovata.")
    else:
        # Mostra tabella ordinata per R² validation
        print("\n📊 Tabella confronto modelli (ordinati per r2_val):\n")
        display_cols = ["model_name", "r2_val", "rmse_val", "mae_val", "mape_val"]
        print(df_metrics[display_cols].sort_values(by="r2_val", ascending=False))

        # Grafico comparativo
        plot_model_comparison(df_metrics, sort_by="r2_val")
