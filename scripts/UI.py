import os
import sys
import pickle
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import folium

# Aggiungiamo la directory principale al path
sys.path.append(os.path.abspath(".."))
from src import config

# Carica il modello e lo scaler
model_path = os.path.join(config.MODELS_PATH, "knn_model.pkl")
scaler_path = os.path.join(config.MODELS_PATH, "scaler.pkl")

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Titolo dell'app
st.title(" Previsione Prezzi Immobiliari (KNN)")
st.markdown("""
Digita la latitudine e longitudine oppure utilizza la mappa interattiva.
""")

# finestre di inserimento manuale
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitudine", value=24.95, format="%.5f")
with col2:
    longitude = st.number_input("Longitudine", value=121.54, format="%.5f")

# Mappa interagibile
st.markdown("### Oppure clicca su una posizione nella mappa:")
map_center = [latitude, longitude]
m = folium.Map(location=map_center, zoom_start=12)
m.add_child(folium.LatLngPopup())
output = st_folium(m, width=700, height=400)

# Se l'utente clicca sulla mappa, aggiorna lat e lon
if output and output.get("last_clicked"):
    latitude = output["last_clicked"]["lat"]
    longitude = output["last_clicked"]["lng"]
    st.success(f"Coordinate selezionate: {latitude:.5f}, {longitude:.5f}")

# Bottone per la previsione
if st.button("Prevedi Prezzo"):
    input_df = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.markdown(f"### Prezzo stimato: **{prediction:.2f}** per unità di superficie")
