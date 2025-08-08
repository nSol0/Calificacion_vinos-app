# app.py
import streamlit as st
import joblib
import numpy as np
import sklearn  # asegúrate de importar la librería que usa el modelo

# === Cargar el modelo guardado ===
modelo = joblib.load("mejor_modelo.pkl")  # Cambia al nombre real de tu archivo

st.title("🍷 Predicción de Calidad de Vino Rojo")
st.write("Introduce los valores del vino y obtendrás la predicción de su calidad final.")

# === Entradas de usuario ===
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Botón para predecir
if st.button("Predecir Calidad"):
    # Crear vector de entrada
    entrada = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                         residual_sugar, chlorides, free_sulfur_dioxide,
                         total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    
    prediccion = modelo.predict(entrada)
    st.success(f"La calidad estimada del vino es: {prediccion[0]}")
