import streamlit as st
import joblib
import numpy as np

st.title("Predicción de calidad de vinos")

# --- Menú lateral ---
opcion_vino = st.sidebar.selectbox(
    "Selecciona el tipo de vino",
    ["Vino Tinto", "Vino Blanco"]
)

# Cargar el modelo según selección
if opcion_vino == "Vino Tinto":
    modelo = joblib.load("mejor_modelo.pkl")
else:
    modelo = joblib.load("mejor_modelo_white.pkl")

# --- Entrada de datos ---
st.subheader(f"Ingrese características del {opcion_vino.lower()}")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Botón para predecir
if st.button("Predecir calidad"):
    datos = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                       density, pH, sulphates, alcohol]])
    prediccion = modelo.predict(datos)
    st.success(f"Calidad estimada: {prediccion[0]:.2f}")
