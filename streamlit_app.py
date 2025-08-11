import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

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
    calidad = float(prediccion[0])

    st.success(f"Calidad estimada: {calidad:.2f}")

    # --- Gauge chart ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=calidad,
        title={'text': "Calidad del Vino", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 2], 'color': "red"},
                {'range': [2, 4], 'color': "orange"},
                {'range': [4, 6], 'color': "yellow"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(fig)

