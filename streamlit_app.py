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

def obtener_color(valor):
    """Devuelve el color según el rango de valor"""
    if valor < 2:
        return "red"
    elif valor < 4:
        return "orange"
    elif valor < 6:
        return "yellow"
    elif valor < 8:
        return "lightgreen"
    else:
        return "green"

if st.button("Predecir calidad"):
    datos = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                       density, pH, sulphates, alcohol]])
    prediccion = modelo.predict(datos)
    valor = prediccion[0]

    # Color según rango
    color_rango = obtener_color(valor)

    # Mostrar valor con fondo del color correspondiente
    st.markdown(
        f"<h3 style='text-align:center; color:white; background-color:{color_rango}; "
        f"padding:10px; border-radius:8px;'>Calidad estimada: {valor:.2f}</h3>",
        unsafe_allow_html=True
    )

    # Crear gauge chart con barra más amigable
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        number={'font': {'color': color_rango, 'size': 48}},  # Número con color del rango
        title={'text': "Calidad del Vino"},
        gauge={
            'axis': {'range': [0, 10]},
            'steps': [
                {'range': [0, 2], 'color': "red"},
                {'range': [2, 4], 'color': "orange"},
                {'range': [4, 6], 'color': "yellow"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"},
            ],
            'bar': {'color': "#00BFFF"}  # Turquesa amigable
        }
    ))

    st.plotly_chart(fig)


