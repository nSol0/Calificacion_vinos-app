import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# === Función para asignar colores ===
def obtener_color(valor):
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

# === Interfaz de selección de vino ===
st.title("🍷 Predicción de Calidad del Vino")
opcion_vino = st.radio("Selecciona el tipo de vino:", ("Vino Tinto", "Vino Blanco"))

# Definir error medio según el tipo
if opcion_vino == "Vino Tinto":
    modelo = joblib.load("mejor_modelo.pkl")
    error_medio = 0.45
else:
    modelo = joblib.load("mejor_modelo_white.pkl")
    error_medio = 0.35

# === Entradas del usuario ===
fixed_acidity = st.number_input("Acidez fija", value=7.0)
volatile_acidity = st.number_input("Acidez volátil", value=0.27)
citric_acid = st.number_input("Ácido cítrico", value=0.36)
residual_sugar = st.number_input("Azúcar residual", value=20.7)
chlorides = st.number_input("Cloruros", value=0.045)
free_sulfur_dioxide = st.number_input("SO₂ libre", value=45.0)
total_sulfur_dioxide = st.number_input("SO₂ total", value=170.0)
density = st.number_input("Densidad", value=1.0010)
pH = st.number_input("pH", value=3.0)
sulphates = st.number_input("Sulfatos", value=0.45)
alcohol = st.number_input("Alcohol", value=8.8)

# === Botón de predicción ===
if st.button("Predecir calidad"):
    datos = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                       density, pH, sulphates, alcohol]])
    
    prediccion = modelo.predict(datos)
    valor = float(prediccion[0])

    # Calcular rango con error medio
    valor_min = max(0, valor - error_medio)
    valor_max = min(10, valor + error_medio)

    color_rango = obtener_color(valor)

    # Mostrar texto con color
    st.markdown(
        f"<h3 style='text-align:center; color:white; background-color:{color_rango}; "
        f"padding:10px; border-radius:8px;'>Calidad estimada: {valor:.2f} "
        f"(rango aprox: {valor_min:.2f} - {valor_max:.2f})</h3>",
        unsafe_allow_html=True
    )

    # === Gauge básico con barra amigable ===
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        number={'font': {'color': color_rango, 'size': 48}},
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
            'bar': {'color': "#4B9CD3"}  # Color amigable azul suave
        }
    ))

    st.plotly_chart(fig)
