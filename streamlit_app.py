import pandas as pd
import numpy as np
import streamlit as st

# Datos inventados para tráfico a leads
data = pd.DataFrame({
    'ubicacion': [1, 2, 1, 3, 2, 1, 2, 3, 1, 2],
    'industria': [1, 2, 3, 1, 2, 1, 3, 2, 1, 3],
    'tamano_empresa': [100, 200, 150, 300, 250, 100, 200, 150, 300, 250],
    'cargo': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
    'fuente_trafico': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
    'paginas_vistas': [5, 6, 5, 7, 6, 5, 6, 5, 7, 6],
    'tiempo_sitio': [300, 400, 350, 450, 400, 300, 400, 350, 450, 400],
    'interacciones': [2, 3, 2, 4, 3, 2, 3, 2, 4, 3],
    'contenido_visto': [1, 2, 1, 3, 2, 1, 2, 1, 3, 2],
    'alineacion_producto': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
    'ofertas_promocionales': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
    'reputacion_testimonios': [4, 5, 4, 5, 4, 4, 5, 4, 5, 4],
    'conversion_a_lead': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
})

# Implementar una simple regresión logística manualmente
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    m = len(y)
    h = predict_proba(X, theta)
    epsilon = 1e-5  # para evitar log(0)
    cost = (-1/m) * (y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return np.sum(cost)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = np.dot(X.T, (predict_proba(X, theta) - y)) / m
        theta -= learning_rate * gradient
    return theta

# Preparar los datos para la regresión logística
X = data[['ubicacion', 'industria', 'tamano_empresa', 'cargo', 'fuente_trafico', 'paginas_vistas',
          'tiempo_sitio', 'interacciones', 'contenido_visto', 'alineacion_producto', 
          'ofertas_promocionales', 'reputacion_testimonios']]
y = data['conversion_a_lead']
X = np.c_[np.ones((X.shape[0], 1)), X]  # agregar la columna de intercepto
y = y.values

# Inicializar los parámetros theta
theta = np.zeros(X.shape[1])

# Configurar los parámetros de entrenamiento
learning_rate = 0.01
iterations = 1000

# Entrenar el modelo
theta = gradient_descent(X, y, theta, learning_rate, iterations)

# Crear la interfaz con Streamlit
st.title("Predicción de Conversión de Tráfico a Leads")

# Formularios interactivos para ingresar nuevos datos
ubicacion = st.selectbox("Ubicación", [1, 2, 3])
industria = st.selectbox("Industria", [1, 2, 3])
tamano_empresa = st.number_input("Tamaño de la Empresa", min_value=1)
cargo = st.selectbox("Cargo", [1, 2])
fuente_trafico = st.selectbox("Fuente de Tráfico", [1, 2])
paginas_vistas = st.number_input("Páginas Vistas", min_value=1)
tiempo_sitio = st.number_input("Tiempo en el Sitio (segundos)", min_value=1)
interacciones = st.number_input("Interacciones", min_value=1)
contenido_visto = st.selectbox("Contenido Visto", [1, 2, 3])
alineacion_producto = st.selectbox("Alineación del Producto", [1, 2])
ofertas_promocionales = st.selectbox("Ofertas Promocionales", [1, 2])
reputacion_testimonios = st.number_input("Reputación y Testimonios", min_value=1, max_value=5)

# Predicción con nuevos datos
if st.button("Predecir Conversión"):
    nuevo_dato = np.array([1, ubicacion, industria, tamano_empresa, cargo, fuente_trafico, paginas_vistas,
                           tiempo_sitio, interacciones, contenido_visto, alineacion_producto, 
                           ofertas_promocionales, reputacion_testimonios])
    probabilidad_conversion = predict_proba(nuevo_dato, theta)
    st.write(f"Probabilidad de Conversión: {probabilidad_conversion:.2f}")

