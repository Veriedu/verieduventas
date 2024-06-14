import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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

# Variables independientes y dependiente para tráfico a leads
X_traffic_to_leads = data[['ubicacion', 'industria', 'tamano_empresa', 'cargo', 'fuente_trafico', 
                           'paginas_vistas', 'tiempo_sitio', 'interacciones', 'contenido_visto', 
                           'alineacion_producto', 'ofertas_promocionales', 'reputacion_testimonios']]
y_traffic_to_leads = data['conversion_a_lead']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_traffic, X_test_traffic, y_train_traffic, y_test_traffic = train_test_split(X_traffic_to_leads, y_traffic_to_leads, test_size=0.3, random_state=42)

# Entrenar el modelo de regresión logística para tráfico a leads
model_traffic_to_leads = LogisticRegression()
model_traffic_to_leads.fit(X_train_traffic, y_train_traffic)

# Hacer predicciones y evaluar el modelo para tráfico a leads
y_pred_traffic = model_traffic_to_leads.predict(X_test_traffic)
accuracy_traffic = accuracy_score(y_test_traffic, y_pred_traffic)
roc_auc_traffic = roc_auc_score(y_test_traffic, model_traffic_to_leads.predict_proba(X_test_traffic)[:, 1])

# Crear la interfaz con Streamlit
st.title("Predicción de Conversión de Tráfico a Leads")
st.write(f"Accuracy: {accuracy_traffic}")
st.write(f"ROC AUC Score: {roc_auc_traffic}")

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
    nuevo_dato = pd.DataFrame({
        'ubicacion': [ubicacion],
        'industria': [industria],
        'tamano_empresa': [tamano_empresa],
        'cargo': [cargo],
        'fuente_trafico': [fuente_trafico],
        'paginas_vistas': [paginas_vistas],
        'tiempo_sitio': [tiempo_sitio],
        'interacciones': [interacciones],
        'contenido_visto': [contenido_visto],
        'alineacion_producto': [alineacion_producto],
        'ofertas_promocionales': [ofertas_promocionales],
        'reputacion_testimonios': [reputacion_testimonios]
    })
    
    probabilidad_conversion = model_traffic_to_leads.predict_proba(nuevo_dato)[:, 1][0]
    st.write(f"Probabilidad de Conversión: {probabilidad_conversion:.2f}")

# Ejecuta este script con: streamlit run streamlit_app.py



