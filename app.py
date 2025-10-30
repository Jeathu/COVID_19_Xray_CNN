import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="COVID-19 Detector", page_icon="🫁", layout="wide")

# CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Charge le modèle."""
    try:
        # Essaie format .h5
        model = tf.keras.models.load_model('models/covid_model.h5')
        return model, True
    except:
        try:
            # Essaie format SavedModel
            model = tf.keras.models.load_model('models/covid_model')
            return model, True
        except:
            return None, False


def preprocess_image(image):
    """Prétraite l'image pour le modèle (ajuste selon ton modèle)."""
    # Redimensionne selon la taille d'entrée de ton modèle
    img = image.resize((128, 128))  # Changé à 128x128
    img = img.convert('RGB')  # Convertit en RGB
    img_array = np.array(img) / 255.0  # Normalise
    img_array = np.expand_dims(img_array, axis=0)  # Ajoute batch dimension
    return img_array


# Header
st.markdown("""
<div class="main-header">
    <h1>🫁 COVID-19 X-Ray Detector</h1>
    <p>Analyse de radiographies pulmonaires par IA</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Informations")
    st.info("**Utilisation:**\n1. Upload X-Ray\n2. Analyse automatique\n3. Résultats instantanés")
    st.warning("⚠️ 🔬 À titre informatif uniquement - Les résultats ne remplacent pas un avis médical. Projet de recherche académique.")
    st.markdown("---")
    st.markdown("**Modèle:** CNN TensorFlow keras\n**Auteur:** JEATHUSAN KUGATHAS\n**Licence:** MIT")

# Charge le modèle
model, model_loaded = load_model()

if not model_loaded:
    st.error("❌ Modèle non trouvé ! Assure-toi que `covid_model.h5` existe.")
    st.info("💡 Dans ton Jupyter : `model.save('covid_model.h5')`")
    st.stop()

# Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload X-Ray")
    uploaded_file = st.file_uploader("Choisir une radiographie", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Radiographie", use_container_width=True)

with col2:
    st.subheader("🔬 Résultats")
    
    if uploaded_file:
        with st.spinner('Analyse en cours...'):
            # Prédiction
            img_array = preprocess_image(image)
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            covid_prob = float(prediction * 100)
            normal_prob = float((1 - prediction) * 100)
            has_covid = bool(prediction > 0.5)
            confidence = float(max(prediction, 1 - prediction) * 100)
        
        # Résultat
        if has_covid:
            st.error("### 🦠 COVID-19 DÉTECTÉ")
        else:
            st.success("### ✅ PAS DE COVID-19")
        
        st.markdown("---")
        
        # Métriques
        m1, m2 = st.columns(2)
        m1.metric("Confiance", f"{confidence:.1f}%")
        m2.metric("COVID-19", f"{covid_prob:.1f}%")
        
        st.markdown("---")
        st.subheader("📊 Probabilités")
        
        st.write("**COVID-19:**")
        st.progress(covid_prob / 100)
        st.write(f"{covid_prob:.2f}%")
        
        st.write("**Normal:**")
        st.progress(normal_prob / 100)
        st.write(f"{normal_prob:.2f}%")
    else:
        st.info("👆 Uploadez une radiographie")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>Auteur : JEATHUSAN KUGATHAS - Étudiant en Master Informatique et Big Data</div>", unsafe_allow_html=True)