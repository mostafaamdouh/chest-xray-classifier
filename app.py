import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

@st.cache_resource
def load_model():
    data = joblib.load('xray_model.joblib')
    model = tf.keras.models.model_from_json(data['config'])
    model.set_weights(data['weights'])
    return model

model = load_model()
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

st.title("🫁 Chest X-Ray Classifier")
st.write("Upload a chest X-ray image to classify it as Normal, Pneumonia, or COVID-19")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray', use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
    
    st.subheader("Result:")
    if predicted_class == 'Normal':
        st.success(f"✅ {predicted_class} ({confidence:.1f}%)")
    elif predicted_class == 'Pneumonia':
        st.warning(f"⚠️ {predicted_class} ({confidence:.1f}%)")
    else:
        st.error(f"🔴 {predicted_class} ({confidence:.1f}%)")

    st.subheader("All Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(predictions[0][i]), text=f"{cls}: {predictions[0][i]*100:.1f}%")
