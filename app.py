import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Judul Aplikasi
st.set_page_config(page_title="Klasifikasi Buah UAS", layout="centered")
st.title("🍎 Klasifikasi Buah & Sayur")

# 1. Load Label (Pastikan nama file JSON sama persis dengan di GitHub)
@st.cache_data
def load_labels():
    with open('klasifikasi class name.json.json', 'r') as f:
        return json.load(f)

# 2. Load Model (Pastikan nama file .h5 sama persis dengan di GitHub)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')

# Eksekusi Load
try:
    labels = load_labels()
    model = load_model()
    st.success("Model dan Label berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat file: {e}")

# 3. Antarmuka Unggah Gambar
uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Preprocessing
    # Catatan: MobileNetV2 biasanya menggunakan input 224x224
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button('Prediksi'):
        predictions = model.predict(img_array)
        class_id = str(np.argmax(predictions[0]))
        confidence = np.max(predictions[0]) * 100
        
        # Ambil nama dari JSON
        nama_buah = labels.get(class_id, "Tidak Diketahui")
        
        st.subheader(f"Hasil: {nama_buah}")
        st.write(f"Tingkat Keyakinan: {confidence:.2f}%")
