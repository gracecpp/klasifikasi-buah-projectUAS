import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Pengaturan Halaman
st.set_page_config(page_title="Klasifikasi Buah & Sayur", layout="centered")

# 1. Load Data Label dari JSON
@st.cache_data
def load_labels():
    with open('klasifikasi class name.json.json', 'r') as f:
        return json.load(f)

# 2. Load Model AI (Ganti 'model_anda.h5' dengan nama file model Anda)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_anda.h5')

# Eksekusi Load
labels = load_labels()
try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("File model (.h5) tidak ditemukan! Pastikan file model ada di folder yang sama.")

# 3. Antarmuka Pengguna (UI)
st.title("🍎 Klasifikasi Buah & Sayur")
st.write("Sistem Identifikasi Gambar menggunakan Deep Learning")

uploaded_file = st.file_uploader("Unggah foto buah/sayur...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Pre-processing Gambar
    # Catatan: Sesuaikan (224, 224) dengan input_shape model Anda saat training
    img_resized = image.resize((224, 224)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    if st.button('Mulai Klasifikasi'):
        predictions = model.predict(img_array)
        class_idx = str(np.argmax(predictions[0])) # Ambil index tertinggi sebagai string
        confidence = np.max(predictions[0]) * 100
        
        # Ambil nama dari JSON berdasarkan key index
        nama_produk = labels.get(class_idx, "Label tidak ditemukan")

        st.success(f"Hasil: **{nama_produk}**")
        st.info(f"Tingkat Keyakinan: {confidence:.2f}%")