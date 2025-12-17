import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Buah UAS", layout="centered")
st.title("🍎 Klasifikasi Buah & Sayur")

# --- 1. LOAD MODEL & LABEL ---
@st.cache_resource
def load_resources():
    # Load Model .h5
    model = tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')
    
    # Mencoba load label (menangani error nama file .json atau .json.json)
    try:
        with open('klasifikasi class name.json', 'r') as f:
            labels = json.load(f)
    except FileNotFoundError:
        with open('klasifikasi class name.json.json', 'r') as f:
            labels = json.load(f)
            
    return model, labels

try:
    model, labels = load_resources()
    st.success("✅ Model dan Label Siap!")
except Exception as e:
    st.error(f"❌ Gagal memuat file: {e}")
    st.stop()

# --- 2. ANTARMUKA UNGGAH ---
uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    if st.button('Mulai Prediksi'):
        with st.spinner('Sedang menganalisis...'):
          
            # Mengubah ukuran menjadi 100x100 agar sesuai dengan input model Fruits-360
            img_resized = image.resize((100, 100)) 
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            
            # Menambah dimensi batch sehingga menjadi (1, 100, 100, 3)
            img_array = np.expand_dims(img_array, axis=0) 
            
            # Normalisasi pixel (0-1)
            img_array = img_array / 255.0 

            try:
                # Menjalankan prediksi (Baris ini tidak akan error lagi)
                predictions = model.predict(img_array)
                class_index = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # Mengambil label berdasarkan index (diubah ke string sesuai format JSON)
                nama_hasil = labels.get(str(class_index), "Tidak Diketahui")
                
                # Menampilkan Hasil
                st.subheader(f"Hasil Prediksi: {nama_hasil}")
                st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")


