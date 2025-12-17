import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Klasifikasi Buah UAS",
    page_icon="🍎",
    layout="centered"
)

# --- 2. FUNGSI LOAD MODEL & LABEL (DENGAN CACHE) ---
@st.cache_resource
def load_all_resources():
    # Load Model .h5
    # Pastikan nama file ini sama persis dengan yang ada di GitHub Anda
    model = tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')
    
    # Load Label JSON
    with open('klasifikasi class name.json', 'r') as f:
        labels = json.load(f)
        
    return model, labels

# Menjalankan fungsi load dan menangani error jika file tidak ditemukan
try:
    model, labels = load_all_resources()
    status_load = True
except Exception as e:
    st.error(f"❌ Gagal memuat file! Pastikan file .h5 dan .json sudah di-upload. Error: {e}")
    status_load = False

# --- 3. ANTARMUKA PENGGUNA (UI) ---
st.title("🍎 Klasifikasi Buah & Sayur")
st.write("Aplikasi ini menggunakan Model MobileNetV2 untuk mengenali 131 jenis buah/sayur.")
st.divider()

# Komponen Unggah Gambar
uploaded_file = st.file_uploader("Pilih gambar dari perangkat Anda...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and status_load:
    # Menampilkan Gambar yang di-upload
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Tombol Prediksi
    if st.button('Mulai Klasifikasi'):
        with st.spinner('Sedang menganalisis gambar...'):
            # --- 4. PRE-PROCESSING GAMBAR ---
            # Menyesuaikan gambar dengan input MobileNetV2 (224x224)
            img_resized = image.resize((100, 100))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalisasi pixel

            # --- 5. PROSES PREDIKSI ---
            predictions = model.predict(img_array)
            
            # Mengambil index dengan probabilitas tertinggi
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Mengambil nama label dari JSON berdasarkan index (string)
            class_id_str = str(class_index)
            nama_hasil = labels.get(class_id_str, "Kategori Tidak Diketahui")

            # --- 6. TAMPILKAN HASIL ---
            st.success(f"### Hasil Prediksi: **{nama_hasil}**")
            st.write(f"Tingkat Keyakinan AI: **{confidence:.2f}%**")
            
            # Memberikan info tambahan jika akurasi rendah
            if confidence < 60:
                st.warning("⚠️ Hasil mungkin kurang akurat. Pastikan gambar jelas dan fokus pada objek.")

st.divider()
st.caption("UAS Project - Klasifikasi Buah 2024")

