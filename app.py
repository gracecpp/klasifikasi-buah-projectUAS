import streamlit as st
import numpy as np
import json
from PIL import Image

# Pengaturan Judul Web
st.set_page_config(page_title="UAS Klasifikasi Buah", page_icon="🍎")
st.title("🍎 Aplikasi Klasifikasi Buah & Sayur")
st.write("Project UAS - Menggunakan Model MobileNetV2")

# 1. Fungsi Load Model & Label
@st.cache_resource
def load_resource():
    # Nama file .h5 harus sama persis dengan yang ada di GitHub Anda
    model = tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')
    
    # Nama file JSON harus sama persis dengan yang ada di GitHub Anda
    with open('klasifikasi class name.json', 'r') as f:
        labels = json.load(f)
    return model, labels

# Menjalankan fungsi load
try:
    model, labels = load_resource()
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

# 2. UI Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if st.button('Prediksi'):
        with st.spinner('Sedang menganalisis...'):
            # Preprocessing (MobileNetV2 menggunakan 224x224)
            img = image.convert('RGB')
            img = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Proses Prediksi
            predictions = model.predict(img_array)
            class_id = str(np.argmax(predictions[0]))
            confidence = np.max(predictions[0]) * 100
            
            # Ambil nama dari JSON
            nama_buah = labels.get(class_id, "Tidak Diketahui")
            
            st.success(f"### Hasil: {nama_buah}")
            st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

