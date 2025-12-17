import streamlit as st
import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Buah UAS", layout="centered")
st.title("🍎 Klasifikasi Buah & Sayur (Lite Version)")

# --- LOAD LABEL & MODEL ---
@st.cache_resource
def load_resources():
    # Load Label JSON
    with open('klasifikasi class name.json', 'r') as f:
        labels = json.load(f)
    
    # Load Model TFLite (Pastikan Anda sudah upload file .tflite)
    interpreter = tflite.Interpreter(model_path="model_buah.tflite")
    interpreter.allocate_tensors()
    return interpreter, labels

try:
    interpreter, labels = load_resources()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ Model Lite & Label Siap!")
except Exception as e:
    st.error(f"❌ Error: {e}. Pastikan file .tflite dan .json sudah benar.")
    st.stop()

# --- UI UPLOAD ---
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar Terpilih', use_container_width=True)
    
    if st.button('Mulai Prediksi'):
        # 1. Preprocessing (Tanpa TF)
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 2. Jalankan Prediksi TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # 3. Ambil Hasil
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_id = str(np.argmax(output_data[0]))
        confidence = np.max(output_data[0]) * 100
        
        nama_hasil = labels.get(class_id, "Kategori Tidak Dikenal")
        
        st.subheader(f"Hasil: {nama_hasil}")
        st.write(f"Keyakinan: {confidence:.2f}%")
