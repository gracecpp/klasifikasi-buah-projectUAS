import streamlit as st
import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite # Library pengganti TF yang sangat ringan

st.set_page_config(page_title="Klasifikasi Buah UAS", layout="centered")
st.title("🍎 Klasifikasi Buah & Sayur (Lite)")

# 1. Load Label
@st.cache_data
def load_labels():
    with open('klasifikasi class name.json', 'r') as f:
        return json.load(f)

# 2. Load Model TFLite
@st.cache_resource
def load_tflite_model():
    a
    interpreter = tflite.Interpreter(model_path="model_buah.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    labels = load_labels()
    interpreter = load_tflite_model()
    st.success("Model Lite Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat file: {e}. Pastikan file .tflite sudah ada.")
    st.stop()

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Preprocessing Manual (Tanpa TF)
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button('Prediksi'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_id = str(np.argmax(predictions[0]))
        
        nama_buah = labels.get(class_id, "Tidak Diketahui")
        st.subheader(f"Hasil: {nama_buah}")
