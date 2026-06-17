import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Import fungsi preprocessing yang baru saja kita buat
# Karena file app.py ada di root, dan preprocessing.py ada di utils/
from utils.preprocessing import process_image, prepare_for_model

# Daftar Kelas (Sesuaikan dengan urutan model generator Anda saat training)
# Contoh: Jika menggunakan flow_from_directory, urutannya alfabetis
CLASSES = ['Diabetic Retinopathy', 'Glaukoma', 'Katarak', 'Normal'] 

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Citra Fundus",
    page_icon="👁️",
    layout="wide"
)

# 1. FUNGSI LOAD MODEL (Menggunakan cache agar cepat)
@st.cache_resource
def load_all_models():
    # Pastikan path ini sesuai dengan struktur direktori Anda
    model_paths = {
        'RGB': 'models/model_best_RGB.h5',
        'LAB': 'models/model_best_LAB.h5',
        'YCBCR': 'models/model_best_YCBCR.h5'
    }

    
    
    models = {}
    for scenario, path in model_paths.items():
        if os.path.exists(path):
            models[scenario] = tf.keras.models.load_model(path)
        else:
            # Jika belum ada modelnya, biarkan None
            models[scenario] = None
            
    return models

# Muat model saat aplikasi dijalankan
models = load_all_models()

# 2. ANTARMUKA PENGGUNA (UI)
# Header
st.markdown("<h1 style='text-align: center;'>Sistem Klasifikasi Penyakit Mata</h1>", unsafe_allow_html=True)
st.markdown("---")

# Bagian Upload & Skenario
col1, col2 = st.columns(2)

with col1:
    st.subheader("Unggah File")
    st.write("Unggah citra fundus retina untuk mendapatkan diagnosa otomatis.")
    uploaded_file = st.file_uploader("Pilih File 200MB per file • JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with col2:
    st.subheader("Pilih Skenario Warna")
    st.write("Pilih Skenario warna Pra-pemrosesan CLAHE untuk dilanjutkan ke klasifikasi")
    scenario_option = st.selectbox("Pilih Skenario", ('Pilih Skenario...', 'RGB', 'LAB', 'YCBCR'), label_visibility="collapsed")
    
    if scenario_option != 'Pilih Skenario...' and models[scenario_option] is None:
        st.warning(f"File model untuk {scenario_option} belum ditemukan di folder 'models'.")

st.markdown("<br>", unsafe_allow_html=True)

# Tombol Klasifikasi (Tengah)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    classify_btn = st.button("Mulai Klasifikasi", type="primary", use_container_width=True)

st.markdown("---")

# Bagian Citra
col_img1, col_img2 = st.columns(2)

processed_img_rgb = None

with col_img1:
    st.markdown("<h3 style='text-align: center;'>Citra Asli</h3>", unsafe_allow_html=True)
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, use_column_width=True)
    else:
        st.info("Silakan unggah gambar di atas.")

with col_img2:
    st.markdown("<h3 style='text-align: center;'>Dengan Skenario Warna</h3>", unsafe_allow_html=True)
    if uploaded_file is not None and scenario_option != 'Pilih Skenario...':
        try:
            processed_img_rgb = process_image(image_bytes, scenario_option)
            st.image(processed_img_rgb, use_column_width=True, caption=f"Skenario: {scenario_option}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            processed_img_rgb = None
    elif uploaded_file is not None:
        st.info("Silakan pilih skenario warna terlebih dahulu.")
    else:
        st.info("Gambar dengan skenario warna akan muncul di sini setelah diunggah.")

st.markdown("---")

# Bagian Hasil
st.markdown("<h2 style='text-align: center;'>Hasil</h2>", unsafe_allow_html=True)

if classify_btn:
    if uploaded_file is None:
        st.warning("Harap unggah gambar terlebih dahulu sebelum memulai klasifikasi.")
    elif processed_img_rgb is None:
        st.error("Gambar gagal diproses, klasifikasi tidak dapat dilanjutkan.")
    elif scenario_option == 'Pilih Skenario...':
        st.warning("Harap pilih skenario warna terlebih dahulu.")
    elif models[scenario_option] is None:
        st.error(f"Gagal mengklasifikasi. Model untuk {scenario_option} tidak tersedia.")
    else:
        with st.spinner("Sedang memproses dan memprediksi..."):
            tensor_img = prepare_for_model(processed_img_rgb)
            model = models[scenario_option]
            predictions = model.predict(tensor_img)
            
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASSES[predicted_class_index]
            confidence = np.max(predictions[0]) * 100
            
        st.success("Klasifikasi Selesai!")
        
        # Tampilkan metrik hasil di tengah
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            st.metric(label="Diagnosis Utama", value=predicted_class_name)
            st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
            
            st.markdown("#### Detail Probabilitas:")
            for i, class_name in enumerate(CLASSES):
                # Menampilkan progress bar untuk tiap probabilitas agar lebih visual
                st.progress(float(predictions[0][i]), text=f"{class_name}: {predictions[0][i]*100:.2f}%")
