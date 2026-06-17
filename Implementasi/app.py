import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Import fungsi preprocessing
from utils.preprocessing import process_all_scenarios, plot_histogram, prepare_for_model

# Daftar Kelas (Sesuaikan dengan urutan model generator Anda saat training)
CLASSES = ['Diabetic Retinopathy', 'Glaukoma', 'Katarak', 'Normal']

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Klasifikasi Penyakit Mata",
    # page_icon="👁️",
    layout="wide"
)

# ===== SESSION STATE =====
if 'image_buffer' not in st.session_state:
    st.session_state['image_buffer'] = None
if 'processed_images' not in st.session_state:
    st.session_state['processed_images'] = None
if 'is_classified' not in st.session_state:
    st.session_state['is_classified'] = False
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = {}


# ===== FUNGSI LOAD MODEL (Menggunakan cache agar cepat) =====
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
            models[scenario] = None

    return models


# Muat model saat aplikasi dijalankan
models = load_all_models()


# ================================================================
# ANTARMUKA PENGGUNA (UI) — Sesuai Wireframe Rancangan Antarmuka
# ================================================================

# ---------- CUSTOM CSS: Rata Tengah Upload & Tombol ----------
st.markdown("""
<style>
    /* Rata tengah untuk area upload citra */
    [data-testid="stFileUploader"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    [data-testid="stFileUploader"] section {
        width: 100%;
    }
    [data-testid="stFileUploader"] section > input + div {
        display: flex;
        justify-content: center;
    }

    /* Rata tengah untuk tombol dalam kontainer */
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>Sistem Klasifikasi Penyakit Mata</h1>", unsafe_allow_html=True)
st.markdown("---")


# ---------- BAGIAN UPLOAD (TERPUSAT) ----------
col_upload_l, col_upload_c, col_upload_r = st.columns([1, 2, 1])

with col_upload_c:
    st.markdown("<h3 style='text-align: center;'>Unggah File</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Unggah citra fundus retina untuk mendapatkan diagnosa otomatis.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-weight: bold;'>Pilih File 2MB per file • JPG, JPEG, PNG</p>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Validasi ukuran file (2MB = 2 * 1024 * 1024 bytes)
    if uploaded_file is not None:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("Ukuran file melebihi batas 2MB. Silakan pilih file yang lebih kecil.")
            uploaded_file = None
        else:
            # Simpan bytes ke session state (belum diproses, hanya buffer)
            image_bytes = uploaded_file.read()
            st.session_state['image_buffer'] = image_bytes

    # Tombol Mulai Klasifikasi (terpusat, di dalam kolom tengah)
    st.markdown("<br>", unsafe_allow_html=True)
    classify_btn = st.button("Mulai Klasifikasi", type="primary", use_container_width=True)

st.markdown("---")


# ---------- PROSES KLASIFIKASI (dipicu tombol) ----------
if classify_btn:
    if st.session_state['image_buffer'] is None:
        st.warning("Harap unggah gambar terlebih dahulu sebelum memulai klasifikasi.")
    else:
        # 1. Preprocessing — proses semua skenario warna
        with st.spinner("Sedang memproses citra..."):
            try:
                st.session_state['processed_images'] = process_all_scenarios(
                    st.session_state['image_buffer']
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.session_state['processed_images'] = None

        # 2. Klasifikasi — prediksi dengan 3 model
        if st.session_state['processed_images'] is not None:
            images = st.session_state['processed_images']
            predictions = {}

            with st.spinner("Sedang memprediksi dengan 3 model..."):
                for scenario in ['RGB', 'LAB', 'YCBCR']:
                    model = models.get(scenario)
                    if model is not None:
                        tensor_img = prepare_for_model(images[scenario])
                        preds = model.predict(tensor_img)

                        predicted_idx = np.argmax(preds[0])
                        predicted_name = CLASSES[predicted_idx]
                        confidence = np.max(preds[0]) * 100

                        predictions[scenario] = {
                            'class': predicted_name,
                            'confidence': confidence,
                            'probabilities': preds[0]
                        }
                    else:
                        predictions[scenario] = None

            st.session_state['predictions'] = predictions
            st.session_state['is_classified'] = True


# ---------- TAMPILAN HASIL (hanya muncul setelah klasifikasi) ----------
if st.session_state['is_classified'] and st.session_state['processed_images'] is not None:
    images = st.session_state['processed_images']

    # --- BAGIAN PERBANDINGAN CITRA ---
    st.markdown("<h2 style='text-align: center;'>Perbandingan Citra</h2>", unsafe_allow_html=True)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    citra_data = [
        ("Citra Asli", images['Asli']),
        ("RGB", images['RGB']),
        ("CIE Lab", images['LAB']),
        ("YCbCr", images['YCBCR']),
    ]

    for col, (label, img) in zip([col_c1, col_c2, col_c3, col_c4], citra_data):
        with col:
            st.image(img, caption=label, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BAGIAN GRAFIK HISTOGRAM CITRA ---
    st.markdown("<h2 style='text-align: center;'>Grafik Histogram Citra</h2>", unsafe_allow_html=True)

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)

    hist_data = [
        ("Histogram Citra Asli", images['Asli']),
        ("Histogram RGB-CLAHE", images['RGB']),
        ("Histogram CIE Lab-CLAHE", images['LAB']),
        ("Histogram YCbCr-CLAHE", images['YCBCR']),
    ]

    for col, (title, img) in zip([col_h1, col_h2, col_h3, col_h4], hist_data):
        with col:
            fig = plot_histogram(img, title)
            st.pyplot(fig)

    st.markdown("---")

    # --- BAGIAN HASIL KLASIFIKASI ---
    if st.session_state['predictions']:
        predictions = st.session_state['predictions']

        st.markdown("<h2 style='text-align: center;'>Hasil Klasifikasi</h2>", unsafe_allow_html=True)
        st.success("Klasifikasi Selesai!")

        col_r1, col_r2, col_r3 = st.columns(3)

        hasil_data = [
            ("Hasil Klasifikasi RGB", 'RGB', col_r1),
            ("Hasil Klasifikasi CIE Lab", 'LAB', col_r2),
            ("Hasil Klasifikasi YCbCr", 'YCBCR', col_r3),
        ]

        for title, scenario, col in hasil_data:
            with col:
                st.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)

                if images is not None:
                    st.image(images[scenario], use_container_width=True)

                pred = predictions.get(scenario)
                if pred is not None:
                    st.metric(label="Diagnosis", value=pred['class'])
                    st.metric(label="Confidence", value=f"{pred['confidence']:.2f}%")

                    st.markdown("**Detail Probabilitas:**")
                    for i, class_name in enumerate(CLASSES):
                        st.progress(
                            float(pred['probabilities'][i]),
                            text=f"{class_name}: {pred['probabilities'][i]*100:.2f}%"
                        )
                else:
                    st.error(f"Model {scenario} tidak tersedia.")
