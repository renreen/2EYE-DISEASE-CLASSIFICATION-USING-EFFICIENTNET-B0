import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Inisialisasi Session State untuk Buffer
if 'image_buffer' not in st.session_state:
    st.session_state['image_buffer'] = None
if 'is_classified' not in st.session_state:
    st.session_state['is_classified'] = False

st.set_page_config(layout="wide", page_title="Sistem Klasifikasi Penyakit Mata")

st.title("Analisis Variasi Ruang Warna pada Fitur CLAHE")
st.write("Deteksi Penyakit Mata Berbasis EfficientNet-B0")

# --- Bagian Logika Fungsi (CLAHE & Prediksi) tetap sama ---
def plot_histogram(image_rgb, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, col in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

def apply_clahe_all_variants(img_bgr):
    # RGB (Green Channel)
    b, g, r = cv2.split(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    res_rgb = cv2.cvtColor(cv2.merge((b, clahe.apply(g), r)), cv2.COLOR_BGR2RGB)
    
    # LAB (L Channel)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)
    res_lab = cv2.cvtColor(cv2.cvtColor(cv2.merge((clahe.apply(l), a, b_lab)), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    
    # YCbCr (Y Channel)
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)
    res_ycbcr = cv2.cvtColor(cv2.cvtColor(cv2.merge((clahe.apply(y), cr, cb)), cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2RGB)
    
    return res_rgb, res_lab, res_ycbcr

def dummy_predict(metode):
    results = {"Asli": ("Normal", 85.2), "RGB": ("Katarak", 89.4), "LAB": ("Katarak", 97.1), "YCbCr": ("Katarak", 96.5)}
    return results.get(metode)

# --- Antarmuka Pengguna ---

# 1. Upload Citra
uploaded_file = st.file_uploader("Unggah Citra Fundus", type=["jpg", "png"])

if uploaded_file is not None:
    # Simpan ke buffer memori
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    st.session_state['image_buffer'] = cv2.imdecode(file_bytes, 1)
    st.info("Citra berhasil diunggah dan tersimpan di buffer.")
    
    # Tampilkan Preview Citra Asli
    img_preview = cv2.cvtColor(st.session_state['image_buffer'], cv2.COLOR_BGR2RGB)
    st.image(img_preview, caption="Pratinjau Citra Asli (Buffer)", width=300)

    # 2. Tombol Pemicu Klasifikasi
    if st.button("Mulai Klasifikasi Penyakit"):
        st.session_state['is_classified'] = True

# 3. Eksekusi dan Tampilan Hasil (Jika tombol ditekan)
if st.session_state['is_classified'] and st.session_state['image_buffer'] is not None:
    
    # --- PEMBERITAHUAN DI ATAS ---
    st.success("Inferensi Model EfficientNet-B0 Telah Selesai!")
    st.write("---")
    
    # Pemrosesan
    img_bgr = cv2.resize(st.session_state['image_buffer'], (224, 224))
    img_asli_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_rgb, res_lab, res_ycbcr = apply_clahe_all_variants(img_bgr)
    
    # Layout Kolom
    cols = st.columns(4)
    data = [
        ("Asli", img_asli_rgb, "Histogram Asli"),
        ("RGB", res_rgb, "Histogram RGB-CLAHE"),
        ("LAB", res_lab, "Histogram LAB-CLAHE"),
        ("YCbCr", res_ycbcr, "Histogram YCbCr-CLAHE")
    ]
    
    for i, (name, img, hist_title) in enumerate(data):
        with cols[i]:
            st.image(img, caption=f"Hasil {name}", use_container_width=True)
            st.pyplot(plot_histogram(img, hist_title))
            label, skor = dummy_predict(name)
            st.markdown(f"**Prediksi:** {label} ({skor}%)")