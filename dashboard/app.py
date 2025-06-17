import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import base64
from io import BytesIO

# Load model yang sudah disimpan
model = load_model("dashboard/model.h5", compile=False)

# Load labels dari file JSON
with open("dashboard/labels.json", "r") as f:
    class_indices = json.load(f)

# Balik mapping {label: index} menjadi {index: label}
labels = {v: k for k, v in class_indices.items()}

# Dictionary informasi tambahan rempah
with open("dashboard/rempah_info.json", "r", encoding="utf-8") as f:
    rempah_info = json.load(f)

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Klasifikasi Rempah-Rempah")

# Sidebar 
with st.sidebar:

    st.header("ℹ️ About")
    st.markdown("""
    Aplikasi ini merupakan sistem klasifikasi gambar rempah-rempah Indonesia menggunakan model deep learning berbasis MobileNetV2. Proyek ini bertujuan untuk membantu pelestarian budaya dan edukasi mengenai keragaman rempah Nusantara melalui teknologi. Model dilatih untuk mengenali 36 jenis rempah dengan akurasi validasi mencapai lebih dari 86%.
    """)

    st.header("🛠️ How to Use")
    st.markdown("""
    1. Siapkan gambar rempah-rempah (format: .jpg, .png, atau .jpeg).  
    2. Upload gambar melalui menu di halaman utama.  
    3. Klik tombol **"🔍 prediksi"** untuk melihat hasil klasifikasi.  
    4. Sistem akan menampilkan jenis rempah, tingkat akurasi, dan informasi tambahan jika tersedia.  
    
    Pastikan gambar cukup jelas dan menampilkan objek rempah secara dominan untuk hasil yang optimal.
    """)  

    st.header("👨‍💻 Developed by")
    st.markdown("""
    **Name:** Fiyanda Ma'muri  
    **Email:** fiyandamamuri@gmail.com  
    **LinkedIn:** [Profil LinkedIn](https://id.linkedin.com/in/fiyandamamuri/)  
    **GitHub:** [Profil GitHub](https://github.com/fiyandamamuri)  
    """)


# Fungsi untuk memproses gambar sebelum prediksi
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Streamlit UI
st.title("🔍 Klasifikasi Rempah-Rempah Indonesia")
st.image("dashboard/cover.svg", use_container_width=True)
st.subheader("Upload gambar rempah-rempah, model akan memprediksi jenisnya!")
st.caption("Note : klasifikasi hanya terbatas pada beberapa jenis rempah saja seperti, adas, andaliman, asam jawa, asam kandis, bawang bombai, bawang merah, bawang putih, bunga lawang, cabai, cengkeh, daun jeruk, daun kemangi, daun ketumbar, daun kunyit, daun pandan, daun salam, jahe, jinten, kapulaga, kayu manis, kayu secang, kemiri, kemukus, kencur, ketumbar, kluwek, kunyit, lada putih, lengkuas, pala, saffron, serai, temu kunci, vanili, wijen.")


uploaded_file = st.file_uploader("Unggah gambar rempah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    # Konversi gambar ke base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    # Tampilkan gambar dengan ukuran asli
    st.markdown(
        f"""
        <div style="text-align: center">
            <img src="data:image/png;base64,{img_b64}" style="max-width: 500px; height: auto;" alt="Gambar" />
            <p><em>🖼️ Gambar yang diunggah</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Prediksi saat tombol ditekan
    if st.button("🔍 prediksi"):
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]
        confidence = float(np.max(predictions)) * 100 

        st.success(f"🧠 Prediksi : {predicted_label} ")
        st.success(f"📊 Confidence : {confidence:.2f}%")


        # Menampilkan informasi tambahan jika tersedia
        if predicted_label in rempah_info:
            info = rempah_info[predicted_label]
            st.subheader("ℹ️ **Informasi Tambahan**")
            st.write(f"**Nama:** {info['Nama']}")
            st.write(f"**Aroma:** {info['Aroma']}")
            st.write(f"**Rasa:** {info['Rasa']}")
            st.write(f"**Kegunaan:** {info['Kegunaan']}")
            st.write(f"**Manfaat Kesehatan:** {info['Manfaat Kesehatan']}")
        else:
            st.warning("⚠️ Informasi tambahan belum tersedia untuk gambar ini.")
st.caption('Copyright © Fiyanda Mamuri - 2025')