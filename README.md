# 🌿 Klasifikasi Gambar Rempah-Rempah Indonesia

Proyek ini bertujuan untuk membangun model klasifikasi gambar **rempah-rempah Indonesia** menggunakan metode **Transfer Learning dengan MobileNetV2** serta arsitektur tambahan berbasis CNN (Dense, Dropout, GlobalPooling). Model ini diintegrasikan ke dalam aplikasi web menggunakan **Streamlit** agar dapat diakses dan digunakan secara interaktif.

## 📂 Dataset

Dataset terdiri dari gambar berbagai jenis rempah-rempah Indonesia yang dikumpulkan dan dikurasi secara manual dari berbagai sumber. Kelas yang tersedia saat ini:

> **Adas, Andaliman, Asam Jawa, Asam Kandis, Bawang Bombai, Bawang Merah, Bawang Putih, Bunga Lawang, Cabai, Cengkeh, Daun Jeruk, Daun Kemangi, Daun Ketumbar, Daun Kunyit, Daun Pandan, Daun Salam, Jahe, Jinten, Kapulaga, Kayu Manis, Kayu Secang, Kemiri, Kemukus, Kencur, Ketumbar, Kluwek, Kunyit, Lada Putih, Lengkuas, Pala, Saffron, Serai, Temu Kunci, Vanili, Wijen** serta satu kelas **Bukan Rempah**.

Total: **36 kelas**

**link datasets** : https://drive.google.com/file/d/1lxy6xCOOas5TGmRLCilpVsIkMiBy8xQE/view?usp=sharing



## 🧠 Arsitektur Model

Model dikembangkan menggunakan Keras dan TensorFlow dengan pendekatan transfer learning:

- Base Model: `MobileNetV2` (tanpa top layers, pretrained on ImageNet)
- Input Image Size: **224x224 px**
- Arsitektur tambahan:
  ```python
  Sequential([
      base_model,
      GlobalMaxPooling2D(),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(64, activation='relu'),
      Dropout(0.2),
      Dense(36, activation='softmax')
  ])

- Optimizer: Adamax (lr=0.001)
- Loss Function: categorical_crossentropy
- Metrics: Accuracy, Precision, Recall

## 📈 Hasil Pelatihan

- Akurasi Validasi Tertinggi: 86.88% pada epoch ke-28
- F1-score per kelas berkisar antara 0.75 – 0.98
- Performa stabil pada sebagian besar kelas meskipun terdapat kemiripan visual antar rempah
- Disertai dengan metrik evaluasi precision, recall, dan confusion matrix

## 💻 Tampilan Aplikasi

Aplikasi ini dibangun menggunakan Streamlit dan memiliki fitur:

- Upload gambar rempah (jpg/png/jpeg)
- Prediksi jenis rempah
- Menampilkan persentase keyakinan model
- Menyajikan informasi tambahan (aroma, rasa, manfaat, kegunaan) dari file rempah_info.json
- Antarmuka sederhana dan interaktif

## 🛠️ Teknologi yang Digunakan

- Python
- TensorFlow & Keras
- MobileNetV2 (Transfer Learning)
- Streamlit
- NumPy, PIL, Matplotlib
-JSON untuk data informasi rempah