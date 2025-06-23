# 🌿 Klasifikasi Gambar Rempah-Rempah Indonesia

Proyek ini bertujuan untuk membangun model klasifikasi gambar rempah-rempah Indonesia menggunakan metode Transfer Learning dengan MobileNetV2 dan lapisan tambahan berbasis CNN (Dense, Dropout, dan Global Pooling). Model kemudian diintegrasikan ke dalam aplikasi web interaktif menggunakan Streamlit.

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
      GlobalAveragePooling2D(),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(64, activation='relu'),
      Dropout(0.2),
      Dense(36, activation='softmax')
  ])

- Optimizer: Adamax (lr=0.001)
- Loss Function: categorical_crossentropy
- Metrics: Accuracy, Precision, Recall

> ✳️ Setelah training awal (feature extraction), dilakukan **fine-tuning** dengan membuka sebagian lapisan akhir `MobileNetV2` (mulai dari layer ke-100) dan menggunakan learning rate rendah untuk meningkatkan akurasi secara bertahap.

## 📈 Hasil Pelatihan

- ✅ Akurasi Validasi Akhir: >93% setelah fine-tuning
- ✅ F1-score per kelas berkisar antara 0.80 hingga 1.00
- ✅ Confusion matrix menunjukkan performa yang baik dan konsisten antar kelas
- ✅ Performa tinggi bahkan pada kelas yang memiliki kemiripan visual

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