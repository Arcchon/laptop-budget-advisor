# Laptop Budget Advisor — Sistem Cerdas Evaluasi & Rekomendasi Laptop Berbasis Budget

Aplikasi Web *Full-Stack* berbasis **Django** dan **Machine Learning (Scikit-Learn Random Forest Regressor)** yang berfungsi sebagai penasihat cerdas (*Budget Advisor*). Sistem membantu pengguna menganalisis kelayakan spesifikasi wajar di pasar berdasarkan nominal uang yang dimiliki, serta memberikan rekomendasi laptop terbaik (*Best Value for Money*) dari data riil.

---

## Masalah yang Diselesaikan
Banyak konsumen, khususnya mahasiswa dan masyarakat awam, mengalami kesulitan saat ingin membeli laptop. Masalah utamanya adalah:
1. **Asimetri Informasi Pasar**: Sering kali pembeli membeli laptop dengan spesifikasi perangkat keras di bawah standar kelayakan untuk harga yang mereka bayarkan karena minimnya pemahaman teknis.
2. **Kebutuhan vs Spesifikasi**: Pembeli bingung menentukan batas bawah spesifikasi (RAM, SSD, CPU, GPU) yang wajib dituntut dari penjual sesuai anggaran yang ada.
3. **Pembeda dari Proyek Kelompok**: Jika pada proyek kelompok sebelumnya mengangkat prediksi harga indekos lokal Madiun, pada tugas akhir individu ini dikembangkan proyek mandiri bertema **Laptop Budget Intelligence** dengan dataset baru berskala internasional, algoritma *Multi-Criteria Decision Scoring*, integrasi Machine Learning Regresi, dan antarmuka web Django interaktif.

---

## Dataset
* **Sumber Data**: [Kaggle Open Dataset — Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv)
* **Author / Uploader**: Muhammet Varli
* **Lisensi**: CC0: Public Domain (Open Data)
* **Jumlah Data**: 1.303 baris data spesifikasi laptop dunia (Merk, Tipe, CPU, GPU, RAM, Storage, Layar, Resolusi, dan Harga).

---

## Metode
1. **Data Preprocessing & Feature Engineering**:
   - Pembersihan fitur teks dan ekstraksi kapasitas numerik: RAM (GB), SSD (GB), Touchscreen (Boolean flag 0/1).
   - Kategorisasi kelompok CPU (Intel Core i3/i5/i7/i9, AMD Processor, Celeron) dan GPU (Nvidia, AMD, Intel).
   - Konversi mata uang dari Euro ke Rupiah stabil (asumsi kurs Rp 17.000 / Euro).
2. **Machine Learning Pipeline**:
   - `ColumnTransformer`: `OneHotEncoder` untuk fitur kategorikal & `passthrough` untuk fitur numerik.
   - Algoritma: **Random Forest Regressor** (`n_estimators=120`, `max_depth=16`, `random_state=42`).
   - Pembagian data: *Train/Test Split* (85% Training Data : 15% Testing Data).
3. **Algoritma Sistem Cerdas (Multi-Criteria Smart Recommender)**:
   - **Evaluasi Baseline Ambang Batas Wajar**: Menghitung spesifikasi minimum yang wajib didapat pengguna pada tier budget tertentu (Entry-level, Mid-range, High-performance, Flagship).
   - **Penilaian Skor Performa**: Pembobotan performa perangkat keras sesuai profil kebutuhan (Kuliah/Coding, Gaming/Render, Office/Mobilitas).
   - **Skor Nilai Manfaat (*Value for Money*)**:
     $$\text{Value Score} = \left( \frac{\text{Skor Performa}}{\text{Harga Rupiah}} \right) \times 10.000.000$$
     Menyaring dan mengurutkan unit laptop yang memberikan rasio performa tertinggi per Rupiah yang dibayarkan.

---

## Hasil & Evaluasi
Model Machine Learning diuji secara objektif pada 15% data uji (*test set* sebanyak 196 baris data independen):
* **Koefisien Determinasi ($R^2$ Score)**: **0.7901 (79.01%)** — Menandakan model mampu menjelaskan $\approx 79\%$ variansi harga pasar laptop secara akurat.
* **Mean Absolute Error (MAE)**: **Rp 3.480.453** — Simpangan rata-rata yang wajar di pasar laptop akibat adanya faktor tak terukur seperti kualitas bahan bodi (*chassis material*), reputasi *branding*, dan layanan purnajual/garansi.
* **Root Mean Squared Error (RMSE)**: **Rp 5.883.095**.

---

## Cara Menjalankan Aplikasi

1. **Aktifkan Virtual Environment**:
   ```bash
   # Di Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

2. **Jalankan Pelatihan Model (Jika ingin melatih ulang)**:
   ```bash
   python ml_engine/train_model.py
   ```

3. **Masuk ke Folder Web App & Jalankan Server**:
   ```bash
   cd web_app
   python manage.py runserver
   ```

4. **Akses Melalui Browser**:
   * Halaman Utama (Budget Advisor & ML Calculator): [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   * Katalog 1.303 Data Laptop: [http://127.0.0.1:8000/katalog/](http://127.0.0.1:8000/katalog/)

---

## Refleksi Singkat
* **Apa yang Berhasil**:
  Sistem berhasil menjembatani kebutuhan riil konsumen awam melalui konsep *"Uangmu berapa, dapat apa"*. Kombinasi evaluasi batas bawah spek wajar dan *scoring* rekomendasi berbasis data Kaggle berjalan sangat cepat tanpa beban API eksternal (100% lokal & *offline-ready*).
* **Keterbatasan Sistem**:
  Dataset mencakup rentang laptop hingga tahun rilis dataset tersebut (2020/2021) sehingga nama-nama prosesor generasi paling baru (Intel Gen 13/14 atau AMD Ryzen 7000/8000) belum terdaftar secara spesifik di database mentah.
* **Rencana Perbaikan & Pengembangan**:
  1. Melakukan *scraping* atau penambahan data berkala dari toko e-commerce lokal Indonesia (Tokopedia / Shopee).
  2. Menambahkan fitur komparasi *head-to-head* langsung antara dua unit laptop pilihan pengguna.
