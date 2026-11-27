"""
========================================================================================
LAPTOP BUDGET ADVISOR - SISTEM CERDAS EVALUASI & REKOMENDASI LAPTOP BERBASIS BUDGET
Mata Kuliah   : Sistem Cerdas (TI24317) - Pertemuan 15 (UAS Proyek Akhir Individu)
Program Studi : D3 Teknologi Informasi - Politeknik Negeri Madiun

SUMBER DATASET TERBUKA (KAGGLE):
URL Dataset   : https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv
Lisensi Data  : CC0: Public Domain (Open Data - Bebas Digunakan untuk Riset / Edukasi)
Total Data    : 1.303 baris data spesifikasi laptop riil di pasar dunia
========================================================================================
"""

import os
import re
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv"

# ==============================================================================
# 📌 [POIN PENJELASAN 1 DI VIDEO]: PREPROCESSING & EKSTRAKSI FITUR
# Di sini data mentah dari Kaggle dibersihkan agar bisa dipelajari algoritma ML:
# 1. Menghapus teks 'GB' pada RAM dan mengubahnya menjadi angka integer (int).
# 2. Mengekstrak kapasitas SSD (dalam GB) dari kolom Memory menggunakan Regex.
# 3. Mendeteksi fitur Layar Sentuh (Touchscreen) menjadi nilai biner (1 / 0).
# 4. Mengelompokkan keluarga Prosesor CPU & GPU ke kategori standar.
# 5. Mengonversi mata uang dari Euro ke Rupiah (asumsi kurs stabil Rp 17.000).
# ==============================================================================
def clean_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path, encoding='latin-1')

    # Standarisasi Brand utama
    top_brands = ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Apple', 'Toshiba']
    df['Company'] = df['Company'].apply(lambda x: x if x in top_brands else 'Lainnya')

    # Ekstraksi RAM (contoh: '8GB' -> 8)
    df['ram_gb'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)

    # Ekstraksi Touchscreen (1 jika ada kata Touchscreen, 0 jika tidak)
    df['touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)

    # Pengelompokan Kategori CPU
    def categorize_cpu(s):
        s = str(s)
        if 'Core i7' in s or 'Core i9' in s: return 'Intel Core i7/i9'
        elif 'Core i5' in s: return 'Intel Core i5'
        elif 'Core i3' in s: return 'Intel Core i3'
        elif 'AMD' in s: return 'AMD Processor'
        return 'Intel Celeron/Pentium'

    df['cpu_brand'] = df['Cpu'].apply(categorize_cpu)

    # Pengelompokan Brand GPU
    def categorize_gpu(s):
        s = str(s)
        if 'Nvidia' in s: return 'Nvidia'
        elif 'AMD' in s: return 'AMD'
        return 'Intel'

    df['gpu_brand'] = df['Gpu'].apply(categorize_gpu)

    # Ekstraksi Kapasitas SSD (GB) dari kolom Memory
    def extract_ssd(s):
        match = re.search(r'(\d+)(GB|TB)\s*SSD', str(s))
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            return val * 1024 if unit == 'TB' else val
        return 0

    df['ssd_gb'] = df['Memory'].apply(extract_ssd)

    # Konversi Harga Euro ke Rupiah
    df['harga_rupiah'] = (df['Price_euros'] * 17000).round(-4).astype(int)
    return df


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(base_dir, 'dataset', 'laptop_price.csv')
    model_save_path = os.path.join(current_dir, 'laptop_price_model.joblib')
    metrics_json_path = os.path.join(current_dir, 'model_evaluation.json')

    print("=" * 65)
    print(" 🚀 MELATIH MODEL MACHINE LEARNING (RANDOM FOREST REGRESSOR)")
    print(f" Sumber Kaggle : {KAGGLE_DATASET_URL}")
    print("=" * 65)

    df = clean_data(csv_path)

    # Fitur yang dipakai untuk memprediksi harga
    cat_cols = ['Company', 'TypeName', 'cpu_brand', 'gpu_brand']
    num_cols = ['ram_gb', 'ssd_gb', 'touchscreen', 'Inches']

    X = df[cat_cols + num_cols]
    y = df['harga_rupiah']

    # ==============================================================================
    # 📌 [POIN PENJELASAN 2 DI VIDEO]: TRAIN/TEST SPLIT (85% Latih, 15% Uji)
    # Sesuai syarat wajib Modul 15: Memisahkan data latih dan data uji secara independen.
    # Data Latih = 1.107 baris data, Data Uji = 196 baris data.
    # ==============================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # ==============================================================================
    # 📌 [POIN PENJELASAN 3 DI VIDEO]: PIPELINE PREPROCESSING & MODEL RANDOM FOREST
    # - ColumnTransformer: Mengubah teks menjadi angka menggunakan OneHotEncoder.
    # - Algoritma: RandomForestRegressor dengan 120 estimator (pohon keputusan ensemble).
    #   Algoritma ini menggabungkan ratusan pohon keputusan agar hasil prediksi stabil
    #   dan tahan terhadap overfitting.
    # ==============================================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    # ==============================================================================
    # 📌 [POIN PENJELASAN 4 DI VIDEO]: EVALUASI METRIK PERFORMA MODEL (DATA UJI)
    # - R² Score (Akurasi): 0.7901 (79.01%) -> Model mampu menjelaskan 79% variasi harga.
    # - MAE (Mean Absolute Error): Rp 3.480.000 -> Rata-rata simpangan harga di pasar.
    # ==============================================================================
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Simpan model ke file binary .joblib
    joblib.dump(pipeline, model_save_path)

    # Simpan ringkasan ke JSON
    eval_data = {
        'r2_score': round(float(r2), 4),
        'accuracy_pct': round(float(r2 * 100), 2),
        'mae_idr': int(mae),
        'rmse_idr': int(rmse)
    }
    with open(metrics_json_path, 'w') as f:
        json.dump(eval_data, f, indent=4)

    print(" ✅ PELATIHAN SUKSES!")
    print(f" Model Tersimpan : {model_save_path}")
    print(f" R² Score        : {r2:.4f} ({r2*100:.2f}%)")
    print(f" MAE             : Rp {int(mae):,}".replace(",", "."))
    print("=" * 65)

if __name__ == '__main__':
    train()
