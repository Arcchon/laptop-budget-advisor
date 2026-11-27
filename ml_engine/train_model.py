"""
========================================================================================
LAPTOP BUDGET ADVISOR - SISTEM CERDAS EVALUASI & REKOMENDASI LAPTOP BERBASIS BUDGET
Mata Kuliah   : Sistem Cerdas (TI24317) - Pertemuan 15 (UAS Proyek Akhir Individu)
Program Studi : D3 Teknologi Informasi - Politeknik Negeri Madiun

SUMBER DATASET TERBUKA (KAGGLE):
URL Dataset   : https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv
Lisensi Data  : CC0: Public Domain (Open Data)
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

def clean_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path, encoding='latin-1')

    # Standarisasi Brand
    top_brands = ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Apple', 'Toshiba']
    df['Company'] = df['Company'].apply(lambda x: x if x in top_brands else 'Lainnya')

    # Ekstraksi RAM & Layar Sentuh
    df['ram_gb'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
    df['touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)

    # Kategorisasi CPU
    def categorize_cpu(s):
        s = str(s)
        if 'Core i7' in s or 'Core i9' in s: return 'Intel Core i7/i9'
        elif 'Core i5' in s: return 'Intel Core i5'
        elif 'Core i3' in s: return 'Intel Core i3'
        elif 'AMD' in s: return 'AMD Processor'
        return 'Intel Celeron/Pentium'

    df['cpu_brand'] = df['Cpu'].apply(categorize_cpu)

    # Kategorisasi GPU
    def categorize_gpu(s):
        s = str(s)
        if 'Nvidia' in s: return 'Nvidia'
        elif 'AMD' in s: return 'AMD'
        return 'Intel'

    df['gpu_brand'] = df['Gpu'].apply(categorize_gpu)

    # Ekstraksi SSD (GB)
    def extract_ssd(s):
        match = re.search(r'(\d+)(GB|TB)\s*SSD', str(s))
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            return val * 1024 if unit == 'TB' else val
        return 0

    df['ssd_gb'] = df['Memory'].apply(extract_ssd)

    # Konversi Euro ke Rupiah (Kurs 17.000)
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

    cat_cols = ['Company', 'TypeName', 'cpu_brand', 'gpu_brand']
    num_cols = ['ram_gb', 'ssd_gb', 'touchscreen', 'Inches']

    X = df[cat_cols + num_cols]
    y = df['harga_rupiah']

    # Train / Test Split (85% Training, 15% Testing) sesuai ketentuan Modul 15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

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

    # Evaluasi metrik
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Simpan model binary
    joblib.dump(pipeline, model_save_path)

    # Simpan hasil evaluasi ke JSON agar mudah dibaca Django
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