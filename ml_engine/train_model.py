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

# Link dataset terbuka dari Kaggle (1.303 data riil pasar laptop)
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv"

def clean_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Sederhanakan merk populer agar model tidak bingung dengan merk langka
    top_brands = ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Apple', 'Toshiba']
    df['Company'] = df['Company'].apply(lambda x: x if x in top_brands else 'Lainnya')

    # Hilangkan tulisan 'GB' agar RAM jadi angka (integer)
    df['ram_gb'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)

    # Cek layar sentuh: 1 jika touchscreen, 0 jika layar biasa
    df['touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)

    # Kelompokkan jenis prosesor CPU utama
    def categorize_cpu(s):
        s = str(s)
        if 'Core i7' in s or 'Core i9' in s: return 'Intel Core i7/i9'
        elif 'Core i5' in s: return 'Intel Core i5'
        elif 'Core i3' in s: return 'Intel Core i3'
        elif 'AMD' in s: return 'AMD Processor'
        return 'Intel Celeron/Pentium'

    df['cpu_brand'] = df['Cpu'].apply(categorize_cpu)

    # Kelompokkan kartu grafis GPU
    def categorize_gpu(s):
        s = str(s)
        if 'Nvidia' in s: return 'Nvidia'
        elif 'AMD' in s: return 'AMD'
        return 'Intel'

    df['gpu_brand'] = df['Gpu'].apply(categorize_gpu)

    # Ambil kapasitas SSD (GB) dari kolom memory pakai regex
    def extract_ssd(s):
        match = re.search(r'(\d+)(GB|TB)\s*SSD', str(s))
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            return val * 1024 if unit == 'TB' else val
        return 0

    df['ssd_gb'] = df['Memory'].apply(extract_ssd)

    # Konversi harga dari Euro ke Rupiah (kurs stabil 17.000)
    df['harga_rupiah'] = (df['Price_euros'] * 17000).round(-4).astype(int)
    return df


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(base_dir, 'dataset', 'laptop_price.csv')
    model_save_path = os.path.join(current_dir, 'laptop_price_model.joblib')
    metrics_json_path = os.path.join(current_dir, 'model_evaluation.json')

    df = clean_data(csv_path)

    # Fitur input yang dipakai untuk prediksi
    cat_cols = ['Company', 'TypeName', 'cpu_brand', 'gpu_brand']
    num_cols = ['ram_gb', 'ssd_gb', 'touchscreen', 'Inches']

    X = df[cat_cols + num_cols]
    y = df['harga_rupiah']

    # Bagi data: 85% latih (1.107 data) dan 15% uji (196 data) - syarat Modul 15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Preprocessing: teks di-onehotencode, angka diteruskan langsung
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    # Model Random Forest Regressor (120 pohon keputusan)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1))
    ])

    # Latih model ke data training
    pipeline.fit(X_train, y_train)

    # Uji akurasi ke data testing independen
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)           # R² score: 0.7901 (akurasi ~79%)
    mae = mean_absolute_error(y_test, y_pred) # MAE: simpangan rata-rata ~3.4 juta
    rmse = root_mean_squared_error(y_test, y_pred)

    # Simpan model jadi file .joblib
    joblib.dump(pipeline, model_save_path)

    # Simpan ringkasan nilai metrik ke JSON
    eval_data = {
        'r2_score': round(float(r2), 4),
        'accuracy_pct': round(float(r2 * 100), 2),
        'mae_idr': int(mae),
        'rmse_idr': int(rmse)
    }
    with open(metrics_json_path, 'w') as f:
        json.dump(eval_data, f, indent=4)

    print(f"Selesai! Akurasi R²: {r2:.4f} | Rata-rata error MAE: Rp {int(mae):,}")

if __name__ == '__main__':
    train()
