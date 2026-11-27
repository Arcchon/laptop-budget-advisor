import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'laptop_price_model.joblib')

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is None:
        if os.path.exists(MODEL_PATH):
            _cached_model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model belum ada di {MODEL_PATH}")
    return _cached_model


def get_budget_baseline(budget_idr: int, need: str = 'coding'):
    """
    Menghasilkan 'Spek Minimal yang Wajib Didapat' berdasarkan budget pengguna.
    """
    if budget_idr < 5_000_000:
        tier = "Entry-Level / Pelajar"
        min_ram = 4
        ideal_ram = 8
        min_ssd = 128
        min_cpu = "Intel Celeron / Core i3 atau AMD Athlon/Ryzen 3"
        gpu_tier = "Integrated Graphics (Intel UHD / Radeon Vega)"
        screen_note = "14.0 - 15.6 inci HD / FHD Standard"
        saran = "Di bawah 5 juta, pastikan laptop sudah memakai SSD (bukan HDD) dan memiliki slot RAM yang bisa ditambah."
    elif budget_idr <= 9_000_000:
        tier = "Mid-Range / Mahasiswa Standard (Sweet Spot)"
        min_ram = 8
        ideal_ram = 16
        min_ssd = 256
        min_cpu = "Intel Core i5 (Gen 10+) atau AMD Ryzen 5"
        gpu_tier = "Integrated Iris Xe / Radeon Graphics"
        screen_note = "14.0 - 15.6 inci Full HD (1080p)"
        saran = "Budget ini sangat ideal untuk mahasiswa! Kamu berhak mendapatkan minimal Core i5 / Ryzen 5 dengan SSD 256GB - 512GB."
    elif budget_idr <= 16_000_000:
        tier = "High-Performance / Gaming & Content Creator"
        min_ram = 16
        ideal_ram = 16
        min_ssd = 512
        min_cpu = "Intel Core i7 / i5 seri H atau AMD Ryzen 7"
        gpu_tier = "Dedicated GPU: Nvidia GeForce GTX / RTX series"
        screen_note = "15.6 inci IPS (120Hz - 144Hz Refresh Rate)"
        saran = "Pada budget ini wajib mendapatkan kartu grafis diskrit (Nvidia GTX/RTX) untuk kebutuhan render video atau gaming berat."
    else:
        tier = "Flagship / Professional Workstation"
        min_ram = 16
        ideal_ram = 32
        min_ssd = 512
        min_cpu = "Intel Core i7/i9 seri H / AMD Ryzen 9 / Apple M-Series"
        gpu_tier = "High-End Nvidia RTX atau Apple Silicon GPU"
        screen_note = "Layar Akurasi Warna Tinggi (100% sRGB / Retina / OLED)"
        saran = "Perhatikan kualitas material bodi (Aluminium), sistem pendingin ganda, dan daya tahan baterai."

    return {
        'budget': budget_idr,
        'tier': tier,
        'min_ram': min_ram,
        'ideal_ram': ideal_ram,
        'min_ssd': min_ssd,
        'min_cpu': min_cpu,
        'gpu_tier': gpu_tier,
        'screen_note': screen_note,
        'saran': saran
    }


def calculate_laptop_score(laptop, need: str = 'coding') -> float:
    """
    Menghitung skor performa teknis (0 - 100) berdasarkan bobot kebutuhan.
    """
    # 1. Skor CPU
    cpu_str = f"{laptop.cpu} {laptop.cpu_brand}"
    if 'Core i9' in cpu_str or 'Ryzen 9' in cpu_str: cpu_score = 100
    elif 'Core i7' in cpu_str or 'Ryzen 7' in cpu_str: cpu_score = 85
    elif 'Core i5' in cpu_str or 'Ryzen 5' in cpu_str: cpu_score = 70
    elif 'Core i3' in cpu_str or 'Ryzen 3' in cpu_str: cpu_score = 50
    else: cpu_score = 30

    # 2. Skor GPU
    gpu_str = f"{laptop.gpu} {laptop.gpu_brand}"
    if 'RTX' in gpu_str or 'GTX' in gpu_str: gpu_score = 95
    elif 'Nvidia' in gpu_str: gpu_score = 80
    elif 'Radeon' in gpu_str or 'AMD' in gpu_str: gpu_score = 70
    elif 'Iris' in gpu_str: gpu_score = 55
    else: gpu_score = 35

    # 3. Skor RAM (32GB = 100)
    ram_score = min(100.0, (laptop.ram_gb / 32.0) * 100.0)

    # 4. Skor SSD (1024GB = 100)
    ssd_score = min(100.0, (laptop.ssd_gb / 1024.0) * 100.0)

    # Pembobotan Multi-Kriteria
    if need == 'gaming':
        perf = (gpu_score * 0.45) + (cpu_score * 0.30) + (ram_score * 0.15) + (ssd_score * 0.10)
    elif need == 'office':
        perf = (cpu_score * 0.30) + (ram_score * 0.35) + (ssd_score * 0.25) + (gpu_score * 0.10)
    else:  # coding / mahasiswa TI
        perf = (cpu_score * 0.40) + (ram_score * 0.35) + (ssd_score * 0.15) + (gpu_score * 0.10)

    return round(perf, 1)


def predict_price(company, typename, cpu_brand, gpu_brand, ram_gb, ssd_gb, touchscreen, inches):
    """
    Prediksi harga menggunakan model Machine Learning yang telah dilatih.
    """
    model = get_model()
    df_input = pd.DataFrame([{
        'Company': company,
        'TypeName': typename,
        'cpu_brand': cpu_brand,
        'gpu_brand': gpu_brand,
        'ram_gb': int(ram_gb),
        'ssd_gb': int(ssd_gb),
        'touchscreen': int(touchscreen),
        'Inches': float(inches)
    }])
    pred = model.predict(df_input)[0]
    return int(round(pred / 50000.0) * 50000)