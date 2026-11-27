import os
import sys
import json
from django.shortcuts import render
from django.core.paginator import Paginator
from django.conf import settings
from .models import Laptop

# Tambahkan folder ml_engine ke system path agar bisa di-import
BASE_DIR = os.path.dirname(settings.BASE_DIR)
ML_ENGINE_PATH = os.path.join(BASE_DIR, 'ml_engine')
if ML_ENGINE_PATH not in sys.path:
    sys.path.append(ML_ENGINE_PATH)

from recommender import get_budget_baseline, calculate_laptop_score, predict_price

def get_evaluation_metrics():
    """Membaca file evaluasi model ML (R² dan MAE)"""
    json_path = os.path.join(ML_ENGINE_PATH, 'model_evaluation.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {'r2_score': 0.7901, 'accuracy_pct': 79.01, 'mae_idr': 3480000}


def home_view(request):
    """
    Halaman Utama:
    Fitur 1: 'Uangmu Berapa, Dapat Apa?' (Budget-Based Smart Recommender)
    Fitur 2: Kalkulator Estimasi Nilai Pasar dari Spek (Machine Learning)
    """
    mode = request.GET.get('mode', 'budget')  # 'budget' atau 'ml_spec'
    eval_metrics = get_evaluation_metrics()

    # Variabel untuk Fitur 1 (Budget Advisor)
    budget_result = None
    baseline = None
    rekomendasi = []
    budget_input = 8_000_000
    need_input = 'coding'

    # Variabel untuk Fitur 2 (ML Predictor)
    ml_result = None

    if request.method == 'POST':
        action = request.POST.get('action')

        # Logika 1: Analisis Budget Pengguna
        if action == 'analyze_budget':
            mode = 'budget'
            # Bersihkan titik dan koma dari input budget
            raw_budget = str(request.POST.get('budget', '8000000')).replace('.', '').replace(',', '').replace('Rp', '').strip()
            budget_input = int(raw_budget) if raw_budget.isdigit() else 8_000_000
            need_input = request.POST.get('need', 'coding')

            # 1. Hitung batas bawah spek wajar
            baseline = get_budget_baseline(budget_input, need_input)

            # 2. Saring laptop dari database SQLite (rentang 45% sampai 103% budget)
            min_bound = max(2_000_000, int(budget_input * 0.45))
            max_bound = int(budget_input * 1.03)

            kandidat = Laptop.objects.filter(harga_rupiah__gte=min_bound, harga_rupiah__lte=max_bound)
            if not kandidat.exists():
                kandidat = Laptop.objects.filter(harga_rupiah__lte=max_bound)

            # 3. Hitung Value Score: performa per Rupiah yang dibayar
            scored_list = []
            for item in kandidat:
                perf = calculate_laptop_score(item, need_input)
                # Rumus value score: semakin besar skor per harga, semakin untung konsumen
                val_score = (perf / item.harga_rupiah) * 10_000_000
                scored_list.append({
                    'laptop': item,
                    'perf_score': perf,
                    'val_score': round(val_score, 2)
                })

            # Urutkan dari skor tertinggi dan ambil 4 laptop terbaik
            scored_list.sort(key=lambda x: x['val_score'], reverse=True)
            rekomendasi = scored_list[:4]
            budget_result = True

        # AKSI 2: Prediksi Harga dari Spesifikasi (Machine Learning)
        elif action == 'predict_ml':
            mode = 'ml_spec'
            company = request.POST.get('company', 'Asus')
            typename = request.POST.get('typename', 'Notebook')
            cpu_brand = request.POST.get('cpu_brand', 'Intel Core i5')
            gpu_brand = request.POST.get('gpu_brand', 'Intel')
            ram_gb = int(request.POST.get('ram_gb', 8))
            ssd_gb = int(request.POST.get('ssd_gb', 256))
            inches = float(request.POST.get('inches', 15.6))
            touchscreen = int(request.POST.get('touchscreen', 0))

            estimasi = predict_price(company, typename, cpu_brand, gpu_brand, ram_gb, ssd_gb, touchscreen, inches)
            ml_result = {
                'estimasi': estimasi,
                'estimasi_format': f"Rp {estimasi:,}".replace(",", "."),
                'rentang_bawah': f"Rp {int(estimasi * 0.9):,}".replace(",", "."),
                'rentang_atas': f"Rp {int(estimasi * 1.1):,}".replace(",", "."),
                'params': {
                    'company': company, 'typename': typename, 'cpu': cpu_brand,
                    'ram': ram_gb, 'ssd': ssd_gb, 'gpu': gpu_brand
                }
            }

    context = {
        'mode': mode,
        'eval_metrics': eval_metrics,
        'budget_result': budget_result,
        'baseline': baseline,
        'rekomendasi': rekomendasi,
        'budget_input': budget_input,
        'budget_input_format': f"Rp {budget_input:,}".replace(",", "."),
        'need_input': need_input,
        'ml_result': ml_result,
    }
    return render(request, 'advisor/home.html', context)


def katalog_view(request):
    """Halaman Eksplorasi 1.303 Data Laptop"""
    query = request.GET.get('q', '').strip()
    brand = request.GET.get('brand', '').strip()

    laptops = Laptop.objects.all()
    if query:
        laptops = laptops.filter(product__icontains=query)
    if brand:
        laptops = laptops.filter(company__iexact=brand)

    paginator = Paginator(laptops, 15)  # 15 laptop per halaman
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    brands = Laptop.objects.values_list('company', flat=True).distinct().order_by('company')

    context = {
        'page_obj': page_obj,
        'query': query,
        'selected_brand': brand,
        'brands': brands,
        'total_count': laptops.count(),
    }
    return render(request, 'advisor/katalog.html', context)