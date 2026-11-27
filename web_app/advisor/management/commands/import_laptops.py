import os
import re
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from advisor.models import Laptop

class Command(BaseCommand):
    help = "Mengimpor 1.303 data laptop dari dataset Kaggle ke database SQLite"

    def handle(self, *args, **kwargs):
        # Jalur ke dataset di folder luar
        base_dir = os.path.dirname(settings.BASE_DIR)
        csv_path = os.path.join(base_dir, 'dataset', 'laptop_price.csv')

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"Dataset tidak ditemukan di: {csv_path}"))
            return

        self.stdout.write("Membaca dataset Kaggle...")
        df = pd.read_csv(csv_path, encoding='latin-1')

        # Hapus data lama jika sudah ada agar tidak duplikat
        Laptop.objects.all().delete()

        def extract_ssd(mem_str):
            match = re.search(r'(\d+)(GB|TB)\s*SSD', str(mem_str))
            if match:
                val = int(match.group(1))
                unit = match.group(2)
                return val * 1024 if unit == 'TB' else val
            return 0

        def get_cpu_brand(cpu_str):
            s = str(cpu_str)
            if 'Core i7' in s or 'Core i9' in s:
                return 'Intel Core i7/i9'
            elif 'Core i5' in s:
                return 'Intel Core i5'
            elif 'Core i3' in s:
                return 'Intel Core i3'
            elif 'AMD' in s:
                return 'AMD Processor'
            return 'Intel Celeron/Pentium'

        def get_gpu_brand(gpu_str):
            s = str(gpu_str)
            if 'Nvidia' in s:
                return 'Nvidia'
            elif 'AMD' in s:
                return 'AMD'
            return 'Intel'

        laptops_to_create = []
        for _, row in df.iterrows():
            ram = int(str(row['Ram']).replace('GB', '').strip())
            ssd = extract_ssd(row['Memory'])
            touchscreen = True if 'Touchscreen' in str(row['ScreenResolution']) else False
            harga_rp = int(round(float(row['Price_euros']) * 17000, -4))

            laptops_to_create.append(
                Laptop(
                    company=row['Company'].strip(),
                    product=row['Product'].strip(),
                    typename=row['TypeName'].strip(),
                    inches=float(row['Inches']),
                    cpu=row['Cpu'].strip(),
                    cpu_brand=get_cpu_brand(row['Cpu']),
                    ram_gb=ram,
                    memory=row['Memory'].strip(),
                    ssd_gb=ssd,
                    gpu=row['Gpu'].strip(),
                    gpu_brand=get_gpu_brand(row['Gpu']),
                    opsys=row['OpSys'].strip(),
                    touchscreen=touchscreen,
                    price_euros=float(row['Price_euros']),
                    harga_rupiah=harga_rp
                )
            )

        # Simpan sekaligus (bulk_create) agar cepat
        Laptop.objects.bulk_create(laptops_to_create)
        self.stdout.write(self.style.SUCCESS(f"✅ SUKSES: Berhasil mengimpor {len(laptops_to_create):,} laptop ke database SQLite!"))