from django.db import models

class Laptop(models.Model):
    company = models.CharField(max_length=50, verbose_name="Merk")
    product = models.CharField(max_length=150, verbose_name="Nama Seri")
    typename = models.CharField(max_length=50, verbose_name="Kategori")
    inches = models.FloatField(verbose_name="Ukuran Layar")
    cpu = models.CharField(max_length=150, verbose_name="Spesifikasi CPU")
    cpu_brand = models.CharField(max_length=50, verbose_name="Keluarga CPU")
    ram_gb = models.IntegerField(verbose_name="RAM (GB)")
    memory = models.CharField(max_length=150, verbose_name="Penyimpanan Asli")
    ssd_gb = models.IntegerField(default=0, verbose_name="Kapasitas SSD (GB)")
    gpu = models.CharField(max_length=150, verbose_name="Spesifikasi GPU")
    gpu_brand = models.CharField(max_length=50, verbose_name="Brand GPU")
    opsys = models.CharField(max_length=50, verbose_name="Sistem Operasi")
    touchscreen = models.BooleanField(default=False, verbose_name="Layar Sentuh")
    price_euros = models.FloatField(verbose_name="Harga Euro")
    harga_rupiah = models.BigIntegerField(verbose_name="Harga Rupiah")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Data Laptop"
        verbose_name_plural = "Katalog Laptop"
        ordering = ['harga_rupiah']

    @property
    def harga_format(self):
        """Format harga ke rupiah dengan titik pemisah ribuan"""
        return f"Rp {self.harga_rupiah:,}".replace(",", ".")

    def __str__(self):
        return f"{self.company} {self.product} — {self.harga_format}"