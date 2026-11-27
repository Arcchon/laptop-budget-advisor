from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('katalog/', views.katalog_view, name='katalog'),
]