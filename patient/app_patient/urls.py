from django.urls import path
from . import views

urlpatterns = [
    path('home_patient', views.index, name='index'),
    path('', views.base, name='base'),
    path('ajouter_patient', views.telecharger_excel, name='telecharger_excel'),
    path('visualiser-par-maladie/', views.visualiser_par_maladie, name='visualiser_par_maladie'),
]