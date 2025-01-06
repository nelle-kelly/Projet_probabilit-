from django.urls import path
from . import views

urlpatterns = [
   # path('home_patient', views.index, name='index'),
    path('Hom_base', views.base, name='base'),
    path('Lois_probabilite/', views.calculer_loi, name='lois_de_probabilite'),
    path('', views.telecharger_excel, name='excel'),
    path('visualiser-par-donnee/', views.visualiser_par_donnee, name='visualiser_par_donnee'),
    path('inferentielles/', views.inferentielles, name='test_inferentiel'),
    path('test_traitement/', views.test_traitement, name='test_traitement'),
]