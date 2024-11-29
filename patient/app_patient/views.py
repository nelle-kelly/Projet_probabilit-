from django.shortcuts import render, redirect
from .models import Patient
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
import os
import base64
from patient.settings import MEDIA_ROOT
from io import BytesIO
from django.core.files.storage import default_storage

# Create your views here.
def base(request):
    # Définir le chemin du fichier
    chemin_fichier = os.path.join(MEDIA_ROOT, 'patients.xlsx')

    patients = []
    error_message = None
    success_message = None

    # Si le fichier Excel existe déjà, le charger
    if os.path.exists(chemin_fichier):
        try:
            df = pd.read_excel(chemin_fichier)
            patients = df.to_dict(orient='records')
        except Exception as e:
            error_message = f"Erreur lors de la lecture du fichier Excel : {str(e)}"
    else:
        error_message = "Le fichier Excel n'a pas été trouvé. Veuillez le télécharger."

    # Gestion de l'upload de fichier
    if request.method == "POST" and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        try:
            # Sauvegarder le fichier dans le répertoire MEDIA
            file_path = os.path.join(MEDIA_ROOT, 'patients.xlsx')
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            success_message = "Le fichier Excel a été téléchargé avec succès."
            # Charger à nouveau les patients après l'upload
            df = pd.read_excel(file_path)
            patients = df.to_dict(orient='records')
        except Exception as e:
            error_message = f"Erreur lors du téléchargement du fichier : {str(e)}"

    context = {
        'patient': patients,
        'error_message': error_message,
        'success_message': success_message,
    }

    return render(request, 'base.html', context)

def telecharger_excel(request):
    chemin_fichier = os.path.join(MEDIA_ROOT, 'patients.xlsx')

    # Vérifie si le fichier existe
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'rb') as fichier:
            response = HttpResponse(fichier.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename=patients.xlsx'
            return response

   
    return HttpResponse("Le fichier n'existe pas. Veuillez d'abord ajouter des patients.", status=404)


def visualiser_par_maladie(request):
    
    chemin_fichier = os.path.join(MEDIA_ROOT, 'patients.xlsx')

    # Vérifier si le fichier existe
    if not os.path.exists(chemin_fichier):
        return render(request, 'base.html', {'error_message': 'Fichier Excel non trouvé.'})

    
    df = pd.read_excel(chemin_fichier)

    pathologies = df['Pathologie'].unique()

    image_stream = None

    # Si la pathologie est sélectionnée via un formulaire, filtrer les patients
    pathologie_filtrée = request.GET.get('pathologie', '')
    if pathologie_filtrée:
        df = df[df['Pathologie'] == pathologie_filtrée]

    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Pathologie', palette='Set2')

    
    plt.title('Distribution des patients par pathologie')

    # Enregistrer le graphique dans un objet BytesIO pour l'afficher dans la réponse HTTP
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    
    image_stream = base64.b64encode(img_io.getvalue()).decode('utf-8')
   


    context = {
        'patients': df.to_dict(orient='records'),  # Liste des patients filtrés
        'pathologies': pathologies,  # Liste des pathologies uniques pour le formulaire
        'image_stream': image_stream,  # Image du graphique
        'pathologie_filtrée': pathologie_filtrée,  # Pathologie sélectionnée
         'image_stream': image_stream,
    }

    # Retourner le graphique dans la réponse HTTP comme image PNG
    return render(request, 'visualiser_par_maladie.html', context)
    #return HttpResponse(image_stream, content_type='image/png')

def index(request):
    if request.method == 'POST':
        nom = request.POST.get('nom')
        prenom = request.POST.get('prenom')
        sex = request.POST.get('sex')
        phone = request.POST.get('phone')
        statut = request.POST.get('status')

        patient = Patient(first_name=nom, last_name=prenom, sex=sex, phone=phone, status=statut)
        patient.save()

        #return redirect('index.html')


    return render(request, 'index.html')