from django.shortcuts import render, redirect
from .models import Patient
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
import os
import math
import random
from django.http import JsonResponse

import plotly.express as px
import plotly.figure_factory as ff
from django import forms
from scipy.stats import bernoulli, norm, t, expon, poisson, uniform

#from .forms import BinomialForm,FileUploadForm,ExponentielleForm,TraitementForm,UniformeForm,PoissonForm,NormaleForm
import base64
from patient.settings import MEDIA_ROOT
from io import BytesIO
from django.core.files.storage import default_storage

# Create your views here.
class FileUploadForm(forms.Form):
    file = forms.FileField(
        label='Fichier Excel',
        required=True,
        widget=forms.ClearableFileInput(
            attrs={
                'class': 'form-control-file',  # Ajoute une classe CSS Bootstrap
                'style': 'border: 2px solid #000bf; padding: 10px; border-radius: 5px;',  # Style inline
                'accept': '.xls,.xlsx',  # Limite les fichiers sélectionnés dans la boîte de dialogue
            }
        ),
    )



def base(request):
    uploaded_file_path = None
    data_preview = None
    error_message = None
    success_message = None
    colonnes_disponibles = []
    statistiques = {}

    # Gestion de l'upload de fichier
    if request.method == "POST" and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        try:
            # Sauvegarder le fichier dans MEDIA_ROOT
            uploaded_file_path = os.path.join(MEDIA_ROOT, uploaded_file.name)
            with default_storage.open(uploaded_file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            success_message = "Le fichier Excel a été téléchargé avec succès."

            # Charger et afficher les premières lignes du fichier
            df = pd.read_excel(uploaded_file_path)
            data_preview = df.head(10).to_dict(orient='records')
            colonnes_disponibles = df.columns.tolist()
        except Exception as e:
            error_message = f"Erreur lors du traitement du fichier : {str(e)}"
    
    # Charger le fichier si déjà téléchargé
    elif uploaded_file_path and os.path.exists(uploaded_file_path):
        try:
            df = pd.read_excel(uploaded_file_path)
            colonnes_disponibles = df.columns.tolist()
        except Exception as e:
            error_message = f"Erreur lors de la lecture du fichier existant : {str(e)}"

    # Calculs statistiques si une colonne est sélectionnée
    colonne_selectionnee = request.GET.get('colonne', '')
    if colonne_selectionnee and colonne_selectionnee in colonnes_disponibles:
        # Vérifiez si la colonne contient des données numériques
        if pd.api.types.is_numeric_dtype(df[colonne_selectionnee]):
            statistiques = {
                'Moyenne': df[colonne_selectionnee].mean(),
                'Médiane': df[colonne_selectionnee].median(),
                'Écart-type': df[colonne_selectionnee].std(),
                'Variance': df[colonne_selectionnee].var(),
                'Minimum': df[colonne_selectionnee].min(),
                'Maximum': df[colonne_selectionnee].max(),
            }
        else:
            statistiques['error'] = "La colonne sélectionnée ne contient pas de données numériques."

    context = {
        'data_preview': data_preview,  # Aperçu des données
        'error_message': error_message,
        'success_message': success_message,
        'colonnes_disponibles': colonnes_disponibles,
        'colonne_selectionnee': colonne_selectionnee,
        'statistiques': statistiques,
    }

    return render(request, 'base.html', context)

# Vue pour télécharger un fichier Excel depuis le serveur
def telecharger_excel(request):
    form = FileUploadForm()
    error_message = None
    df_html = None
    column_names = []

    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)

        if form.is_valid():
            fichier = form.cleaned_data['file']
            if fichier.name.endswith(('.xls', '.xlsx')):
                try:
                    # Lecture du fichier Excel
                    df = pd.read_excel(fichier)
                    df_html = df.to_html(classes='table table-bordered', index=False)
                    column_names = df.columns.tolist()

                    # Sauvegarde des données pour session ou traitement ultérieur
                    request.session['df_json'] = df.to_json()
                except Exception as e:
                    error_message = f"Erreur lors de la lecture du fichier : {str(e)}"
            else:
                error_message = "Seuls les fichiers Excel (.xls, .xlsx) sont autorisés."
        else:
            error_message = "Veuillez sélectionner un fichier valide."

    context = {
        'form': form,
        'df_html': df_html,
        'column_names': column_names,
        'error_message': error_message,
    }

    return render(request, 'excel.html', context)


def visualiser_par_donnee(request):
    # Vérifier si les données sont disponibles dans la session
    df_json = request.session.get('df_json', None)
    if not df_json:
        return render(request, 'visualiser_par_donnee.html', {
            'error_message': "Aucun fichier n'a été téléchargé. Veuillez d'abord télécharger un fichier Excel.",
        })

    # Charger les données depuis la session
    try:
        df = pd.read_json(df_json)
    except Exception as e:
        return render(request, 'visualiser_par_donnee.html', {
            'error_message': f"Erreur lors du chargement des données : {str(e)}",
        })

    # Vérifier si le DataFrame est vide
    if df.empty:
        return render(request, 'visualiser_par_donnee.html', {
            'error_message': "Le fichier ne contient aucune donnée.",
        })

    # Liste des colonnes disponibles
    colonnes_disponibles = df.columns.tolist()

    # Lecture des paramètres de filtre
    colonne_filtrée = request.GET.get('colonne', '')
    valeur_filtrée = request.GET.get('valeur', '')
    type_chart = request.GET.get('type_chart', 'Barplot')
    col1 = request.GET.get('col1', '')
    col2 = request.GET.get('col2', '')

    # Appliquer un filtre si nécessaire
    if colonne_filtrée and colonne_filtrée in df.columns and valeur_filtrée:
        df = df[df[colonne_filtrée].astype(str) == valeur_filtrée]

    # Génération du graphique
    graph_json = None
    if not df.empty and type_chart:
        if type_chart == 'Barplot' and col1 and col2:
            fig = px.bar(df, x=col1, y=col2)
            fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Bar Plot')
            graph_json = fig.to_json()

        elif type_chart == 'histogram' and col1:
            fig = px.histogram(df, x=col1)
            fig.update_layout(xaxis_title=col1, yaxis_title='Count', title='Histogram', barmode='overlay', bargap=0.1)
            graph_json = fig.to_json()

        elif type_chart == 'piechart' and col1:
            value_counts = df[col1].value_counts().reset_index()
            value_counts.columns = [col1, 'Count']
            fig = px.pie(value_counts, values='Count', names=col1, title='Pie Chart')
            graph_json = fig.to_json()

        elif type_chart == 'scatterplot' and col1 and col2:
            fig = px.scatter(df, x=col1, y=col2)
            fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Scatter Plot')
            graph_json = fig.to_json()

        elif type_chart == 'heatmap':
            df_encoded = df.copy()
            for column in df_encoded.columns:
                if df_encoded[column].dtype == 'object':
                    df_encoded[column], _ = pd.factorize(df_encoded[column])
            fig = px.imshow(df_encoded.corr(), color_continuous_scale='Viridis')
            fig.update_layout(title='Heatmap')
            graph_json = fig.to_json()

        elif type_chart == 'lineplot' and col1 and col2:
            fig = px.line(df, x=col1, y=col2, markers=True)
            fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Line Plot')
            graph_json = fig.to_json()

        elif type_chart == 'boxplot' and col1:
            fig = px.box(df, x=col1)
            fig.update_layout(title='Box Plot')
            graph_json = fig.to_json()

        elif type_chart == 'violinplot' and col1:
            fig = px.violin(df, y=col1, box=True)
            fig.update_layout(yaxis_title=col1, title='Violin Plot')
            graph_json = fig.to_json()

      
    # Préparer le contexte pour le template
    context = {
        'colonnes_disponibles': colonnes_disponibles,
        'data': df.to_dict(orient='records'),
        'colonne_filtrée': colonne_filtrée,
        'valeur_filtrée': valeur_filtrée,
        'type_chart': type_chart,
        'graph_json': graph_json,
        'col1': col1,
        'col2': col2
    }

    return render(request, 'visualiser_par_donnee.html', context)



def lister_fichiers_disponibles(media_root):
    """
    Liste les fichiers disponibles dans le répertoire MEDIA_ROOT.
    """
    return [f for f in os.listdir(media_root) if os.path.isfile(os.path.join(media_root, f))]


def statistiques_par_colonne(request, filename):
    file_path = os.path.join(MEDIA_ROOT, filename)
    
    if not os.path.exists(file_path):
        return render(request, 'base.html', {'error_message': "Aucun fichier n'a été téléchargé."})

    # Charger le fichier Excel
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        return render(request, 'base.html', {'error_message': f"Erreur lors du chargement du fichier : {str(e)}"})

    if df.empty:
        return render(request, 'base.html', {'error_message': "Le fichier ne contient aucune donnée."})

    colonnes_disponibles = df.columns.tolist()
    colonne_selectionnee = request.GET.get('colonne', '')

    statistiques = {}
    if colonne_selectionnee and colonne_selectionnee in df.columns:
        # Vérifiez si la colonne contient des données numériques
        if pd.api.types.is_numeric_dtype(df[colonne_selectionnee]):
            statistiques = {
                'Moyenne': df[colonne_selectionnee].mean(),
                'Médiane': df[colonne_selectionnee].median(),
                'Écart-type': df[colonne_selectionnee].std(),
                'Variance': df[colonne_selectionnee].var(),
                'Minimum': df[colonne_selectionnee].min(),
                'Maximum': df[colonne_selectionnee].max(),
            }
        else:
            statistiques['error'] = "La colonne sélectionnée ne contient pas de données numériques."

    context = {
        'colonnes_disponibles': colonnes_disponibles,
        'colonne_selectionnee': colonne_selectionnee,
        'statistiques': statistiques,
    }

    return render(request, 'statistiques_par_colonne.html', context)






################  LOI  #####################################################################################def calculer_loi(request):
# Fonctions pour les lois de probabilité
import scipy.stats as stats
import math

def bernoullii(p):
    return p

def binomialee(n, p, k):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def uniformee(a, b):
    return f"Uniforme dans [{a}, {b}]"

def poissonn(lambda_, k):
    return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)

def exponentiellee(lambda_, x):
    return lambda_ * math.exp(-lambda_ * x)

# Normal distribution
def normalee(moyenne, ecart_type, x):
    return stats.norm.pdf(x, loc=moyenne, scale=ecart_type)

# Main calculation view

def calculer_loi(request):
    result = None
    exact_value = None
    errors = None
    loi = None
    plot_data = None  # Variable pour stocker le graphique encodé

    if request.method == 'POST':
        loi = request.POST.get('loi')  # Type de loi choisie
        n = request.POST.get('n')  # Taille de l'échantillon (entier)

        # Paramètres pour chaque loi
        param1 = request.POST.get('probabilite')  # Bernoulli
        param2 = request.POST.get('n_binomiale')  # Binomiale (nombre d'essais)
        param3 = request.POST.get('p_binomiale')  # Binomiale (probabilité de succès)
        param4 = request.POST.get('a_uniforme')  # Uniforme (borne inférieure)
        param5 = request.POST.get('b_uniforme')  # Uniforme (borne supérieure)
        param6 = request.POST.get('lambda_poisson')  # Poisson (taux moyen)
        param7 = request.POST.get('lambda_exponentielle')  # Exponentielle (taux)
        param8 = request.POST.get('moyenne')  # Normale (moyenne)
        param9 = request.POST.get('ecart_type')  # Normale (écart-type)
        param10 = request.POST.get('k')  # Variable pour les lois binomiale et Poisson

        try:
            # Validation et conversion des paramètres
            if n:
                n = int(n)
            if param1:
                param1 = float(param1)
            if param2:
                param2 = int(param2)
            if param3:
                param3 = float(param3)
            if param4:
                param4 = float(param4)
            if param5:
                param5 = float(param5)
            if param6:
                param6 = float(param6)
            if param7:
                param7 = float(param7)
            if param8:
                param8 = float(param8)
            if param9:
                param9 = float(param9)
            if param10:
                param10 = int(param10)
        except ValueError:
            errors = "Erreur : Veuillez entrer des valeurs numériques valides pour les champs requis."
            return render(request, 'lois_de_probabilite.html', {'result': result, 'errors': errors, 'loi': loi})

        try:
            if loi == 'bernoulli' and param1 is not None:
                exact_value = f"Probabilité de succès : {bernoullii(param1)}"
                result = np.random.binomial(1, param1, n).tolist() if n else None
                data_bern = bernoulli.rvs(size=1000, param1=param1)
                # Générer le graphique de Bernoulli
                sns.set(style="whitegrid")
                plt.figure(figsize=(6, 4))
                ax = sns.histplot(data_bern, kde=True, stat='probability')
                ax.set(xlabel='Bernoulli', ylabel='Probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

            elif loi == 'binomiale' and param2 is not None and param3 is not None and param10 is not None:
                exact_value = f"Probabilité exacte pour k={param10} : {binomialee(param2, param3, param10)}"
                result = np.random.binomial(2, param3, n).tolist() if n else None
                data_bin = uniform.rvs(n=n,param1=param1, loc=0, size=1000)

                # Générer le graphique de Binomiale
                sns.set(style="whitegrid")
                plt.figure(figsize=(6, 4))
                ax = sns.histplot(data_bin, kde=True, stat='probability')
                ax.set(xlabel='Binomiale', ylabel='Probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

            elif loi == 'uniforme' and param4 is not None and param5 is not None:
                exact_value = uniformee(param4, param5)
                result = np.random.uniform(param4, param5, n).tolist() if n else None
                data_unif = uniform.rvs(loc=param4, scale=param5-param4, size=1000)
                # Générer le graphique uniforme
                sns.set(style="whitegrid")
                plt.figure(figsize=(6, 4))
                ax = sns.histplot(data_unif, kde=True, stat='probability')
                ax.set(xlabel='Uniforme', ylabel='Probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

            elif loi == 'poisson' and param6 is not None and param10 is not None:
                exact_value = f"Probabilité exacte pour k={param10} : {poissonn(param6, param10)}"
                result = np.random.poisson(param6, n).tolist() if n else None
                data_poisson = poisson.rvs(mu=param6, size=1000)

                # Générer le graphique de Poisson
                
                sns.set(style="whitegrid")
                plt.figure(figsize=(6, 4))
                ax = sns.histplot(data_poisson, kde=True, stat='probability')
                ax.set(xlabel='Poisson', ylabel='Probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

            elif loi == 'exponentielle' and param7 is not None and param10 is not None:
                exact_value = f"Probabilité P(X≤{param10}) : {exponentiellee(param7, param10)}"
                result = np.random.expon.rvs(1 / param7, size=1000).tolist() if n else None
                data_exponentielle = expon.rvs(scale=param7, size=1000)

                # Générer le graphique exponentielle
                sns.set(style="whitegrid")
                plt.figure(figsize=(6, 4))
                sns.kdeplot(data_exponentielle, fill=True)
                plt.title('Distribution Exponentielle')
                plt.xlabel('Valeur')
                plt.ylabel('Densité de probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

            elif loi == 'normale' and param8 is not None and param9 is not None and param10 is not None:
                exact_value = f"Densité en x={param10} : {normalee(param8, param9, param10)}"
                result = np.random.normal(param8, param9, n).tolist() if n else None

                # Générer le graphique normal
                sns.set(style="whitegrid")
                plt.figure(figsize=(10, 6))
                x_vals = np.linspace(param8 - 3*param9, param8 + 3*param9, 1000)
                y_vals = stats.norm.pdf(x_vals, param8, param9)
                plt.plot(x_vals, y_vals, color="Slateblue", alpha=0.6)
                plt.fill_between(x_vals, y_vals, color="skyblue", alpha=0.4)
                plt.title('Distribution Normale Continue')
                plt.xlabel('Valeur')
                plt.ylabel('Densité de probabilité')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plot_data = base64.b64encode(image_png).decode('utf-8')

        except Exception as e:
            errors = f"Erreur lors du calcul : {str(e)}"

    return render(request, 'lois_de_probabilite.html', {
        'result': result,
        'exact_value': exact_value,
        'errors': errors,
        'loi': loi,
        'plot_data': plot_data,  # Affichage du graphique
    })

###########  Tests  ###############################################################################

def calculate_z_test(field, zTestmi, sigma, n, significance):
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    zTestmi = float(zTestmi.replace(',', '.'))
    z_stat = (field - zTestmi) / (sigma / np.sqrt(n))
    p_value_two_sided = norm.sf(abs(z_stat)) * 2
    hypothesis_result_two_sided = "On rejette l'hypothèse." if p_value_two_sided < significance else "On accepte l'hypothèse."
    return {
        'z_statistic': z_stat,
        'p_value_two_sided': p_value_two_sided,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
    }

def calculate_t_test2(field, tTestmi, sigma, n, significance):
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    tTestmi = float(tTestmi.replace(',', '.'))
    t_statistic = (field - tTestmi) / (sigma / np.sqrt(n))
    p_value_two_sided = t.sf(abs(t_statistic), df=n-1) * 2
    hypothesis_result_two_sided = "On rejette l'hypothèse." if p_value_two_sided < significance else "On accepte l'hypothèse."
    return {
        't_statistic': t_statistic,
        'p_value_two_sided': p_value_two_sided,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
    }

def test_traitement(request):
    if request.method == 'GET':
        test_type = request.GET.get('testType')
        if test_type:
            significance = float(request.GET.get('significance', 0.05))
            if test_type == 'zTest':
                field = request.GET.get('zTestField')
                sigma = request.GET.get('zTestSigma')
                n = request.GET.get('zTestN')
                zTestmi = request.GET.get('zTestmi')
                z_test_results = calculate_z_test(field, zTestmi, sigma, n, significance)
                return JsonResponse({
                    'z_statistic': z_test_results['z_statistic'],
                    'p_value_two_sided': z_test_results['p_value_two_sided'],
                    'hypothesis_result_two_sided': z_test_results['hypothesis_result_two_sided'],
                    'formule': "Z = (X̄ - μ) / (σ/ √n)"
                })
            elif test_type == 'tTest2':
                field = request.GET.get('tTestField2')
                sigma = request.GET.get('tTestSigma2')
                n = request.GET.get('testTestN2')
                tTestmi = request.GET.get('tTestmi2')
                t_test_results = calculate_t_test2(field, tTestmi, sigma, n, significance)
                return JsonResponse({
                    't_statistic': t_test_results['t_statistic'],
                    'p_value_two_sided': t_test_results['p_value_two_sided'],
                    'hypothesis_result_two_sided': t_test_results['hypothesis_result_two_sided'],
                    'formule': "Z = (X̄ - μ) / (σ/ √n)"
                })
            else:
                return JsonResponse({'error': 'Invalid test type'})
        else:
            return JsonResponse({'error': 'Invalid test type'})
    else:
        return JsonResponse({'error': 'Invalid request method'})

def inferentielles(request):
    return render(request, 'test_inferentiel.html')