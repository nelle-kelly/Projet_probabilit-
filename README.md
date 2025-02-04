
# Guide d'Installation et d'Exécution

## 1. Création et activation de l'environnement virtuel

Création :
 ```bash 
    python -m venv venv 
```
Activation : 
```bash 
    venv\Scripts\activate
```

## 2. Bibliotheque
Installation de toutes les bibliothèques nécessaires avec la commande suivante :
```bash
 pip install -r requirements.txt
 ```
## 3. Configuration du projet Django
### 1.  Lancez les migrations
Vous utilisez les commandes suivantes :
```bash
python manage.py makemigrations

python manage.py migrate

```




### 2. Lancez le serveur local
Démarrez le serveur pour exécuter l'application :
Utilisez la commande :
```bash
 python manage.py runserver

 ```

puis ouvrez votre navigateur et accédez à l'application via **http://127.0.0.1:8000**.


## Fonctionnalités principales

### 1. **Téléchargement de fichiers excel et calculs de statistiques**
- Importez un fichiers Excel.
<br />
<img src="patient/app_patient/picture/excel.png"></img>

- Visualisation du fichier excel
<br />
<img src="patient/app_patient/picture/upload.png" >
<br />
<img src="patient/app_patient/picture/image1.png" >
<br />
<img src="patient/app_patient/picture/image2.png" >

### 2. **Loi de probabilités**
- Visualisation et simulation des lois suivantes :
  - **Bernoulli**
  - **Binomiale**
  - **Uniforme**
  - **Poisson**
  - **Exponentielle**
  - **Normale continue**
  <br />
  <img src="patient/app_patient/picture/loi1.png">
  <br />
  <img src="patient/app_patient/picture/loi2.png">

### 3. **Tests d'hypothèses**
- **Z-Test** (échantillons de grande taille, n > 30).
- **T-Test** (échantillons de petite taille, n < 30).
- Affichage interactif des résultats avec calculs des statistiques..


## Bibliothèques utilisées

L'application repose sur les bibliothèques suivante:

- **Django** : Framework backend pour le développement web.
- **Pandas** : Manipulation et analyse de données.
- **NumPy** : Calculs numériques avancés.
- **Plotly** : Visualisations interactives.
- **Matplotlib** et **Seaborn** : Visualisation des données.
- **SciPy** : Tests statistiques et distributions probabilistes.


