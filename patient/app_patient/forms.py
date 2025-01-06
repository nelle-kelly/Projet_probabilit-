from django import forms

class FileUploadForm(forms.Form):
    csv_file = forms.FileField(label='Fichier CSV', required=False)
    excel_file = forms.FileField(label='Fichier Excel', required=False)
    file = forms.FileField()