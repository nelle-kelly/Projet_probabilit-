# app_patient/templatetags/custom_filters.py
import base64
from django import template

register = template.Library()

@register.filter(name='base64encode')
def base64encode(value):
    """Convertit une image en base64 pour l'afficher dans un tag img."""
    with open(value, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
