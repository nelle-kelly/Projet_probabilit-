# app_patient/templatetags/custom_filters.py
import base64
from django import template

register = template.Library()

@register.filter(name='base64encode')
def base64encode(value):
    """Convertit une image en base64 pour l'afficher dans un tag img."""
    with open(value, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@register.filter
def get_attribute(obj, attr):
    """
    Retourne l'attribut ou la clé dynamique d'un objet ou dictionnaire.
    """
    try:
        # Si c'est un dictionnaire
        if isinstance(obj, dict):
            return obj.get(attr, "")
        # Si c'est un objet avec des attributs
        return getattr(obj, attr, "")
    except Exception:
        return ""