from django import forms
from .models import Imager


class ImageForm(forms.ModelForm):
    class Meta:
        model = Imager
        fields = ('coord', 'image')