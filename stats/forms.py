from django import forms
from django.contrib.auth.models import User

from .models import Code



class StatsInputForm(forms.ModelForm):

    class Meta:
        model = Code
        fields = ('title',)
        widgets = {
            'title': forms.fields.TextInput(attrs={'size':'70',}),
            # 'text': forms.fields.TextInput(attrs={'size':'100'})
        }
