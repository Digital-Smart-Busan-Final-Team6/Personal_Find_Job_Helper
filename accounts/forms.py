# accounts/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile

class LoginForm(forms.Form):
    username = forms.CharField(label='ID', max_length=150, widget=forms.TextInput(attrs={'class': 'form-input'}))
    password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class': 'form-input'}))

class RegisterForm(UserCreationForm):
    email = forms.EmailField(label='Email', required=True, widget=forms.EmailInput(attrs={'class': 'form-input'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        return super().save(commit=commit)


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['education', 'job', 'location', 'skills']  # 새 필드명
