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

    career = forms.CharField(label='경력사항', widget=forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}), required=False)
    certifications = forms.CharField(label='자격증', widget=forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}), required=False)
    awards = forms.CharField(label='수상 내역', widget=forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}), required=False)
    activities = forms.CharField(label='대외 활동', widget=forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}), required=False)
    skills = forms.CharField(label='기술 스택', widget=forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}), required=False)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=commit)
        if commit:
            UserProfile.objects.create(
                user=user,
                career=self.cleaned_data['career'],
                certifications=self.cleaned_data['certifications'],
                awards=self.cleaned_data['awards'],
                activities=self.cleaned_data['activities'],
                skills=self.cleaned_data['skills']
            )
        return user

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['career', 'certifications', 'awards', 'activities', 'skills']
        widgets = {
            'career': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}),
            'certifications': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}),
            'awards': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}),
            'activities': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}),
            'skills': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3}),
        }
