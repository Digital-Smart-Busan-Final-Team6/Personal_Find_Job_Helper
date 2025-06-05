from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

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

class ProfileUpdateForm(forms.Form):
    career = forms.CharField(label='경력사항', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    certificates = forms.CharField(label='자격증', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    awards = forms.CharField(label='수상 내역', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    external_activities = forms.CharField(label='대외 활동', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    skills = forms.CharField(label='기술 스택', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)


    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'career', 'certifications', 'awards', 'activities', 'skills']

    def clean(self):
        cleaned_data = super().clean()
        pw1 = cleaned_data.get('password1')
        pw2 = cleaned_data.get('password2')
        if pw1 and pw2 and pw1 != pw2:
            raise forms.ValidationError("비밀번호가 일치하지 않습니다.")
