# accounts/models.py
from django.db import models
from django.contrib.auth.models import User
from django import forms

# 유저 프로필
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    career = models.TextField(blank=True)
    certifications = models.TextField(blank=True)
    awards = models.TextField(blank=True)
    activities = models.TextField(blank=True)
    skills = models.TextField(blank=True)

# 마이페이지 프로필 업데이트트
class ProfileUpdateForm(forms.Form):
    career = forms.CharField(label='경력사항', widget=forms.Textarea(attrs={'rows': 3}), required=False)
    certifications = forms.CharField(label='자격증', widget=forms.Textarea(attrs={'rows': 3}), required=False)
    awards = forms.CharField(label='수상 내역', widget=forms.Textarea(attrs={'rows': 3}), required=False)
    activities = forms.CharField(label='대외 활동', widget=forms.Textarea(attrs={'rows': 3}), required=False)
    skills = forms.CharField(label='기술 스택', widget=forms.Textarea(attrs={'rows': 3}), required=False)

    def __str__(self):
        return self.user.username
