# accounts/models.py
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    career = models.TextField(blank=True)
    certifications = models.TextField(blank=True)
    awards = models.TextField(blank=True)
    activities = models.TextField(blank=True)
    skills = models.TextField(blank=True)

    def __str__(self):
        return self.user.username