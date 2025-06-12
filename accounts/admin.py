# accounts/admin.py

from django.contrib import admin
from .models import UserProfile

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'education', 'job', 'location', 'skills')  # 바뀐 필드명으로 수정

admin.site.register(UserProfile, UserProfileAdmin)