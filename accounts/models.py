from django.db import models
from django.contrib.auth.models import User  # 유저랑 연결하기 위해

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    # ▶ 추가할 필드
    education = models.TextField(blank=True, null=True)   # 학력/학점
    job = models.CharField(max_length=100, blank=True, null=True)  # 희망 직무
    location = models.CharField(max_length=100, blank=True, null=True)  # 희망 근무지
    skills = models.TextField(blank=True, null=True)   # 보유 기술

    def __str__(self):
        return self.user.username
