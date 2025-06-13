# accounts/models.py
from django.db import models

class Resume(models.Model):
    # 이력서 제목
    title = models.CharField(max_length=200, default="새 이력서")
    
    # --- 학력 관련 필드 (세분화) ---
    university = models.CharField('대학명', max_length=100, blank=True, null=True)
    major = models.CharField('전공', max_length=100, blank=True, null=True)
    education_status = models.CharField('학적상태', max_length=50, blank=True, null=True) # 예: 학사, 석사
    gpa = models.CharField('학점', max_length=10, blank=True, null=True) # 예: 4.2 / 4.5
    graduation_status = models.CharField('졸업상태', max_length=50, blank=True, null=True) # 예: 졸업, 졸업예정
    
    # --- 기존 필드 ---
    # 희망 직무 (여러 개 선택 가능하도록 TextField로 변경)
    job = models.TextField('희망 직무', blank=True, null=True)
    # 희망 근무지 (여러 개 선택 가능하도록 TextField로 변경)
    location = models.TextField('희망 근무지', blank=True, null=True)
    skills = models.TextField('보유 기술', blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} (ID: {self.id})"