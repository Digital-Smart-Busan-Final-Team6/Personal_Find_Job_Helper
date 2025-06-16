from django.db import models
from django.contrib.auth.models import User

class Resume(models.Model):
    user = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL,
        null=True, 
        blank=True
    )
    title = models.CharField(max_length=200, verbose_name="이력서 제목")
    
    # --- 학력 정보 ---
    EDUCATION_LEVEL_CHOICES = [
        ('고졸', '고등학교 졸업'),
        ('초대졸', '전문대 졸업'),
        ('대졸', '대학교 졸업 (학사)'),
        ('석사', '대학원 졸업 (석사)'),
        ('박사', '대학원 졸업 (박사)'),
    ]
    education_level = models.CharField(
        max_length=10, 
        choices=EDUCATION_LEVEL_CHOICES, 
        blank=True, 
        verbose_name="최종 학력"
    )
    university = models.CharField(max_length=100, blank=True, verbose_name="학교명")
    major = models.CharField(max_length=100, blank=True, verbose_name="전공")
    gpa = models.CharField(max_length=10, blank=True, verbose_name="학점")

    experience_years = models.IntegerField(null=True, blank=True, verbose_name="경력(년)")
    
    # --- 희망 조건 ---
    job = models.CharField(max_length=255, verbose_name="희망 직무")
    location = models.CharField(max_length=255, verbose_name="희망 근무지")

    skills = models.TextField(blank=True, verbose_name="보유 기술")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.user.username})"