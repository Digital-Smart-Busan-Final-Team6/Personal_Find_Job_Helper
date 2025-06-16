from django import forms
from .models import Resume # 실제 Resume 모델이 있는 앱 이름으로 수정

class ResumeForm(forms.ModelForm):
    class Meta:
        model = Resume
        # 1. models.py와 동일하게 필드 목록을 수정합니다.
        fields = [
            'title', 
            'education_level', 'university', 'major', 'gpa', 
            'experience_years', # 'work_experience' 대신 'experience_years' 추가
            'job', 'location', 'skills'
        ]
        
        # 2. 위젯 설정도 새로운 필드에 맞게 수정합니다.
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 백엔드 개발자 이력서'}),
            
            # --- 학력 위젯 ---
            'education_level': forms.Select(attrs={'class': 'form-select'}),
            'university': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: OO대학교'}),
            'major': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 컴퓨터공학과'}),
            'gpa': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 4.2'}),
            
            # --- 경력(년) 위젯 추가 ---
            'experience_years': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '숫자만 입력'}),
            
            # --- 나머지 위젯 ---
            'skills': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '보유 기술을 입력하고 Enter를 누르세요' # placeholder 텍스트 변경
            }),
            'job': forms.HiddenInput(),
            'location': forms.HiddenInput(),
        }