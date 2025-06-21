from django import forms
from .models import Resume

class ResumeForm(forms.ModelForm):
    class Meta:
        model = Resume
        fields = [
            'title', 
            'education_level', 'university', 'major', 'gpa', 
            'experience_years', # 'work_experience' 대신 'experience_years' 추가
            'job', 'location', 'skills',
            'experience', 'certifications'
        ]
        
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 백엔드 개발자 이력서'}),
            
            # --- 학력 위젯 ---
            'education_level': forms.Select(attrs={'class': 'form-select'}),
            'university': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: OO대학교'}),
            'major': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 컴퓨터공학과'}),
            'gpa': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 4.2'}),
            
            # --- 경력(년) 위젯 ---
            'experience_years': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '숫자만 입력'}),
            
            # --- 나머지 위젯 ---
            'skills': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '보유 기술을 입력하고 Enter를 누르세요' # placeholder 텍스트 변경
            }),
            'job': forms.HiddenInput(),
            'location': forms.HiddenInput(),
            'experience': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 10, 
                'placeholder': '대외활동, 인턴, 프로젝트, 교육 이수 내용 등을 자유롭게 기재해주세요.\n\n예시)\n\n[프로젝트]\n- 개인 포트폴리오 웹사이트 개발 (2024.01 ~ 2024.03)\n  - 주요 기술: Python, Django, JavaScript'
            }),
            'certifications': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 8,
                'placeholder': '취득한 자격증, 수상 내역, 어학 능력 등을 자유롭게 기재해주세요.\n\n예시)\n\n[자격증]\n- 정보처리기사 (2023.05)\n\n[어학]\n- 영어 (비즈니스 회화 가능, TOEIC 950점)'
            }),
        }