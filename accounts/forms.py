# accounts/forms.py

from django import forms
from .models import Resume

class ResumeForm(forms.ModelForm):
    # 희망 직무와 근무지는 템플릿에서 직접 처리할 것이므로, 폼에서는 숨겨진 필드로 만듭니다.
    # 이렇게 하면 뷰에서 데이터를 처리하기 용이해집니다.
    job = forms.CharField(widget=forms.HiddenInput(), required=False)
    location = forms.CharField(widget=forms.HiddenInput(), required=False)

    class Meta:
        model = Resume
        # ★★★ 여기가 핵심! models.py와 필드 목록을 일치시킵니다. ★★★
        fields = [
            'title', 
            'university', 
            'major', 
            'education_status', 
            'gpa', 
            'graduation_status', 
            'job', 
            'location', 
            'skills'
        ]
        # --------------------------------------------------------
        
        # 위젯 설정은 그대로 유지해도 좋습니다. 템플릿에 스타일을 적용해줍니다.
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '이력서 제목 (예: 백엔드 개발자 지원)'}),
            'university': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 서울대학교'}),
            'major': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 컴퓨터공학과'}),
            'education_status': forms.Select(attrs={'class': 'form-select'}, choices=[
                ('', '선택'), ('중졸', '중졸'), ('고졸', '고졸'), ('학사', '학사'), ('석사', '석사'), ('박사', '박사')
            ]),
            'gpa': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '예: 4.2'}),
            'graduation_status': forms.Select(attrs={'class': 'form-select'}, choices=[
                ('', '선택'), ('졸업', '졸업'), ('졸업예정자', '졸업예정자')
            ]),
             'skills': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 10, 
                'placeholder': """보유한 역량, 스킬을 입력하세요
예 1) Java, Spring, Notion, Jira, Git
예 2) PPT, Figma, Slack, 의사소통, 발표, 기능정의서 작성
예 3) SQL, Google Analytics, Tableau, 데이터 분석, 데이터 시각화"""
            }),
        }