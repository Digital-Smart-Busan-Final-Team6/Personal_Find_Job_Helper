from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(label='ID', max_length=150, widget=forms.TextInput(attrs={'class':'form-input'}))
    password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class':'form-input'}))

class RegisterForm(forms.Form):
    username = forms.CharField(label='ID', max_length=150, widget=forms.TextInput(attrs={'class':'form-input'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class':'form-input'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(attrs={'class':'form-input'}))

    career = forms.CharField(label='경력사항', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    certificates = forms.CharField(label='자격증', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    awards = forms.CharField(label='수상 내역', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    external_activities = forms.CharField(label='대외 활동', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)
    skills = forms.CharField(label='기술 스택', widget=forms.Textarea(attrs={'class':'form-textarea', 'rows':3}), required=False)

    def clean(self):
        cleaned_data = super().clean()
        pw1 = cleaned_data.get('password1')
        pw2 = cleaned_data.get('password2')

        if pw1 and pw2 and pw1 != pw2:
            raise forms.ValidationError("비밀번호가 일치하지 않습니다.")
