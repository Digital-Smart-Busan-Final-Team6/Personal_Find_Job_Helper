from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from accounts.forms import LoginForm, RegisterForm

def home(request):
    return render(request, 'home.html')

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, '아이디나 비밀번호가 올바르지 않습니다.')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password1 = form.cleaned_data['password1']
            password2 = form.cleaned_data['password2']

            if User.objects.filter(username=username).exists():
                messages.error(request, '이미 존재하는 사용자입니다.')
            elif password1 != password2:
                messages.error(request, '비밀번호가 일치하지 않습니다.')
            else:
                User.objects.create_user(username=username, password=password1)
                messages.success(request, '회원가입이 완료되었습니다.')
                return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

def mypage_view(request):
    return render(request, 'mypage.html')

def logout_view(request):
    logout(request)
    return redirect('home')
