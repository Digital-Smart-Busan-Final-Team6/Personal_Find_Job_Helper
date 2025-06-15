# accounts/views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib import messages
from .forms import LoginForm, RegisterForm, ProfileUpdateForm
from .models import UserProfile

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
            user = form.save()
            login(request, user)
            messages.success(request, '회원가입이 완료되었습니다. 환영합니다!')
            return redirect('home')
        else:
            messages.error(request, '회원가입에 실패했습니다. 입력 내용을 확인해주세요.')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def mypage_view(request):
    user = request.user
    profile = user.profile

    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "프로필이 성공적으로 수정되었습니다.")
            return redirect('mypage')
    else:
        form = ProfileUpdateForm(instance=profile)

    return render(request, 'accounts/mypage.html', {
        'form': form,
        'username': user.username,
        'email': user.email,
    })
