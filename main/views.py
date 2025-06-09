from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages

from accounts.forms import LoginForm, RegisterForm, ProfileUpdateForm
from accounts.models import UserProfile

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .chain import chain # chain.py에서 모듈화 해둔 chain

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
            user = form.save()  # RegisterForm이 User + UserProfile 생성
            messages.success(request, '회원가입이 완료되었습니다.')
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def mypage_view(request):
    user = request.user
    profile = user.profile  # UserProfile 객체

    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, '정보가 성공적으로 수정되었습니다.')
            return redirect('mypage')
    else:
        form = ProfileUpdateForm(instance=profile)

    return render(request, 'accounts/mypage.html', {
        'form': form,
        'username': user.username,
        'email': user.email,
    })

def logout_view(request):
    logout(request)
    messages.success(request, '로그아웃되었습니다.')
    return redirect('home')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '').strip()
            if not user_input:
                return JsonResponse({'response': '질문을 입력해주세요.'})
            
            # LangChain 처리
            result = chain.invoke(user_input)
            return JsonResponse({'response': result})
        except Exception as e:
            return JsonResponse({'response': f'에러 발생: {str(e)}'})