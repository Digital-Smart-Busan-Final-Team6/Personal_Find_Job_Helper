# main/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from dotenv import load_dotenv
from Run_Pipeline.Main_Pipeline import main

from accounts.models import Resume
from accounts.forms import ResumeForm

load_dotenv()

#  1. 체인을 전역 변수로 선언만 해두고, 초기값은 None으로 설정합니다.
chain = None

def home(request):
    return render(request, 'home.html')

def mypage_view(request):
    resume, created = Resume.objects.get_or_create(pk=1, defaults={'title': '내 기본 이력서'})

    if request.method == 'POST':
        post_data = request.POST.copy()
        post_data['job'] = ','.join(request.POST.getlist('job'))
        post_data['location'] = ','.join(request.POST.getlist('location'))
        form = ResumeForm(post_data, instance=resume)
        if form.is_valid():
            form.save()
            return redirect('mypage')
    else:
        form = ResumeForm(instance=resume)

    selected_jobs = resume.job.split(',') if resume.job else []
    selected_locations = resume.location.split(',') if resume.location else []

    context = {
        'form': form,
        'resume': resume,
        'selected_jobs': selected_jobs,
        'selected_locations': selected_locations,
    }
    return render(request, 'mypage.html', context)

@csrf_exempt
def chat_api(request):
    #  2. 전역 변수 chain을 사용하겠다고 선언합니다.
    global chain
    
    #  3. chain이 아직 생성되지 않았다면 (최초 요청이라면) 그 때 생성합니다.
    if chain is None:
        print("Initializing LangChain chain for the first time...")
        chain = main(return_chain_only=True)
        print("Chain initialized.")

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '').strip()
            if not user_input:
                return JsonResponse({'response': '질문을 입력해주세요.'})
            
            # 이제 안전하게 chain을 호출할 수 있습니다.
            result = chain.invoke(user_input)
            return JsonResponse({'response': result})
        except Exception as e:
            return JsonResponse({'response': f'에러 발생: {str(e)}'})