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

chain = None

# --- [View 1] 챗봇이 있는 메인 페이지 ---
def home(request):
    """
    프로젝트의 메인 페이지로, 챗봇 기능을 담당합니다.
    """
    context = {
        'current_page': 'home',
        'switch_url_name': 'resume_list',
    }
    return render(request, 'home.html', context)


# --- [View 2] 이력서 '목록'을 보여주는 페이지 (Read) ---
def resume_list_view(request):
    """
    데이터베이스에 저장된 모든 이력서를 가져와 목록 페이지에 보여줍니다.
    """
    resumes = Resume.objects.all().order_by('-updated_at')
    context = {
        'resumes': resumes,
        'current_page': 'resume',
        'switch_url_name': 'home',
    }
    return render(request, 'resume_list.html', context)


# --- [View 3] 새 이력서를 '추가'하는 페이지 (Create) ---
def resume_add_view(request):
    """
    새로운 이력서를 작성하고 저장하는 폼 페이지입니다.
    """
    if request.method == 'POST':
        form = ResumeForm(request.POST)
        job_list = request.POST.getlist('job')
        location_list = request.POST.getlist('location')
        
        if form.is_valid():
            new_resume = form.save(commit=False)
            new_resume.job = ",".join(job_list)
            new_resume.location = ",".join(location_list)
            new_resume.save()
            return redirect('resume_list')
    else:
        form = ResumeForm()
    
    context = {
        'form': form,
        'selected_jobs': [],
        'selected_locations': [],
        'is_edit_mode': False,
    }
    return render(request, 'resume_form.html', context)


# --- [View 4] 특정 이력서를 '수정'하는 페이지 (Update) ---
def resume_edit_view(request, resume_id):
    """
    기존에 작성된 특정 이력서의 내용을 수정하는 폼 페이지입니다.
    """
    resume = get_object_or_404(Resume, id=resume_id)

    if request.method == 'POST':
        form = ResumeForm(request.POST, instance=resume)
        job_list = request.POST.getlist('job')
        location_list = request.POST.getlist('location')
        
        if form.is_valid():
            updated_resume = form.save(commit=False)
            updated_resume.job = ",".join(job_list)
            updated_resume.location = ",".join(location_list)
            updated_resume.save()
            return redirect('resume_list')
    else:
        form = ResumeForm(instance=resume)

    context = {
        'form': form,
        'resume': resume,
        'selected_jobs': resume.job.split(',') if resume.job else [],
        'selected_locations': resume.location.split(',') if resume.location else [],
        'is_edit_mode': True,
    }
    return render(request, 'resume_form.html', context)


# --- [View 5] 특정 이력서를 '삭제'하는 기능 (Delete) ---
def resume_delete_view(request, resume_id):
    """
    특정 이력서를 데이터베이스에서 삭제하는 기능을 담당합니다.
    """
    resume = get_object_or_404(Resume, id=resume_id)
    if request.method == 'POST':
        resume.delete()
        return redirect('resume_list')
    
    return redirect('resume_list')


# --- [View 6] 모든 이력서를 JSON 파일로 내보내는 기능 ---
def export_resumes_to_json(request):
    """
    데이터베이스에 있는 모든 이력서 정보를 가져와 JSON 형식으로 응답합니다.
    """
    resumes = Resume.objects.all()
    data_list = []
    for resume in resumes:
        data = {
            'id': resume.id,
            'title': resume.title,
            'university': resume.university,
            'major': resume.major,
            'education_status': resume.education_status,
            'gpa': resume.gpa,
            'graduation_status': resume.graduation_status,
            'job_interests': resume.job.split(',') if resume.job else [],
            'location_interests': resume.location.split(',') if resume.location else [],
            'skills': resume.skills,
            'created_at': resume.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': resume.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
        }
        data_list.append(data)
    return JsonResponse(data_list, safe=False, json_dumps_params={'ensure_ascii': False, 'indent': 2})


# --- [View 7] 챗봇 API 엔드포인트 ---
@csrf_exempt
def chat_api(request):
    """
    챗봇의 질문에 대한 응답을 처리하는 API입니다.
    """
    global chain
    
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
            
            result = chain.invoke(user_input)
            return JsonResponse({'response': result})
        except Exception as e:
            return JsonResponse({'response': f'에러 발생: {str(e)}'})