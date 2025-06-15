# main/views.py

import time  # Agent 호출 시뮬레이션을 위한 time 모듈
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from accounts.models import Resume
from accounts.forms import ResumeForm

# Agent 관련 import (실제 구현 시 주석 해제)
# from Run_Pipeline.Agent_Factory import get_agent_chain
# from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks


# --- [View 1] 챗봇이 있는 메인 페이지 ---
def home(request):
    context = {
        'current_page': 'home',
        'switch_url_name': 'resume_list',
    }
    return render(request, 'home.html', context)


# --- [View 2] 이력서 '목록'을 보여주는 페이지 (Read) ---
def resume_list_view(request):
    resumes = Resume.objects.all().order_by('-updated_at')
    context = {
        'resumes': resumes,
        'current_page': 'resume',
        'switch_url_name': 'home',
    }
    return render(request, 'resume_list.html', context)


# --- [View 3] 새 이력서를 '추가'하는 페이지 (Create) ---
def resume_add_view(request):
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
    resume = get_object_or_404(Resume, id=resume_id)
    if request.method == 'POST':
        resume.delete()
        return redirect('resume_list')
    return redirect('resume_list')


# --- [View 6] 모든 이력서를 JSON 파일로 내보내는 기능 ---
def export_resumes_to_json(request):
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


# --- [View 7] 공고 검색 리포트 페이지 ---
def job_search_report_page(request):
    resume_ids_str = request.GET.get('resumes', '')
    if not resume_ids_str:
        return redirect('resume_list')

    resume_ids = resume_ids_str.split(',')
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    context = {
        'selected_resumes': selected_resumes,
        'resume_ids_str': resume_ids_str,
        'current_page': 'resume',
        'switch_url_name': 'home',
    }
    return render(request, 'job_search_report.html', context)


# --- [View 8] 챗봇 API 엔드포인트 ---
@csrf_exempt
def chat_api(request):
    """
    챗봇의 질문에 대한 응답을 실시간 스트리밍으로 처리하는 API입니다.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()

        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        if not message:
            return StreamingHttpResponse("data: {\"error\": \"질문을 입력해주세요.\"}\n\n", content_type="text/event-stream")

        agent = get_agent_chain()

        # 스트리밍 응답을 위한 제너레이터 함수를 정의합니다.
        def stream_response_generator():
            # 이 리스트는 콜백 함수가 토큰을 담아두는 임시 저장소 역할을 합니다.
            tokens_to_yield = []
            
            # 콜백 함수 정의: Agent가 생성하는 최종 답변(result)을 리스트에 추가합니다.
            def result_callback(result: str):
                tokens_to_yield.append(result)

            # AgentStreamParser를 팀원분의 의도대로 콜백과 함께 생성합니다.
            parser = AgentStreamParser(
                callbacks=AgentCallbacks(result_callback=result_callback)
            )

            # Agent의 stream 메서드를 호출합니다.
            result_stream = agent.stream(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            
            # 스트림에서 나오는 각 '조각(chunk)'을 처리합니다.
            for chunk in result_stream:
                # 파서의 process_agent_steps가 내부적으로 콜백을 호출합니다.
                parser.process_agent_steps(chunk)
                
                # 콜백 함수가 리스트에 담아둔 토큰이 있다면, 즉시 yield로 보냅니다.
                while tokens_to_yield:
                    token = tokens_to_yield.pop(0)
                    yield f"data: {json.dumps({'token': token})}\n\n"
        
        response = StreamingHttpResponse(stream_response_generator(), content_type="text/event-stream")
        response['X-Accel-Buffering'] = 'no'
        return response

    except json.JSONDecodeError:
        return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)
    except Exception as e:
        print(f"챗봇 API 오류 발생: {e}")
        error_message = json.dumps({'error': f'서버 오류가 발생했습니다: {str(e)}'})
        return StreamingHttpResponse(f"data: {error_message}\n\n", content_type="text/event-stream", status=500)


# ★★★★★ '추천 기반 리포트 생성' 플로우를 위한 뷰 함수 2개 추가 ★★★★★

# --- [View 9] 추천 로딩 페이지 뷰 ---
def recommend_recommending_view(request):
    """
    이력서 선택 후 추천을 시작하면 보여지는 로딩 페이지.
    Agent 호출을 시뮬레이션하고 결과 페이지로 리디렉션합니다.
    """
    resume_ids_str = request.GET.get('resumes', '')
    if not resume_ids_str:
        return redirect('resume_list')

    # AJAX를 사용하지 않는 동기 방식에서는 로딩 페이지를 먼저 보여주고,
    # JavaScript를 통해 다음 단계로 넘어가는 것이 일반적입니다.
    # 여기서는 단순화를 위해 로딩 페이지 템플릿만 렌더링하고,
    # 해당 템플릿에서 meta refresh나 JS로 결과 페이지를 요청하도록 합니다.
    
    # 1. 먼저 로딩 페이지를 렌더링해서 사용자에게 보여줍니다.
    resume_ids = resume_ids_str.split(',')
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    # 2. Agent 호출 로직 (시뮬레이션)
    # TODO: 이 부분에 실제 Agent 호출 로직을 구현합니다.
    #       Agent는 이력서 정보를 받아 추천 공고 목록을 반환해야 합니다.
    print(f"Agent 호출 시작: {len(resume_ids)}개의 이력서로 공고 추천 중...")
    time.sleep(3)  # Agent가 작업하는 시간을 3초로 가정
    
    # Agent가 반환했다고 가정한 가짜 데이터
    fake_recommended_jobs = [
        {'id': 101, 'company': '삼성 SDS', 'title': '하반기 석박사 채용', 'location': '서울'},
        {'id': 102, 'company': 'Apple', 'title': '강남 MD / Staff 모집', 'location': '서울'},
        {'id': 103, 'company': '네이버', 'title': '클라우드 플랫폼 백엔드 개발자', 'location': '성남'},
        {'id': 104, 'company': '카카오', 'title': '데이터 사이언티스트 (인턴)', 'location': '제주'},
    ]
    print("Agent 호출 완료. 추천 공고 반환.")
    
    # 3. 추천 결과와 이력서 ID를 세션에 저장하여 다음 뷰에서 사용하도록 합니다.
    request.session['recommended_jobs'] = fake_recommended_jobs
    request.session['selected_resume_ids'] = resume_ids
    
    # 4. 모든 작업이 끝났으면 결과 페이지로 리디렉션합니다.
    # 사용자는 로딩 페이지를 잠시 본 후 이 페이지로 자동 이동하게 됩니다.
    return redirect('recommend_result')


# --- [View 10] 추천 결과 페이지 뷰 ---
def recommend_result_view(request):
    """
    Agent가 추천한 공고 목록을 보여주고 사용자가 선택할 수 있게 합니다.
    """
    # 세션에서 추천 결과와 이력서 ID 가져오기
    recommended_jobs = request.session.get('recommended_jobs')
    resume_ids = request.session.get('selected_resume_ids')

    # 세션에 데이터가 없으면 (예: 페이지를 새로고침하거나 직접 URL로 접근한 경우) 시작 페이지로 이동
    if not recommended_jobs or not resume_ids:
        # 적절한 오류 메시지를 보여주거나, 이력서 목록으로 리디렉션
        return redirect('resume_list')
    
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    context = {
        'selected_resumes': selected_resumes,
        'recommended_jobs': recommended_jobs,
        'current_page': 'resume',
        'switch_url_name': 'home',
    }

    # 세션 데이터를 사용 후에는 삭제하여 다음 요청에 영향을 주지 않도록 합니다.
    if 'recommended_jobs' in request.session:
        del request.session['recommended_jobs']
    if 'selected_resume_ids' in request.session:
        del request.session['selected_resume_ids']

    return render(request, 'recommend_result.html', context)