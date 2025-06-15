# main/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

# from dotenv import load_dotenv
# from Run_Pipeline.Main_Pipeline import main

from accounts.models import Resume
from accounts.forms import ResumeForm

from Run_Pipeline.Agent_Factory import get_agent_chain
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks

# load_dotenv()
# chain = None

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