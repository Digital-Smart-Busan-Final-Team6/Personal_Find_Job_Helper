# main/views.py

import time  # Agent 호출 시뮬레이션을 위한 time 모듈
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from accounts.models import Resume
from accounts.forms import ResumeForm

# Agent 관련 import (실제 구현 시 주석 해제)
from Run_Pipeline.Agent_Factory import get_agent_chain
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks


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

# --- [View 9] 추천 로딩 페이지 뷰 ---
def recommend_recommending_view(request):
    """
    이력서 선택 후 추천을 시작하면 보여지는 로딩 페이지.
    Agent를 호출하여 결과를 생성하고 결과 페이지로 리디렉션합니다.
    """
    resume_ids_str = request.GET.get('resumes', '')
    if not resume_ids_str:
        return redirect('resume_list')

    resume_ids = resume_ids_str.split(',')
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    # --- 1. 이력서 데이터 준비 ---
    resume_texts = []
    for i, resume in enumerate(selected_resumes, 1):
        text = f"""### 이력서 {i}: {resume.title} ###
- 희망 직무: {resume.job}
- 희망 근무지: {resume.location}
- 학력: {resume.university} {resume.major} ({resume.gpa})
- 보유 기술: {resume.skills}
"""
        resume_texts.append(text)
    combined_resume_text = "\n---\n".join(resume_texts)

    # --- 2. LLM Agent를 위한 프롬프트 설계 ---
    prompt = f"""
당신은 최고의 커리어 컨설턴트입니다. 아래에 제공된 이력서 내용을 바탕으로, 이 사람(들)에게 가장 적합할 것으로 예상되는 최신 채용 공고를 5개 추천해 주세요.

[이력서 내용]
{combined_resume_text}

[지시사항]
1. 이력서의 희망 직무, 보유 기술, 학력 등을 종합적으로 고려하여 추천해야 합니다.
2. 추천 결과는 반드시 아래와 같은 JSON 형식의 리스트로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
3. 각 공고의 id는 임의의 고유한 정수 값을 부여하세요.

```json
[
  {{"id": 101, "company": "회사명", "title": "공고 제목", "location": "근무지"}},
  {{"id": 102, "company": "회사명", "title": "공고 제목", "location": "근무지"}}
]
"""
    
    # --- 3. Agent 호출 및 4. 결과 파싱 ---
    recommended_jobs = []
    try:
        # 3.1 Agent 생성
        agent = get_agent_chain()
        
        # 3.2 Agent 실행 (채팅이 아닌 단일 요청/응답에는 .invoke()가 더 적합)
        #    이 프롬프트를 Agent에게 전달하여 결과를 요청합니다.
        #    세션 ID 가져오기 (없으면 생성)
        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        #    invoke 호출 시 config에 session_id 전달
        print("실제 Agent 호출 시작...")
        result = agent.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 3.3 결과 추출 (Agent의 출력 key가 'output'이라고 가정)
        agent_output = result.get('output', '')
        print(f"Agent로부터 받은 결과: {agent_output}")

        # 3.4 LLM이 생성한 ```json ... ``` 코드 블록을 정리 (이 부분은 유지)
        if "```json" in agent_output:
            agent_output = agent_output.split("```json")[1].split("```")[0].strip()
        
        # 3.5 JSON 텍스트를 파이썬 리스트로 변환
        recommended_jobs = json.loads(agent_output)
        print("Agent 결과 파싱 성공!")

    except json.JSONDecodeError:
        print("오류: Agent가 유효한 JSON을 반환하지 않았습니다.")
        recommended_jobs = [] 
    except Exception as e:
        print(f"Agent 호출 중 오류 발생: {e}")
        recommended_jobs = []

    # 추천 결과와 원본 이력서 ID를 세션에 저장
    request.session['recommended_jobs'] = recommended_jobs
    request.session['selected_resume_ids'] = resume_ids

    # 결과 페이지로 리디렉션
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
        'resume_ids_str': ','.join(resume_ids),
        'current_page': 'resume',
        'switch_url_name': 'home',
    }

    # 세션 데이터를 사용 후에는 삭제하여 다음 요청에 영향을 주지 않도록 합니다.
    if 'recommended_jobs' in request.session:
        del request.session['recommended_jobs']
    if 'selected_resume_ids' in request.session:
        del request.session['selected_resume_ids']

    return render(request, 'recommend_result.html', context)

# --- [View 11] 최종 리포트 생성을 위한 페이지 뷰 ---
def generate_final_report_view(request):
    """
    선택된 이력서와 추천 공고를 바탕으로 최종 매칭 리포트를 생성합니다.
    """
    if request.method != 'POST':
        # POST 요청이 아니면 비정상적인 접근으로 보고 리디렉션
        return redirect('resume_list')

    # --- 폼에서 전송된 데이터 가져오기 ---
    # 1. 'recommend_result.html'의 <input type="hidden" name="resume_ids" ...> 태그가 보낸 값을 받음
    resume_ids_str = request.POST.get('resume_ids', '')
    
    # 2. 'recommend_result.html'의 <input type="checkbox" name="selected_jobs" ...> 태그들이 보낸 값들을 받음
    selected_job_ids = request.POST.getlist('selected_jobs')

    if not resume_ids_str or not selected_job_ids:
        # 필요한 정보가 없으면 오류 처리 또는 리디렉션
        return redirect('resume_list')

    # --- 최종 리포트 생성을 위한 데이터 준비 ---
    # 이력서 정보 가져오기
    resume_ids = resume_ids_str.split(',')
    resumes = Resume.objects.filter(id__in=resume_ids)
    
    # TODO: 실제 서비스에서는 DB에서 공고 정보를 가져와야 합니다.
    # 지금은 가상의 공고 데이터에서 선택된 것만 필터링합니다.
    all_fake_jobs = [
        {'id': '101', 'company': '삼성 SDS', 'title': '하반기 석박사 채용'},
        {'id': '102', 'company': 'Apple', 'title': '강남 MD / Staff 모집'},
        {'id': '103', 'company': '네이버', 'title': '클라우드 플랫폼 백엔드 개발자'},
        {'id': '104', 'company': '카카오', 'title': '데이터 사이언티스트 (인턴)'},
        {'id': '105', 'company': '현대자동차', 'title': '자율주행 소프트웨어 엔지니어'}
    ]
    selected_jobs = [job for job in all_fake_jobs if job['id'] in selected_job_ids]

    # --- 최종 리포트 생성을 위한 Agent 호출 ---
    
    # 1. 두 번째 Agent를 위한 새로운 프롬프트 설계
    # 이력서와 공고 정보를 모두 텍스트로 변환하여 프롬프트에 포함시킵니다.
    resume_details = "\n".join([f"- 이력서: {r.title}" for r in resumes])
    job_details = "\n".join([f"- 공고: {j['company']} - {j['title']}" for j in selected_jobs])
    
    final_prompt = f"""
당신은 경력 개발 전문가입니다. 아래 제공된 이력서들과 채용 공고들의 적합도를 상세하게 분석하고, 최종 매칭 리포트를 생성해 주세요.

[분석 대상 이력서]
{resume_details}

[분석 대상 공고]
{job_details}

[리포트 작성 지시사항]
1. 각 이력서와 공고의 핵심 요구사항 및 강점을 요약하세요.
2. 이력서의 기술, 경험과 공고의 자격 요건을 비교하여 적합도를 분석해 주세요 (예: "매우 적합", "부분적으로 적합").
3. 어떤 점에서 강점이 있고, 어떤 점이 부족할 수 있는지 구체적인 근거를 들어 설명해 주세요.
4. 최종적으로 종합적인 평가와 지원자에게 도움이 될 만한 조언을 포함하여 리포트를 완성해 주세요.
5. 친절하고 전문적인 어조로 작성해 주세요.
"""

    # 2. Agent 호출 및 결과 사용
    try:
        agent = get_agent_chain()
        #  세션 ID 가져오기 (없으면 생성)
        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        #  invoke 호출 시 config에 session_id 전달
        print("최종 리포트 생성을 위해 Agent 호출...")
        result = agent.invoke(
            {"input": final_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        final_report_content = result.get('output', '리포트 생성 중 오류가 발생했습니다.')

    except Exception as e:
        print(f"최종 리포트 생성 Agent 호출 중 오류: {e}")
        final_report_content = "AI 분석 중 오류가 발생하여 리포트를 생성할 수 없습니다."
        

    context = {
        'resumes': resumes,
        'jobs': selected_jobs,
        'report_content': final_report_content, # ★ Agent가 생성한 실제 내용으로 교체됨
        'current_page': 'resume',
        'switch_url_name': 'home',
    }
    
    return render(request, 'report_detail.html', context)