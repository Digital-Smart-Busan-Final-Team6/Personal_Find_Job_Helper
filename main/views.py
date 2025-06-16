# main/views.py

import time  # Agent 호출 시뮬레이션을 위한 time 모듈
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from accounts.models import Resume
from accounts.forms import ResumeForm

# Agent 관련 import (실제 구현 시 주석 해제)
from Run_Pipeline.Agent_Manager import get_agent_chain
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

def load_skills_whitelist():
    """JSON 파일에서 스킬 목록을 읽어오는 도우미 함수"""
    # 파일 경로만 새로운 JSON 파일로 변경될 수 있습니다.
    file_path = settings.BASE_DIR / 'Data_Files' / 'skills_dataset.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # 파일이 없을 경우를 대비한 예외 처리
        print(f"경고: 스킬 데이터 파일({file_path})을 찾을 수 없습니다.")
        return "{}" # 빈 JSON 객체 반환
    except Exception as e:
        # 그 외 다른 오류 발생 시
        print(f"오류: 스킬 데이터 파일 처리 중 문제 발생 - {e}")
        return "{}"

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
        'skills_whitelist': load_skills_whitelist()
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
        'skills_whitelist': load_skills_whitelist()
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
            'education_level': resume.education_level,
            'gpa': resume.gpa,
            'experience_years': resume.experience_years,
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
    # --- 기본 정보 로드 ---
    resume_ids_str = request.GET.get('resumes', '')
    if not resume_ids_str:
        return redirect('resume_list')

    resume_ids = resume_ids_str.split(',')
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    # --- 검색어 처리 및 Agent 호출 ---
    search_query = request.GET.get('query', '').strip() # 폼에서 보낸 검색어 가져오기
    search_results = [] # 검색 결과를 담을 리스트

    # 검색어가 있을 경우에만 Agent를 호출합니다.
    if search_query:
        # Agent가 따라야 할 작업 계획을 담은 쿼리 생성
        query_for_agent = f"""
        [작업 목표]
        사용자가 입력한 검색어 "{search_query}" 와 가장 관련성 높은 채용 공고를 찾아서 JSON 형식으로 반환해야 합니다.

        [작업 절차]
        1. `document_search` 툴을 사용하여 위 검색어로 관련 공고를 검색하세요.
        2. 찾은 결과를 아래 [결과 형식]에 맞는 JSON 리스트로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.

        [결과 형식]
        ```json
        [
          {{"id": "고유 ID", "company": "회사명", "title": "공고 제목", "location": "근무지"}},
          {{"id": "고유 ID", "company": "회사명", "title": "공고 제목", "location": "근무지"}}
        ]
        ```
        """
        
        # Agent 호출 및 결과 파싱
        try:
            agent = get_agent_chain()
            if not request.session.session_key:
                request.session.create()
            session_id = request.session.session_key

            print(f'Agent 공고 검색 시작 (검색어: "{search_query}")...')
            result = agent.invoke(
                {"input": query_for_agent},
                config={"configurable": {"session_id": session_id}}
            )
            
            agent_output = result.get('output', '')
            if "```json" in agent_output:
                agent_output = agent_output.split("```json")[1].split("```")[0].strip()
            
            search_results = json.loads(agent_output)

        except Exception as e:
            print(f"Agent 공고 검색 중 오류 발생: {e}")
            search_results = []

    # --- 템플릿에 전달할 최종 context ---
    context = {
        'selected_resumes': selected_resumes,
        'resume_ids_str': resume_ids_str,
        'search_query': search_query, # 사용자가 입력한 검색어를 다시 템플릿으로 전달
        'search_results': search_results, # Agent가 찾은 검색 결과를 전달
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
    사용자의 요청을 받아, Agent가 스스로 이력서를 파악하고 공고를 추천하도록
    '작업 계획'이 담긴 단일 쿼리를 생성하여 전달합니다.
    """
    resume_ids_str = request.GET.get('resumes', '')
    if not resume_ids_str:
        return redirect('resume_list') 
    resume_ids = resume_ids_str.split(',')
    # --- Agent에게 전달할 단일 'query' 문자열 생성 ---
    
    query = """
    [최종 목표]
    내 이력서에 가장 적합한 채용 공고 5개를 JSON 형식으로 추천해줘.

    [작업 절차와 사고 과정]
    1. 먼저, 너가 가진 `resume_qa` 툴을 사용해서 "이력서의 핵심 직무와 주요 기술 스택을 쉼표로 구분해서 나열해줘"라고 질문해. 이렇게 하면 내 이력서의 핵심 정보를 얻을 수 있을 거야.
    2. `resume_qa`가 알려준 이력서 정보를 바탕으로, `document_search` 툴에서 사용할 가장 효과적인 검색 키워드(query)를 만들어.
    3. 그 키워드로 `document_search` 툴을 호출해서 관련성이 높은 채용 공고들을 검색해.
    4. `document_search`의 검색 결과와 `resume_qa`로 파악한 전체 이력서 내용을 종합적으로 비교하고 분석해서, 가장 적합하다고 판단되는 5개의 공고를 신중하게 선정해.
    5. 최종 답변은 다른 부가적인 설명이나 인사말 없이, 반드시 아래 [결과 형식]을 따르는 순수한 JSON 리스트 문자열로만 생성해야 해.

    [결과 형식]
    ```json
    [
      {{"id": 101, "company": "실제 회사명", "title": "실제 공고 제목", "location": "근무지"}},
      {{"id": 102, "company": "실제 회사명", "title": "실제 공고 제목", "location": "근무지"}}
    ]
    ```
    """

    # ---  Agent 호출 및 결과 파싱 ---
    recommended_jobs = []
    try:
        agent = get_agent_chain()
        
        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        print("Agent 호출 시작 (모든 정보가 담긴 단일 쿼리 전달)...")
        result = agent.invoke(
            {"input": query}, # '지시사항이 담긴 쿼리'를 input으로 전달
            config={"configurable": {"session_id": session_id}}
        )
        
        agent_output = result.get('output', '')
        print(f"Agent로부터 받은 결과: {agent_output}")

        if "```json" in agent_output:
            agent_output = agent_output.split("```json")[1].split("```")[0].strip()
        
        recommended_jobs = json.loads(agent_output)
        print("Agent 결과 파싱 성공!")

    except Exception as e:
        print(f"Agent 호출 또는 결과 파싱 중 오류 발생: {e}")
        recommended_jobs = []
    # 세션 저장, 리디렉션
    request.session['recommended_jobs'] = recommended_jobs
    request.session['selected_resume_ids'] = resume_ids
    return redirect('recommend_result')
    



# --- [View 10] 추천 결과 페이지 뷰 ---
def recommend_result_view(request):
    """
    Agent가 추천한 공고 목록을 보여주고 사용자가 선택할 수 있게 합니다.
    """
   # 세션에서 추천 결과만 가져오기.
    recommended_jobs = request.session.get('recommended_jobs')
    selected_resume_ids = request.session.get('selected_resume_ids', [])


    # 세션에 추천 공고 데이터가 없으면 시작 페이지로 이동
    if not recommended_jobs:
        return redirect('resume_list')

    context = {
        'recommended_jobs': recommended_jobs,
        'resume_ids_str': ",".join(selected_resume_ids),
        'current_page': 'resume',
        'switch_url_name': 'home',
    }

    # 세션 데이터를 사용 후에는 삭제하여 다음 요청에 영향을 주지 않도록 합니다.
    if 'recommended_jobs' in request.session:
        del request.session['recommended_jobs']
    if 'selected_resume_ids' in request.session:
        del request.session['selected_resume_ids']

    return render(request, 'recommend_result.html', context)


# --- [View 11] 최종 리포트 생성을 위한 페이지 뷰 (분할 정복 방식) ---
def generate_final_report_view(request):
    if request.method != 'POST':
        return redirect('resume_list')

    # --- 1. 폼 데이터 가져오기 ---
    resume_ids_str = request.POST.get('resume_ids', '')
    selected_job_ids = request.POST.getlist('selected_jobs')

    if not resume_ids_str or not selected_job_ids:
        return redirect('resume_list')

    # ====================================================================
    # ▼▼▼ 분할 정복 로직 시작 ▼▼▼
    # ====================================================================

    final_report_content = "리포트 생성 중 오류가 발생했습니다."
    try:
        agent = get_agent_chain()
        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key
        
        # --- [1단계: 공고 정보 수집] ---
        # for 루프를 돌며 각 공고의 상세 정보를 Agent에게 요청합니다.
        job_details_list = []
        print(">>> [1/3] 공고 정보 수집 시작...")
        for job_id in selected_job_ids:
            # Agent에게는 "이 ID의 정보만 찾아줘" 라는 매우 간단한 작업을 시킵니다.
            query_for_single_job = f"""
            `document_search` 툴을 사용해서 ID가 "{job_id}"인 채용 공고의 상세 내용을 찾아줘.
            다른 설명 없이, 찾은 공고의 내용(회사명, 직무, 자격요건, 우대사항 등)만 텍스트로 응답해줘.
            """
            result = agent.invoke(
                {"input": query_for_single_job},
                config={"configurable": {"session_id": session_id}}
            )
            job_content = result.get('output', f'ID {job_id} 공고 정보를 찾을 수 없습니다.')
            job_details_list.append(f"--- 공고 ID: {job_id} ---\n{job_content}")
            print(f">>> 공고 ID {job_id} 정보 수집 완료.")
        
        # --- [2단계: 데이터 취합] ---
        # 이력서 정보와 수집된 모든 공고 정보를 하나의 큰 텍스트로 합칩니다.
        print(">>> [2/3] 최종 리포트용 데이터 취합...")
        resumes = Resume.objects.filter(id__in=resume_ids_str.split(','))
        resume_details_list = []
        for r in resumes:
            resume_text = f"""
            - 이력서 제목: {r.title}, 최종 학력: {r.education_level}
            - 학교/전공: {r.university} / {r.major}
            - 경력: {r.experience_years}년, 보유 기술: {r.skills}, 희망 직무: {r.job}
            """
            resume_details_list.append(resume_text.strip())
        
        full_resume_details = "\n\n".join(resume_details_list)
        full_job_details = "\n\n".join(job_details_list)

        # --- [3단계: 최종 리포트 생성 요청] ---
        # Agent에게는 이제 '분석 및 작성'이라는 단 하나의 명확한 작업만 요청합니다.
        print(">>> [3/3] 최종 리포트 생성 Agent 호출...")
        query_for_final_report = f"""
        [작업 목표]
        아래에 제공된 [이력서 정보]와 [채용 공고 정보]를 바탕으로,
        이 둘의 적합도를 상세하게 분석하는 최종 매칭 리포트를 생성해야 합니다.

        [이력서 정보]
        {full_resume_details}

        [채용 공고 정보]
        {full_job_details}

        [리포트 작성 지시사항]
        1. 제공된 정보를 바탕으로, 이력서의 강점과 공고의 핵심 요구사항을 비교 분석해주세요.
        2. 어떤 점에서 적합하고, 어떤 점을 보완하면 좋을지 구체적인 근거를 들어 설명해주세요.
        3. 최종 결과물은 친절하고 전문적인 어조의 상세한 텍스트 리포트여야 합니다. 마크다운 형식을 활용하여 가독성을 높여주세요.
        """
        
        final_result = agent.invoke(
            {"input": query_for_final_report},
            config={"configurable": {"session_id": session_id}}
        )
        final_report_content = final_result.get('output', final_report_content)

    except Exception as e:
        print(f"최종 리포트 생성 중 오류 발생: {e}")
        final_report_content = f"AI 분석 중 오류가 발생하여 리포트를 생성할 수 없습니다.\n\n오류: {e}"

    # ====================================================================
    # ▲▲▲ 분할 정복 로직 종료 ▲▲▲
    # ====================================================================
        
    context = {
        'resumes': resumes,
        'report_content': final_report_content,
        'current_page': 'resume',
        'switch_url_name': 'home',
    }
    
    return render(request, 'report_detail.html', context)