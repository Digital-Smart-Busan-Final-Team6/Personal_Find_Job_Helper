# main/views.py

import time  # Agent 호출 시뮬레이션을 위한 time 모듈
import json
import re 
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from accounts.models import Resume
from accounts.forms import ResumeForm

# Agent 관련 import
from Run_Pipeline.Agent_Manager import get_agent_chain
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from .utils import parse_markdown_table_to_json  # utils에서 함수 호출


# --- [View 1] 챗봇이 있는 메인 페이지 ---
def home(request):
    context = {
        "current_page": "home",
        "switch_url_name": "resume_list",
    }
    return render(request, "home.html", context)


# --- [View 2] 이력서 '목록'을 보여주는 페이지 (Read) ---
def resume_list_view(request):
    resumes = Resume.objects.all().order_by("-updated_at")
    context = {
        "resumes": resumes,
        "current_page": "resume",
        "switch_url_name": "home",
    }
    return render(request, "resume_list.html", context)


# 스킬 목록 읽어오는 페이지
def load_skills_whitelist():
    """JSON 파일에서 스킬 목록을 읽어오는 도우미 함수"""
    # 파일 경로만 새로운 JSON 파일로 변경될 수 있습니다.
    file_path = settings.BASE_DIR / "Data_Files" / "skills_dataset.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # 파일이 없을 경우를 대비한 예외 처리
        print(f"경고: 스킬 데이터 파일({file_path})을 찾을 수 없습니다.")
        return "{}"  # 빈 JSON 객체 반환
    except Exception as e:
        # 그 외 다른 오류 발생 시
        print(f"오류: 스킬 데이터 파일 처리 중 문제 발생 - {e}")
        return "{}"


# --- [View 3] 새 이력서를 '추가'하는 페이지 (Create) ---
def resume_add_view(request):
    if request.method == "POST":
        form = ResumeForm(request.POST)
        job_list = request.POST.getlist("job")
        location_list = request.POST.getlist("location")

        if form.is_valid():
            new_resume = form.save(commit=False)
            new_resume.job = ",".join(job_list)
            new_resume.location = ",".join(location_list)
            new_resume.save()
            return redirect("resume_list")
    else:
        form = ResumeForm()

    context = {
        "form": form,
        "selected_jobs": [],
        "selected_locations": [],
        "is_edit_mode": False,
        "skills_whitelist": load_skills_whitelist(),
    }
    return render(request, "resume_form.html", context)


# --- [View 4] 특정 이력서를 '수정'하는 페이지 (Update) ---
def resume_edit_view(request, resume_id):
    resume = get_object_or_404(Resume, id=resume_id)
    if request.method == "POST":
        form = ResumeForm(request.POST, instance=resume)
        job_list = request.POST.getlist("job")
        location_list = request.POST.getlist("location")

        if form.is_valid():
            updated_resume = form.save(commit=False)
            updated_resume.job = ",".join(job_list)
            updated_resume.location = ",".join(location_list)
            updated_resume.save()
            return redirect("resume_list")
    else:
        form = ResumeForm(instance=resume)

    context = {
        "form": form,
        "resume": resume,
        "selected_jobs": resume.job.split(",") if resume.job else [],
        "selected_locations": resume.location.split(",") if resume.location else [],
        "is_edit_mode": True,
        "skills_whitelist": load_skills_whitelist(),
    }
    return render(request, "resume_form.html", context)


# --- [View 5] 특정 이력서를 '삭제'하는 기능 (Delete) ---
def resume_delete_view(request, resume_id):
    resume = get_object_or_404(Resume, id=resume_id)
    if request.method == "POST":
        resume.delete()
        return redirect("resume_list")
    return redirect("resume_list")


# --- [View 6] 모든 이력서를 JSON 파일로 내보내는 기능 ---
def export_resumes_to_json(request):
    resumes = Resume.objects.all()
    data_list = []
    for resume in resumes:
        data = {
            "id": resume.id,
            "title": resume.title,
            "university": resume.university,
            "major": resume.major,
            "education_level": resume.education_level,
            "gpa": resume.gpa,
            "experience_years": resume.experience_years,
            "job_interests": resume.job.split(",") if resume.job else [],
            "location_interests": resume.location.split(",") if resume.location else [],
            "skills": resume.skills,
            "created_at": resume.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": resume.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        data_list.append(data)
    return JsonResponse(
        data_list, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2}
    )


# --- [View 7] 공고 검색 리포트 페이지 ---
def job_search_report_page(request):
    # --- 기본 정보 로드 ---
    resume_ids_str = request.GET.get("resumes", "")
    if not resume_ids_str:
        return redirect("resume_list")

    resume_ids = resume_ids_str.split(",")
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    # --- 검색어 처리 및 Agent 호출 준비 ---
    search_query = request.GET.get("query", "").strip()

    agent_response = None

    # 검색어가 있을 경우에만 Agent를 호출합니다.
    if search_query:
        query_for_agent = f"""
        사용자 검색어 '{search_query}'와 가장 관련 있는 채용 공고를 분석해줘.

        최종 답변은 다른 부가적인 설명이나 인사말 없이, 반드시 다음 두 부분을 포함하는 **하나의 JSON 객체**로만 응답해야 해.
        ```json
        {{
          "analysis_text": "사용자에게 보여줄 친절한 자연어 분석 텍스트입니다. 예를 들어, '삼성전자 관련 공고는 현재 2건이 있으며, 특히 DX 부문 공고가 이력서와 관련성이 높아 보입니다.' 와 같이 작성해주세요.",
          "job_list": [
            {{"id": "고유 ID", "company": "회사명", "title": "공고 제목", "location": "근무지"}},
            {{"id": "고유 ID", "company": "회사명", "title": "공고 제목", "location": "근무지"}}
          ]
        }}
        ```
        """

        try:
            # Agent 호출
            agent = get_agent_chain()
            if not request.session.session_key:
                request.session.create()
            session_id = request.session.session_key

            print(f"Agent 호출 시작 (구조화된 요청 전달)")

            result = agent.invoke(
                {"input": query_for_agent},
                config={"configurable": {"session_id": session_id}},
            )

            agent_output = result.get("output", "{}")
            print(f"Agent 응답 수신 완료:\n{agent_output}")

            # Agent가 생성한 JSON 문자열을 파싱
            if "```json" in agent_output:
                json_part = agent_output.split("```json")[1].split("```")[0].strip()
            else:
                json_part = agent_output

            agent_response = json.loads(json_part)

        except Exception as e:
            # 예외 발생 시 사용자에게 보여줄 메시지를 설정합니다.
            print(f"Agent 공고 검색 중 오류 발생: {e}")
            # 오류 발생 시, 템플릿에서 에러 처리를 할 수 있도록 analysis_text에 메시지 전달
            agent_response = {
                "analysis_text": "AI 에이전트와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "job_list": [],
            }

    # --- 템플릿에 전달할 최종 context ---
    context = {
        "selected_resumes": selected_resumes,
        "resume_ids_str": resume_ids_str,
        "search_query": search_query,
        "agent_response": agent_response,
        "current_page": "resume",
        "switch_url_name": "home",
    }
    return render(request, "job_search_report.html", context)


# --- [View 8] 챗봇 API 엔드포인트 ---
@csrf_exempt
def chat_api(request):
    """
    챗봇의 질문에 대한 응답을 실시간 스트리밍으로 처리하는 API입니다.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 허용됩니다."}, status=405)

    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()

        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        if not message:
            return StreamingHttpResponse(
                'data: {"error": "질문을 입력해주세요."}\n\n',
                content_type="text/event-stream",
            )

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
                {"input": message}, config={"configurable": {"session_id": session_id}}
            )

            # 스트림에서 나오는 각 '조각(chunk)'을 처리합니다.
            for chunk in result_stream:
                # 파서의 process_agent_steps가 내부적으로 콜백을 호출합니다.
                parser.process_agent_steps(chunk)

                # 콜백 함수가 리스트에 담아둔 토큰이 있다면, 즉시 yield로 보냅니다.
                while tokens_to_yield:
                    token = tokens_to_yield.pop(0)
                    yield f"data: {json.dumps({'token': token})}\n\n"

        response = StreamingHttpResponse(
            stream_response_generator(), content_type="text/event-stream"
        )
        response["X-Accel-Buffering"] = "no"
        return response

    except json.JSONDecodeError:
        return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)
    except Exception as e:
        print(f"챗봇 API 오류 발생: {e}")
        error_message = json.dumps({"error": f"서버 오류가 발생했습니다: {str(e)}"})
        return StreamingHttpResponse(
            f"data: {error_message}\n\n", content_type="text/event-stream", status=500
        )


# --- [View 9] 추천 로딩 페이지 뷰 ---
def recommend_recommending_view(request):
    # --- 기본 정보 로드 ---
    resume_ids_str = request.GET.get("resumes", "")
    if not resume_ids_str:
        return redirect("resume_list")
    resume_ids = resume_ids_str.split(",")

    query = "내 이력서에 가장 적합한 공고를 추천해줘."

    # ---  Agent 호출 및 결과 파싱 ---
    recommended_jobs = []
    try:
        agent = get_agent_chain()

        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        print("Agent 호출 시작 (모든 정보가 담긴 단일 쿼리 전달)...")
        result = agent.invoke(
            {"input": query},  # '지시사항이 담긴 쿼리'를 input으로 전달
            config={"configurable": {"session_id": session_id}},
        )

        agent_output = result.get("output", "")
        print(f"Agent로부터 받은 결과: {agent_output}")

        recommended_jobs = parse_markdown_table_to_json(agent_output)
        print("Agent 결과(Markdown) 파싱 성공!")

    except Exception as e:
        print(f"Agent 호출 또는 결과 파싱 중 오류 발생: {e}")
        recommended_jobs = []
    # 세션 저장, 리디렉션
    request.session["recommended_jobs"] = recommended_jobs
    request.session["selected_resume_ids"] = resume_ids
    return redirect("recommend_result")


# --- [View 10] 추천 결과 페이지 뷰 ---
def recommend_result_view(request):

    # 세션에서 추천 결과만 가져오기.
    recommended_jobs = request.session.get("recommended_jobs")
    selected_resume_ids = request.session.get("selected_resume_ids", [])

    # 세션에 추천 공고 데이터가 없으면 시작 페이지로 이동
    if not recommended_jobs:
        return redirect("resume_list")

    context = {
        "recommended_jobs": recommended_jobs,
        "resume_ids_str": ",".join(selected_resume_ids),
        "current_page": "resume",
        "switch_url_name": "home",
    }

    # 세션 데이터를 사용 후에는 삭제하여 다음 요청에 영향을 주지 않도록 합니다.
    if "recommended_jobs" in request.session:
        del request.session["recommended_jobs"]
    if "selected_resume_ids" in request.session:
        del request.session["selected_resume_ids"]

    return render(request, "recommend_result.html", context)


# --- [View 11] 최종 리포트 생성을 위한 페이지 뷰 (최종 수정 버전) ---
def generate_final_report_view(request):
    if request.method != "POST":
        return redirect("resume_list")

    # --- 기본 정보 로드 ---
    resume_ids_str = request.POST.get("resume_ids", "")
    selected_job_ids = request.POST.getlist("selected_jobs")

    if not resume_ids_str or not selected_job_ids:
        messages.error(request, "이력서와 채용 공고를 선택해야 합니다.")
        return redirect("resume_list")

    main_resume_id = resume_ids_str.split(",")[0]
    selected_job_id_str = selected_job_ids[0]

    # --- 화면 표시에 필요한 추가 정보 로드 ---
    job_detail = None
    try:
        job_file_path = settings.BASE_DIR / "Data_Files" / "wanted_detail_improve_20250616.json"
        with open(job_file_path, 'r', encoding='utf-8') as f:
            all_jobs = json.load(f)

        if 'postings' in all_jobs and isinstance(all_jobs['postings'], dict):
            job_data_source = all_jobs['postings']
        else:
            job_data_source = all_jobs

        job_detail = job_data_source.get(selected_job_id_str)

    except Exception as e:
        print(f"공고 정보 조회 중 오류 발생: {e}")

    # --- Agent 호출 ---
    report_markdown = "### 분석 실패\nAI 리포트 생성 중 오류가 발생했습니다."
    try:
        if job_detail:
            agent = get_agent_chain()
            if not request.session.session_key:
                request.session.create()
            session_id = request.session.session_key

            simple_query = f"채용 공고 {selected_job_id_str}에 대한 상세 분석 리포트를 작성해줘."
            
            print(f">>> 최종 리포트 생성 Agent 호출 (Query: {simple_query})")
            final_result = agent.invoke(
                {"input": simple_query}, config={"configurable": {"session_id": session_id}}
            )
            
            # ★★★★★★★★★★★★★★★ 핵심 수정 부분 ★★★★★★★★★★★★★★★
            # 1. AI가 생성한 '친절한 문장 전체'를 가져옵니다.
            agent_output = final_result.get("output", "")
            
            # 2. 정규 표현식을 사용해 문장 속에서 'Reports/...' 형태의 파일 경로만 추출합니다.
            #    슬래시(/)와 역슬래시(\\)를 모두 처리할 수 있도록 패턴을 작성합니다.
            match = re.search(r"Reports[/\\]job_report_[\w-]+\.md", agent_output)
            
            # 3. 파일 경로를 성공적으로 추출했다면,
            if match:
                # 추출한 경로(예: 'Reports\\job_report_...md')를 사용합니다.
                report_file_path_str = match.group(0)
                report_full_path = settings.BASE_DIR / report_file_path_str
                
                # 파일을 열어서 내용을 읽습니다.
                with open(report_full_path, 'r', encoding='utf-8') as f:
                    report_markdown = f.read()
                print(f"성공: 리포트 파일 '{report_file_path_str}'의 내용을 읽었습니다.")
            else:
                # 만약 문장에서 파일 경로를 찾지 못했다면, AI의 답변을 그대로 보여줍니다.
                report_markdown = f"### 분석 결과\nAI가 리포트 파일을 생성했지만, 경로를 찾을 수 없습니다.\n\n**AI 응답 원문:**\n```\n{agent_output}\n```"
                print(f"경고: AI 응답에서 파일 경로를 찾지 못했습니다. 원문: {agent_output}")
            # ★★★★★★★★★★★★★★★ 수정 끝 ★★★★★★★★★★★★★★★

        else:
            report_markdown = "### 분석 실패\n요청하신 채용 공고 정보를 찾을 수 없어 AI 분석을 진행할 수 없습니다."

    except Exception as e:
        print(f"최종 리포트 생성 중 오류 발생: {e}")
        report_markdown = f"### 리포트 생성 오류\nAI 분석 중 오류가 발생했습니다.\n\n**오류:** `{e}`"

    # --- 결과 렌더링 ---
    resumes = Resume.objects.filter(id__in=resume_ids_str.split(","))

    context = {
        "resumes": resumes,
        "selected_job_id": selected_job_id_str,
        "job_detail": job_detail,
        "report_markdown": report_markdown,
        "current_page": "resume",
        "switch_url_name": "home",
    }
    return render(request, "report_detail.html", context)