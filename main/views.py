import time
import json
import re
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.views.decorators.http import require_POST
from pathlib import Path

from accounts.models import Resume
from accounts.forms import ResumeForm

# Agent 관련 import
from Run_Pipeline.Agent_Manager import get_agent_chain
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from .utils import parse_markdown_table_to_json
from django.forms.models import model_to_dict


# --- [View 1] 챗봇이 있는 메인 페이지 ---
def home(request):
    context = {
        "current_page": "home",
        "switch_url_name": "resume_list",
    }
    return render(request, "home.html", context)


# --- [View 2] 이력서 목록을 보여주는 페이지 (Read) ---
def resume_list_view(request):
    resumes = Resume.objects.all().order_by("-updated_at")
    selected_resume_ids = request.session.get('selected_resume_ids', [])
    context = {
        "resumes": resumes,
        "current_page": "resume",
        "switch_url_name": "home",
        # 템플릿의 자바스크립트에서 사용할 수 있도록 JSON 문자열 형태로 전달합니다.
        "selected_resume_ids_json": json.dumps(selected_resume_ids)
    }
    return render(request, "resume_list.html", context)


# 스킬 목록 읽어오는 페이지
def load_skills_whitelist():
    """JSON 파일에서 스킬 목록을 읽어오는 도우미 함수"""
    file_path = settings.BASE_DIR / "Data_Files" / "skills_dataset.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"경고: 스킬 데이터 파일({file_path})을 찾을 수 없습니다.")
        return "{}"
    except Exception as e:
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


# --- [View 4] 특정 이력서를 수정하는 페이지 (Update) ---
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


# --- [View 5] 특정 이력서를 삭제하는 기능 (Delete) ---
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
    resume_ids_str = request.GET.get("resumes", "")
    if not resume_ids_str:
        return redirect("resume_list")

    resume_ids = resume_ids_str.split(",")
    selected_resumes = Resume.objects.filter(id__in=resume_ids)

    # ---  Agent 호출 준비 ---
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
            agent = get_agent_chain(mode = "chat")
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
            print(f"Agent 공고 검색 중 오류 발생: {e}")
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
    챗봇 질문에 실시간 스트리밍으로 응답하는 API이다.
    세션에 저장된 이력서 정보를 Agent에게 전달한다.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 허용된다."}, status=405)

    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()

        # 세션 초기화
        if not request.session.session_key:
            request.session.create()
        session_id = request.session.session_key

        # 빈 질문 체크
        if not message:
            return StreamingHttpResponse(
                'data: {"error": "질문을 입력해라."}\n\n',
                content_type="text/event-stream",
            )

        # --- ✨ 핵심 수정 부분 ✨ ---
        selected_resume_ids = request.session.get("selected_resume_ids", [])
        final_input_message = ""

        if selected_resume_ids:
            resume_id = selected_resume_ids[0]  # 여러 개면 첫 번째만 사용
            try:
                resume_obj = Resume.objects.get(id=resume_id)
                resume_data_for_tool = model_to_dict(resume_obj)
                resume_json_str = json.dumps(
                    resume_data_for_tool, ensure_ascii=False, indent=2
                )

                final_input_message = f"""
[사용자 컨텍스트 정보]
- 현재 사용자가 선택한 이력서 데이터가 아래와 같이 주어졌다.
- '내 이력서', '제 이력서' 등 이력서 기반 요청이 오면 반드시 `RESUME_DATA`를 활용한다.
- `find_best_job_match`, `write_job_report` 툴 사용 시 `resume` 파라미터에 이 데이터를 전달해야 한다.

```json
"RESUME_DATA": {resume_json_str}
[사용자 실제 질문]
{message}
"""
                print(f"✅ 세션 이력서(ID: {resume_id}) 정보를 포함하여 Agent에 전달한다.")
            except Resume.DoesNotExist:
                final_input_message = message
                print(f"⚠️ 세션의 이력서 ID({resume_id})를 DB에서 찾을 수 없다.")
        else:
            final_input_message = message
            print("ℹ️ 선택된 이력서가 없어, 사용자 질문만 Agent에 전달한다.")
        # --- ✨ 수정 끝 ✨ ---

        agent = get_agent_chain(mode="chat")

        def stream_response_generator():
            is_first_chunk = True
            tokens_to_yield = []

            def result_callback(result: str):
                tokens_to_yield.append(result)

            parser = AgentStreamParser(
                callbacks=AgentCallbacks(result_callback=result_callback)
            )

            # Agent 호출
            result_stream = agent.stream(
                {"input": final_input_message},
                config={"configurable": {"session_id": session_id}},
            )

            for chunk in result_stream:
                parser.process_agent_steps(chunk)
                while tokens_to_yield:
                    token = tokens_to_yield.pop(0)

                    # 첫 토큰 처리
                    if is_first_chunk:
                        cleaned_token = re.sub(
                            r"^\s*\**Answer\**\s*:?\s*", "", token, flags=re.IGNORECASE
                        ).lstrip()
                        if cleaned_token.startswith("**"):
                            cleaned_token = cleaned_token[2:].lstrip()
                        token = cleaned_token
                        is_first_chunk = False

                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"

        response = StreamingHttpResponse(
            stream_response_generator(),
            content_type="text/event-stream",
        )
        response["X-Accel-Buffering"] = "no"
        return response

    except json.JSONDecodeError:
        return JsonResponse({"error": "잘못된 JSON 형식이다."}, status=400)
    except Exception as e:
        print(f"챗봇 API 오류 발생: {e}")
        error_message = json.dumps({"error": f"서버 오류가 발생했다: {str(e)}"})
        return StreamingHttpResponse(
            f"data: {error_message}\n\n",
            content_type="text/event-stream",
            status=500,
        )


# --- [View 9] 추천 로딩 페이지 뷰 ---
# request.session에서 데이터를 읽도록 수정
def recommend_recommending_view(request):
    # 1. URL 파라미터 대신 세션에서 이력서 ID 목록을 가져옵니다.
    resume_ids = request.session.get('selected_resume_ids', [])
    
    # 2. 세션에 이력서 정보가 없으면 목록 페이지로 돌려보냅니다.
    if not resume_ids:
        messages.warning(request, '추천을 위한 이력서가 선택되지 않았습니다.')
        return redirect("resume_list") # resume_list_view의 URL name이 'resume_list'라고 가정

    # resume_ids는 이미 정수형 ID의 리스트일 수 있으나, 안전을 위해 변환합니다.
    try:
        resume_ids = [int(id) for id in resume_ids]
    except (ValueError, TypeError):
        messages.error(request, '잘못된 이력서 정보입니다.')
        return redirect("resume_list")

    # --- 여기부터는 기존 로직과 거의 동일합니다 ---
    resume_objects = Resume.objects.filter(id__in=resume_ids)
    if not resume_objects.exists():
        messages.warning(request, "선택된 이력서를 찾을 수 없습니다.")
        request.session["recommended_jobs"] = []
        return redirect("recommend_result")

    resume_data_list = []
    # Agent에게 보낼 쿼리는 모든 이력서 정보를 종합하여 만들 수 있습니다.
    # 여기서는 첫 번째 이력서를 기준으로 쿼리를 만드는 기존 로직을 유지합니다.
    query = ""
    for resume_obj in resume_objects:
        resume_dict = model_to_dict(
            resume_obj,
            fields=[
                "education_level", "university", "major", "gpa",
                "experience_years", "job", "location",
                "skills", "experience", "certifications"
            ]
        )
        resume_dict['title'] = resume_obj.title
        resume_data_list.append(resume_dict)
    
    # 대표 이력서(첫번째)를 기준으로 쿼리 생성
    if resume_data_list:
        representative_resume = resume_data_list[0]
        query = f"""
        사용자의 이력서 정보를 바탕으로 가장 적합한 공고를 추천해 주세요.
        
        ## 이력서 정보
        {representative_resume}
        
        다음과 같이 find_best_job_match 도구를 호출해 주세요:
        find_best_job_match(resume={representative_resume})
        
        중요: resume 파라미터에 위의 이력서 JSON 데이터를 정확히 전달해 주세요.
        """

    # --- Agent 호출 및 결과 파싱 (기존과 동일) ---
    recommended_jobs = []
    if query:
        try:
            agent = get_agent_chain(mode="job")

            if not request.session.session_key:
                request.session.create()
            session_id = request.session.session_key

            print(f"Agent 호출 시작 ({len(resume_data_list)}개 이력서 기반)...")
            print(f"쿼리 미리보기: {query[:200]}...")

            result = agent.invoke(
                {"input": query},
                {"configurable": {"session_id": session_id}},
            )

            agent_output = result.get("output", "")
            print(f"Agent로부터 받은 결과: {agent_output[:500]}...")

            recommended_jobs = parse_markdown_table_to_json(agent_output)
            print(f"Agent 결과(Markdown) 파싱 성공! {len(recommended_jobs)}개 공고 추천됨")

        except Exception as e:
            print(f"Agent 호출 또는 결과 파싱 중 오류 발생: {e}")
            recommended_jobs = []

    # 세션에 결과 저장, 리디렉션
    request.session["recommended_jobs"] = recommended_jobs
    # 'selected_resume_ids'는 이미 세션에 있으므로 다시 저장할 필요가 없습니다.
    request.session["selected_resumes_data"] = resume_data_list

    return redirect("recommend_result")

# --- [View 10] 추천 결과 페이지 뷰 ---
def recommend_result_view(request):
    recommended_jobs_from_session = request.session.get("recommended_jobs")
    selected_resume_ids = request.session.get("selected_resume_ids", [])

    if not recommended_jobs_from_session:
        messages.warning(request, "추천된 공고가 없습니다. 다시 시도해주세요.")
        return redirect("resume_list")

    job_data_source = {}
    try:
        job_file_path = settings.BASE_DIR / "Data_Files" / "wanted_detail_improve_20250616.json"
        with open(job_file_path, 'r', encoding='utf-8') as f:
            all_jobs_data = json.load(f)
        
        if 'postings' in all_jobs_data and isinstance(all_jobs_data['postings'], dict):
            job_data_source = all_jobs_data['postings']
        else:
            job_data_source = all_jobs_data

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"오류: 공고 데이터 파일({job_file_path})을 로드할 수 없습니다. - {e}")
        messages.error(request, "공고 정보를 불러오는 데 실패했습니다.")

    enriched_recommended_jobs = []
    for job in recommended_jobs_from_session:
        job_id_from_parser = job.get("공고_ID") 
        if not job_id_from_parser:
            print(f"경고: 파싱된 데이터에서 '공고_ID' 키를 찾을 수 없습니다. 데이터: {job}")
            continue

        enriched_job = job.copy()
        enriched_job['job_id'] = job_id_from_parser
        
        job_details = job_data_source.get(str(job_id_from_parser))

        if job_details:
            enriched_job['회사명'] = job_details.get('회사명')
            min_exp = job_details.get('요구 최소 경력', 0)
            enriched_job['경력'] = '신입' if min_exp == 0 else f"{min_exp}년 이상"
            enriched_job['근무지'] = job_details.get('근무지')
            enriched_job['기술스택'] = job_details.get('기술 스택', [])
        else:
            print(f"경고: 원본 데이터에서 공고 ID '{job_id_from_parser}'를 찾을 수 없습니다.")
            enriched_job['회사명'] = '정보 조회 실패'
            enriched_job['경력'], enriched_job['근무지'], enriched_job['기술스택'] = None, None, []
        
        try:
            suitability_score = float(job.get('적합도', 0))
            enriched_job['적합도_정규화'] = suitability_score
            enriched_job['적합도_퍼센트'] = round(suitability_score * 100, 1)
        except (ValueError, TypeError):
            enriched_job['적합도_정규화'], enriched_job['적합도_퍼센트'] = 0.0, 0.0

        enriched_recommended_jobs.append(enriched_job)

    context = {
        "recommended_jobs": enriched_recommended_jobs,
        "resume_ids_str": ",".join(map(str, selected_resume_ids)),
        "current_page": "resume",
        "switch_url_name": "home",
    }

    return render(request, "recommend_result.html", context)


def format_resume_to_markdown(resume: Resume) -> str:
    """Resume 객체를 마크다운 형식의 문자열로 변환합니다."""
    md_parts = []

    # 1. 제목
    if resume.title:
        md_parts.append(f"# {resume.title}")
        md_parts.append("")

    # 2. 기본 정보
    md_parts.append("## 기본 정보")
    if resume.education_level:
        md_parts.append(f"- **최종 학력**: {resume.education_level}")
    if resume.university or resume.major:
        md_parts.append(f"- **학교/전공**: {resume.university} / {resume.major}")
    if resume.gpa:
        md_parts.append(f"- **학점**: {resume.gpa}")
    if resume.experience_years is not None:
        md_parts.append(f"- **총 경력**: {resume.experience_years}년")
    if resume.job:
        md_parts.append(f"- **희망 직무**: {resume.job}")
    if resume.location:
        md_parts.append(f"- **희망 근무지**: {resume.location}")
    md_parts.append("\n---\n")

    # 3. 보유 기술
    if resume.skills:
        md_parts.append("## 보유 기술")
        try:
            # 스킬이 JSON 형식일 경우 (e.g., [{"value":"Python"},...])
            skills_list = json.loads(resume.skills)
            skill_values = [item.get('value') for item in skills_list if item.get('value')]
            md_parts.append(f"- {', '.join(skill_values)}")
        except (json.JSONDecodeError, TypeError):
            # 일반 텍스트나 콤마로 구분된 문자열일 경우
            md_parts.append(f"- {resume.skills}")
        md_parts.append("\n---\n")

    # 4. 경험 / 활동 / 교육
    if resume.experience or resume.certifications:
        md_parts.append("## 경험 / 활동 / 교육")
        if resume.experience:
            md_parts.append(resume.experience)
            md_parts.append("")
        if resume.certifications:
            md_parts.append("## 자격증") # 또는 다른 제목
            md_parts.append(resume.certifications)
            md_parts.append("")

    return "\n".join(md_parts)

# [View 11] 최종 리포트 생성 페이지 뷰 ---
def generate_final_report_view(request):
    if request.method != "POST":
        return redirect("resume_list")

    # 입력 파라미터 검증
    resume_ids_str   = request.POST.get("resume_ids", "")
    selected_job_ids = request.POST.getlist("selected_jobs")
    if not resume_ids_str or not selected_job_ids:
        messages.error(request, "이력서와 채용 공고를 선택해야 합니다.")
        return redirect("resume_list")

    # ✨ 사용자님이 확인해주신 대로 이력서 ID와 공고 ID를 추출합니다.
    main_resume_id = resume_ids_str.split(",")[0]
    selected_job_id_str = selected_job_ids[0]

    # Resume 객체 로드 및 dict 변환
    try:
        resume_obj = Resume.objects.get(id=main_resume_id)
    except Resume.DoesNotExist:
        messages.error(request, "선택한 이력서를 찾을 수 없습니다.")
        return redirect("resume_list")

    resume_dict = model_to_dict(
        resume_obj,
        fields=[
            "education_level","university","major","gpa",
            "experience_years","job","location",
            "skills","experience","certifications",
        ],
    )
    resume_dict['title'] = resume_obj.title

    # 공고 상세 정보 로드
    job_detail = None
    try:
        job_file_path = settings.BASE_DIR / "Data_Files" / "wanted_detail_improve_20250616.json"
        with open(job_file_path, 'r', encoding='utf-8') as f:
            all_jobs = json.load(f)
        job_data_source = all_jobs.get('postings', all_jobs)
        job_detail = job_data_source.get(selected_job_id_str)
    except Exception as e:
        print(f"공고 정보 조회 중 오류 발생: {e}")

    # Agent 호출 및 리포트 생성 (기존과 동일)
    agent = get_agent_chain(mode="job")
    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    prompt = f"""
    이력서를 기반으로 채용공고와 매칭되는 상세 리포트를 생성합니다.
    ## 이력서 정보
    {resume_dict}
    ## 공고 정보
    {job_detail}
    다음과 같이 write_job_report 도구를 호출해 주세요:
    write_job_report(resume={resume_dict}, job={job_detail})
    중요: resume, job_detail 파라미터에 위의 JSON 데이터를 정확히 전달해 주세요.
    """

    clean_report = "### 분석 실패\nAI 리포트 생성 중 오류가 발생했습니다."
    try:
        result = agent.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        agent_output = result.get("output", "").strip()
        
        clean_report = re.sub(r"^\s*\**Answer\**\s*:?\s*", "", agent_output, flags=re.IGNORECASE).strip()
        
        if clean_report.startswith("**"):
            clean_report = clean_report[2:].strip()

    except Exception as e:
        clean_report = f"### 리포트 생성 오류\n{e}"

    # --- ✨ 리포트 파일 저장 로직 ✨ ---
    try:
        # 1. 디렉토리 이름 생성 (기존과 동일)
        report_dir_name = f"report_set_{main_resume_id}_{selected_job_id_str}"
        report_path = settings.BASE_DIR / "exported_reports" / report_dir_name
        report_path.mkdir(parents=True, exist_ok=True)

        # 2. AI 분석 결과 (report.md) 저장 (기존과 동일)
        with open(report_path / "report.md", "w", encoding="utf-8") as f:
            f.write(clean_report)

        # ✨ 3. 이력서 정보를 Markdown 형식으로 변환하여 저장
        resume_markdown = format_resume_to_markdown(resume_obj)
        with open(report_path / "resume.md", "w", encoding="utf-8") as f:
            f.write(resume_markdown)

        # 4. 공고 정보 (job_posting.json) 저장 (기존과 동일)
        if job_detail:
            with open(report_path / "job_posting.json", "w", encoding="utf-8") as f:
                json.dump(job_detail, f, ensure_ascii=False, indent=2)

        print(f"✅ 최종 리포트가 '{report_path}'에 성공적으로 저장되었습니다.")

    except Exception as e:
        print(f"🚨 최종 리포트 파일 저장 중 오류 발생: {e}")

    # ... (키워드 추출 및 context 생성 로직은 기존과 동일) ...
    # ... (이하 코드는 변경 없음) ...
    matching_keywords = []
    try:
        skills_in_resume = []
        if resume_obj.skills:
            try:
                skills_list = json.loads(resume_obj.skills)
                skills_in_resume = [item['value'] for item in skills_list if 'value' in item]
            except json.JSONDecodeError:
                skills_in_resume = [s.strip() for s in resume_obj.skills.split(',')]
        
        match = re.search(r"(강점|매칭)\s*분석\s*\n+(.*?)(?=\n\n##|\n\n\d+\.|$)", clean_report, re.DOTALL)
        analysis_section = match.group(2) if match else clean_report

        report_text_lower = analysis_section.lower()
        found_keywords = []
        for skill in skills_in_resume:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', report_text_lower):
                found_keywords.append(skill)
        
        matching_keywords = found_keywords[:6]

    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {e}")
        matching_keywords = []

    context = {
        "selected_resume": resume_obj,
        "matching_keywords": matching_keywords,
        "report_markdown": clean_report,
        "current_page": "resume",
        "switch_url_name": "home",
    }

    return render(request, "report_detail.html", context)

# [View 12] 선택된 이력서 ID 세션 저장 뷰 ---
# 이 뷰는 오직 '세션에 이력서 ID를 저장하는' 역할만 합니다.
@require_POST
def set_selected_resumes(request):
    try:
        # 프론트엔드(자바스크립트)에서 보낸 JSON 데이터를 파싱합니다.
        data = json.loads(request.body)
        resume_ids = data.get('resume_ids', [])

        # 유효성 검사: resume_ids가 리스트 형태인지 확인합니다.
        if not isinstance(resume_ids, list):
            return JsonResponse({'status': 'error', 'message': '잘못된 데이터 형식입니다.'}, status=400)
        
        # 세션에 선택된 이력서 ID 리스트를 저장합니다.
        # 이 키('selected_resume_ids')를 다른 뷰에서도 동일하게 사용해야 합니다.
        request.session['selected_resume_ids'] = resume_ids
        
        print(f"✅ 세션에 이력서 ID 저장 성공: {resume_ids}")

        # 성공적으로 저장되었음을 프론트엔드에 알리는 JSON 응답을 보냅니다.
        return JsonResponse({'status': 'ok', 'message': '세션에 저장되었습니다.'})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '잘못된 JSON 요청입니다.'}, status=400)
    except Exception as e:
        print(f"🚨 세션 저장 중 오류 발생: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



# def generate_final_report_view(request):
#     if request.method != "POST":
#         return redirect("resume_list")
#
#     # 1) 입력 파라미터 검증
#     resume_ids_str   = request.POST.get("resume_ids", "")
#     selected_job_ids = request.POST.getlist("selected_jobs")
#     if not resume_ids_str or not selected_job_ids:
#         messages.error(request, "이력서와 채용 공고를 선택해야 합니다.")
#         return redirect("resume_list")
#
#     main_resume_id      = resume_ids_str.split(",")[0]
#     selected_job_id_str = selected_job_ids[0]
#
#     # 2) Resume 객체 로드 및 dict 변환
#     try:
#         resume_obj = Resume.objects.get(id=main_resume_id)
#     except Resume.DoesNotExist:
#         messages.error(request, "선택한 이력서를 찾을 수 없습니다.")
#         return redirect("resume_list")
#
#     resume_dict = model_to_dict(
#         resume_obj,
#         fields=[
#             "education_level","university","major","gpa",
#             "experience_years","job","location",
#             "skills","experience","certifications",
#         ],
#     )
#
#     # 3) 선택된 공고 상세 정보 로드
#     job_detail = None
#     try:
#         job_file = Path(settings.BASE_DIR) / "Data_Files" / "wanted_detail_improve_20250616.json"
#         with open(job_file, "r", encoding="utf-8") as f:
#             all_jobs = json.load(f)
#         job_data = all_jobs.get("postings", all_jobs)
#         job_detail = job_data.get(selected_job_id_str)
#     except Exception as e:
#         print(f"[Error] 공고 정보 로드 실패: {e}")
#
#     # 4) Agent 호출 및 리포트 생성
#     report_markdown = "### 분석 실패\nAI 리포트 생성 중 오류가 발생했습니다."
#     if job_detail:
#         agent = get_agent_chain()
#         if not request.session.session_key:
#             request.session.create()
#         session_id = request.session.session_key
#
#         # JSON 문자열로 직렬화
#         resume_json = json.dumps(resume_dict, ensure_ascii=False)
#         job_json    = json.dumps(job_detail, ensure_ascii=False)
#
#         # 2) JSON 문자열로 직렬화
#         params_json = json.dumps({
#             "resume": resume_dict,
#             "job": job_detail,
#             "depth": "detailed"
#         }, ensure_ascii=False)
#
#         # 3) 프롬프트에서 단일 입력으로 넘기기
#         prompt = (
#             "Thought: 이력서와 채용공고를 분석해 상세 리포트를 생성합니다.\n"
#             f"Action: write_job_report {params_json}\n"
#             "Final Answer:"
#         )
#
#         try:
#             result = agent.invoke(
#                 {"input": prompt},
#                 config={"configurable": {"session_id": session_id}},
#             )
#             report_markdown = result.get("output", "").strip()
#         except Exception as e:
#             report_markdown = f"### 리포트 생성 오류\n{e}"
#
#     # 5) 결과 렌더링
#     resumes = Resume.objects.filter(id__in=resume_ids_str.split(","))
#     context = {
#         "resumes": resumes,
#         "selected_job_id": selected_job_id_str,
#         "job_detail": job_detail,
#         "report_markdown": report_markdown,
#         "current_page": "resume",
#         "switch_url_name": "home",
#     }
#
#     return render(request, "report_detail.html", context)
