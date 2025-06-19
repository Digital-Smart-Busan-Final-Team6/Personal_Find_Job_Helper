import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Any
import pandas as pd
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch  # torch 임포트 유지 (SBERT 내부 사용)
from pydantic import BaseModel, Field
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent


class AgentTools:
    # --- Configuration for Job Matching Tool ---
    SBERT_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
    # 가중치 설정 (모든 이력서에 대해 고정)
    WEIGHT_COSINE_SIMILARITY = 0.6
    WEIGHT_EXPERIENCE_MATCH = 0.2
    WEIGHT_TECH_STACK_MATCH = 0.10
    WEIGHT_JOB_ROLE_MATCH = 0.10
    RESUME_JOB_ROLE_KEYWORDS = []
    # 이력서의 직무 관련 키워드 (고정된 가중치를 사용하므로 여기서 직접 정의)
    # 이 부분은 이력서 파싱 후 LLM으로 동적 추출하는 방향으로 고도화될 수 있습니다. 분석", "엔지니어", "개발자", "소프트웨어"]  # 새로 추가/변경: 클래스 변수로 이동
    # --- End Configuration ---

    prompt_content = """
    # 롤 & 톤
    당신은 신뢰할 수 있는 취업 전문가 AI 코치입니다.
    모든 답변은 친절하게 하고 한국어로 작성합니다.
    당신은 채용 공고, 이력서, 면접 정보, 취업 뉴스, 회사 정보 등 취업 관련 질문에만 답변합니다.
    그 외의 질문에는 "죄송합니다. 취업 관련 질문에만 답변할 수 있습니다."라고 답하세요.
    필요하면 표·리스트·예시 코드를 사용해도 좋습니다.

    

    # 워크플로 예시 (Few-shot)
    ## 예시 1 – 문서 검색 성공
    User: "삼성전자 백엔드 신입 연봉 정보 알려줘"
    Assistant:
    - Thought: "채용 공고 문서에 있을 가능성이 높다"
    - Action: document_search {{ "query": "삼성전자 백엔드 신입 연봉" }}

    ## 예시 2 – 문서에 없음 → 웹 검색
    User: "내년 공무원 채용 일정 알려줘"
    Assistant:
    - Thought: "사내 문서엔 없음, 웹 최신 정보 필요"
    - Action: search_web {{ "query": "2025 공무원 채용 일정" }}

    ## 예시 3 – 데이터프레임 통계
    User: "지난 3개월 동안 서울에서 올라온 프론트엔드 평균 연봉 그래프로 보여줘"
    Assistant:
    - Thought: "DataFrame 분석 필요"
    - Action: job_dataframe_analysis {{ "query": "지난 3개월 서울 프론트엔드 평균 연봉 그래프" }}

    ## 예시 4 – 이력서 피드백
    User: "내 이력서에서 개선할 점 알려줘"
    Assistant:
    - Thought: "resume_qa 툴 사용"
    - Action: resume_qa {{ "question": "내 이력서 개선 포인트" }}

    ## 예시 5 – 이력서 기반 공고 추천
    User: "제 이력서에 가장 적합한 채용 공고를 찾아 추천해주세요."
    Assistant:
    - Thought: "이력서 기반 공고 매칭이 필요하므로 find_best_job_match 툴을 사용한다."
    - Action: find_best_job_match {{ "query": "이력서에 가장 적합한 공고 추천" }} # 변경: question -> query로 (이 툴은 사용자 질문에 관계없이 항상 이력서 기반 추천을 하므로)

    대답을 작성할 땐 최종적으로
    **"Answer:"** 섹션에서 사용자에게 보이는 답을 작성하고,
    필요하면 **"Sources:"** 항목에 참고 툴 결과를 간단히 인용한다.
    """

    # ChatPromptTemplate 생성: 올바른 튜플 형태로 지정
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


    # --------------------------------------------------
    @staticmethod
    def get_tools(*, retriever: Any, llm,
                  allowed_tool_names : List[str] = None) -> List[Tool]:
        all_tools = [
            AgentTools._create_retriever_tool(retriever),
            AgentTools._create_search_tool(),
            AgentTools._create_dataframe_tool(llm),
            AgentTools._create_image_generation_tool(),
            AgentTools._create_resume_tool(llm),
            AgentTools._create_job_match_tool(),
            AgentTools._create_analysis_report_tool(llm)
        ]

        # ▒ file-management 툴 (여러 개) 추가
        file_tools = AgentTools._create_file_management_tool()
        if file_tools:
            all_tools.extend(file_tools)
        tools = [t for t in all_tools if t is not None]

        if allowed_tool_names is not None:
            tools = [t for t in tools if t.name in allowed_tool_names]

        return tools

    # ────────────────────── tool builders ──────────────────────
    # 1) 웹 검색
    @staticmethod
    def _search_web(query: str) -> str:
        return TavilySearchResults(k=6).run(query)

    @staticmethod
    def _create_search_tool() -> Tool:
        return Tool.from_function(
            func=AgentTools._search_web,
            name="search_web",
            description="문서에 없으면 최신 웹 검색을 수행한다.",
        )

    # 2) RAG 리트리버
    @staticmethod
    def _create_retriever_tool(retriever: Any) -> Optional[Tool]:
        if retriever is None:
            logging.warning("Retriever is None – document_search 툴 비활성화.")
            return None

        return create_retriever_tool(
            retriever=retriever,
            name="document_search",
            description="항상 먼저 호출해야 하는 툴로, 벡터DB에서 관련 문서를 검색한다."
        )

    # 3) DataFrame 분석
    @staticmethod
    def _create_dataframe_tool(llm) -> Optional[Tool]:
        if llm is None:
            logging.warning("LLM is None – dataframe 툴 비활성화.")
            return None

        data_path = BASE_DIR / "Data_Files" / "wanted_detail_improve_20250616.json"
        if not data_path.exists():
            logging.warning("DataFrame 파일을 찾을 수 없음 – dataframe 툴 비활성화.")
            return None

        df = pd.read_json(data_path).T

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            prefix="너는 채용공고 데이터 분석 전문가다.",
        )

        return Tool.from_function(
            func=lambda q: pandas_agent.run(q),
            name="job_dataframe_analysis",
            description="채용공고 DataFrame을 이용해 통계·시각화를 수행한다.",
        )

    # 4) 이미지 생성
    @staticmethod
    def _create_image_generation_tool() -> Optional[Tool]:
        """
        DALL-E 이미지 생성 툴을 생성합니다.
        현재는 사용하지 않지만, 필요시 활성화할 수 있습니다.
        """
        if not os.getenv("OPENAI_API_KEY"):
            logging.warning("DALL-E API 키가 설정되지 않았습니다. 이미지 생성 툴 비활성화.")
            return None

        dalle_tool = DallEAPIWrapper(
            model="dall-e-3",  # 생성 모델
            size="1024x1024",
            quality="standard",  # 품질 설정 hd도 있음
            n=1  # 생성할 이미지 개수
        )
        return Tool.from_function(
            func=dalle_tool.run,
            name="generate_image",
            description="이미지를 생성합니다.",
        )

    # 5) 파일 관리 툴
    @staticmethod
    def _create_file_management_tool() -> Optional[List[Tool]]:
        """
        파일 시스템을 다루는 여러 Tool(write_file, read_file …)을 돌려준다.
        """
        working_directory = BASE_DIR / "Data_Files"

        file_tools = FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=[
                "write_file", "read_file",
                "list_directory", "move_file",
            ],
        ).get_tools()

        return file_tools

    # 6) 이력서 QA 툴
    @staticmethod
    def _create_resume_tool(llm) -> Tool:
        """
        이력서를 LLM 컨텍스트에 넣어 질의-응답해 주는 툴.
        """
        from accounts.models import Resume
        data_path = BASE_DIR / "Data_Files" / "resume.json"
        try:
            with open(data_path, encoding="utf-8") as f:
                resume = json.load(f)
        except FileNotFoundError:
            logging.error(f"Resume file not found at {data_path}. Resume QA tool will not function.")
            return Tool.from_function(
                func=lambda q: "이력서 파일을 찾을 수 없어 이력서 관련 질문에 답변할 수 없습니다.",
                name="resume_qa",
                description="이력서 파일이 없을 경우 답변할 수 없습니다."
            )
        except json.JSONDecodeError:
            logging.error(f"Error decoding resume JSON from {data_path}. Resume QA tool will not function.")
            return Tool.from_function(
                func=lambda q: "이력서 파일 형식이 올바르지 않아 이력서 관련 질문에 답변할 수 없습니다.",
                name="resume_qa",
                description="이력서 파일 형식이 올바르지 않을 경우 답변할 수 없습니다."
            )

        # ▶ 필요한 형태(평문)로 합치기
        if isinstance(resume, dict):
            resume_text = "\n".join(f"{k}: {v}" for k, v in resume.items())
        elif isinstance(resume, list):
            lines = []
            for idx, item in enumerate(resume, 1):
                if isinstance(item, dict):
                    lines.append(f"--- 항목 {idx} ---")
                    lines.extend(f"{k}: {v}" for k, v in item.items())
                else:
                    lines.append(str(item))
            resume_text = "\n".join(lines)
        else:
            resume_text = str(resume)

        def _answer_about_resume(question: str) -> str:
            prompt = f"""너는 구직자의 이력서 코치야.
            이력서 원문:
            ---
            {resume_text}
            ---
            사용자의 질문: {question}
            """
            return llm.invoke(prompt).content

        return Tool.from_function(
            func=_answer_about_resume,
            name="resume_qa",
            description="이력서 내용에 대해 질문하면, 조언이나 답변을 해 줍니다.",
        )

    # 7) 이력서-공고 매칭 툴  ← 새로 추가
    @staticmethod
    def _create_job_match_tool() -> Tool:
        """
        사용자의 이력서 데이터를 외부에서 받아서
        가장 적합한 공고 상위 5개를 Markdown 표로 반환합니다.
        """
        DATA_DIR = BASE_DIR / "Data_Files"
        JOB_POST_FILE = DATA_DIR / "wanted_detail_improve_20250616.json"
        SBERT_MODEL_NAME = AgentTools.SBERT_MODEL_NAME

        # 모델 캐시
        import functools
        @functools.lru_cache(maxsize=1)
        def _load_model() -> SentenceTransformer:
            return SentenceTransformer(SBERT_MODEL_NAME)

        @functools.lru_cache(maxsize=1)
        def _load_json(path: Path) -> Any:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        # 이력서 dict를 텍스트/년수/스택 리스트로 변환
        def _combine_resume(data: dict) -> tuple[str, int, list[str]]:
            text = (
                f"학력: {data.get('education_level', '')} / {data.get('university', '')} {data.get('major', '')} (GPA {data.get('gpa', '')})\n"
                f"경력: {data.get('experience_years', 0)}년\n"
                f"보유 기술: {data.get('skills', '')}\n"
                f"희망 직무: {data.get('job', '')}\n"
                f"희망 근무지: {data.get('location', '')}\n"
                f"기타 경험: {data.get('experience', '')}\n"
                f"자격증/어학: {data.get('certifications', '')}"
            ).strip()
            years = data.get('experience_years', 0)
            stacks = [s.strip().lower() for s in data.get('skills', '').split(',') if s.strip()]
            # 직무 키워드
            AgentTools.RESUME_JOB_ROLE_KEYWORDS = [
                kw.strip().lower() for kw in data.get('job', '').split(',') if kw.strip()
            ]
            return text, years, stacks

        # 공고 dict를 텍스트로 변환
        def _combine_job(job: dict) -> str:
            return (
                f"제목: {job.get('제목', '')}\n"
                f"회사 소개: {job.get('회사 소개', '')}\n"
                f"주요 업무: {' '.join(job.get('주요 업무', []))}\n"
                f"자격 요건: {' '.join(job.get('자격 요건', []))}\n"
                f"우대 사항: {' '.join(job.get('우대 사항', []))}\n"
                f"지역: {', '.join(job.get('지역', []))}"
            ).strip()

        # 실제 매칭 로직: resume 파라미터 없이, 바깅된 resume 변수를 바로 사용
        def _find_best_job_match(
                resume: Any,
        ) -> str:
            if isinstance(resume, str):
                try:
                    resume = json.loads(resume)
                except json.decoder.JSONDecodeError:
                    return "Error: 이력서 데이터가 올바른 JSON 형식이 아닙니다."

            # 1) 바인딩된 resume 변수를 쓴다
            resume_data = resume

            # 2) 이력서 → 임베딩
            resume_text, resume_years, resume_stacks = _combine_resume(resume_data)
            model = _load_model()
            resume_emb = model.encode(resume_text, convert_to_tensor=True)

            # 3) 공고들 로드 및 점수 계산
            job_posts = _load_json(JOB_POST_FILE)
            scores: list[tuple[float, str, str]] = []
            for job_id, job in job_posts.items():
                job_text = _combine_job(job)
                job_emb = model.encode(job_text, convert_to_tensor=True)

                # 의미적 유사도
                cosine_score = util.cos_sim(resume_emb, job_emb).item()

                # 경력 점수
                min_exp = job.get('요구 최소 경력', 0)
                max_exp = job.get('요구 최대 경력', 999)
                if min_exp <= resume_years <= max_exp:
                    exp_score = 1.0
                elif resume_years > max_exp:
                    exp_score = 0.5
                else:
                    exp_score = 0.0

                # 기술 스택 점수
                job_stacks = [s.strip().lower() for s in job.get('기술 스택', [])]
                if job_stacks:
                    matched = sum(1 for s in resume_stacks if s in job_stacks)
                    tech_score = matched / len(job_stacks)
                else:
                    tech_score = 0.5

                # 직군 점수
                job_role = job.get('직군', '').lower()
                job_jobs = [d.lower() for d in job.get('직무', [])]
                role_score = 1.0 if any(
                    k in job_role or any(k in jj for jj in job_jobs)
                    for k in AgentTools.RESUME_JOB_ROLE_KEYWORDS
                ) else 0.0

                # 최종 점수
                total = (
                        cosine_score * AgentTools.WEIGHT_COSINE_SIMILARITY +
                        exp_score * AgentTools.WEIGHT_EXPERIENCE_MATCH +
                        tech_score * AgentTools.WEIGHT_TECH_STACK_MATCH +
                        role_score * AgentTools.WEIGHT_JOB_ROLE_MATCH
                )
                scores.append((total, job_id, job.get('제목', '')))

            # 4) 상위 5개 선택 및 Markdown 표로 반환
            top5 = sorted(scores, key=lambda x: x[0], reverse=True)[:5]
            if not top5:
                return "채용 공고가 없습니다."
            lines = ["|순위|공고 ID|제목|적합도|", "|---|---|---|---|"]
            for rank, (score, jid, title) in enumerate(top5, start=1):
                lines.append(f"|{rank}|{jid}|{title}|{score:.4f}|")
            return "\n".join(lines)

        return Tool.from_function(
            func=_find_best_job_match,
            name="find_best_job_match",
            description="""
                    사용자 이력서를 받아 가장 적합한 공고 상위 5개를 표 형태로 반환합니다.
                    사용법: 사용자로부터 받은 이력서 JSON 데이터를 resume 파라미터에 전달하세요.
                    예: find_best_job_match(resume={"education_level": "대졸", "skills": "Python, Django", ...})
                    """,
        )

    @staticmethod
    def _create_analysis_report_tool(llm) -> StructuredTool:
        """
        뷰에서 전달된 resume, job 두 개 인자를 받아,
        내부 코어 함수에 전달하는 래퍼를 StructuredTool로 등록합니다.
        """

        # 입력 스키마 정의
        class JobAnalysisInput(BaseModel):
            resume: Dict[str, Any] = Field(description="사용자의 이력서 JSON 데이터")
            job: Dict[str, Any] = Field(description="채용공고 JSON 데이터")

        def _write_job_report(resume: Dict[str, Any], job: Dict[str, Any]) -> str:
            try:
                # Resume 텍스트 포맷 (안전한 접근)
                resume_text = (
                    f"교육: {resume.get('education_level', 'N/A')} / "
                    f"{resume.get('university', 'N/A')} {resume.get('major', 'N/A')} "
                    f"(GPA {resume.get('gpa', 'N/A')})\n"
                    f"경력: {resume.get('experience_years', 0)}년\n"
                    f"보유 기술: {resume.get('skills', 'N/A')}\n"
                    f"희망 직무: {resume.get('job', 'N/A')}\n"
                    f"희망 근무지: {resume.get('location', 'N/A')}\n"
                    f"기타 경험: {resume.get('experience', 'N/A')}\n"
                    f"자격증/어학: {resume.get('certifications', 'N/A')}"
                )

                # Job 텍스트 포맷 (안전한 리스트 처리)
                def safe_join(data, default="N/A"):
                    if isinstance(data, list):
                        return ' '.join(str(item) for item in data) if data else default
                    elif isinstance(data, str):
                        return data
                    else:
                        return default

                job_text = (
                    f"제목: {job.get('제목', 'N/A')}\n"
                    f"회사명: {job.get('회사명', 'N/A')}\n"
                    f"포지션 설명: {job.get('포지션 상세 설명', 'N/A')[:200]}...\n"  # 너무 길면 자름
                    f"주요 업무: {safe_join(job.get('주요 업무'))}\n"
                    f"자격 요건: {safe_join(job.get('자격 요건'))}\n"
                    f"우대 사항: {safe_join(job.get('우대 사항'))}\n"
                    f"근무지: {job.get('근무지', 'N/A')} {job.get('지역', 'N/A')}\n"
                    f"최소 경력: {job.get('요구 최소 경력', 'N/A')}년"
                )

                # LLM 프롬프트 구성
                prompt = f"""
    당신은 취업 컨설턴트이자 채용 공고 분석 전문가입니다.

    ## Resume 정보
    {resume_text}

    ## Job Posting 정보
    {job_text}

    ### 리포트에 포함할 내용
    1. 공고 요약
    2. 요구 사항 표
    3. 매칭 분석 (강점 / 부족 / 중립)
    4. 개선 액션 플랜
    5. 예상 면접 질문 5개 (키워드)

    ### 출력 형식
    * Markdown 형식
    * 분량: 약 1200~1600자
    """.strip()

                # LLM 호출 및 안전한 응답 처리
                response = llm.invoke(prompt)

                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
                else:
                    return "리포트 생성 중 오류가 발생했습니다. 응답을 받을 수 없습니다."

            except Exception as e:
                return f"리포트 생성 중 오류가 발생했습니다: {str(e)}"

        # StructuredTool 등록
        return StructuredTool.from_function(
            func=_write_job_report,
            name="write_job_report",
            description=(
                "사용자의 이력서와 채용 공고를 분석하여 매칭 리포트를 작성합니다. "
                "resume과 job 두 개의 파라미터를 모두 전달해야 합니다."
            ),
            args_schema=JobAnalysisInput
        )