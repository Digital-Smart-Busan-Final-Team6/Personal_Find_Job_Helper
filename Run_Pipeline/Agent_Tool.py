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
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch  # torch 임포트 유지 (SBERT 내부 사용)

BASE_DIR = Path(__file__).resolve().parent.parent


class AgentTools:
    # --- Configuration for Job Matching Tool ---
    SBERT_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
    # 가중치 설정 (모든 이력서에 대해 고정)
    WEIGHT_COSINE_SIMILARITY = 0.6
    WEIGHT_EXPERIENCE_MATCH = 0.2
    WEIGHT_TECH_STACK_MATCH = 0.15
    WEIGHT_JOB_ROLE_MATCH = 0.05
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

    # 툴 사용 규칙
    1. 채용 공고, 취업 뉴스, 면접 정보, 회사 정보 등 **문서**에서 답을 찾을 수 있을 것 같으면
       반드시 `document_search` 툴을 먼저 호출한다.
    2. `document_search`가 결과를 0개 반환하거나(embeddings empty) 내용이 질문과 무관하면
       보강 정보가 필요하다고 판단하고 `search_web` 툴을 호출한다.
    3. 사용자가 “평균”, “통계”, “그래프”, “필터” 같은 키워드로
       **데이터 분석**을 요구하면 `job_dataframe_analysis` 툴을 호출한다.
    4. 사용자가 “이미지 만들어”, “로고 그려” 같이 시각 생성 요청을 하면 `generate_image`.
    5. “내 이력서” 또는 “내 경력”을 직접 언급하면 `resume_qa`.
    6. **"가장 적합한 공고", "내 이력서에 맞는 공고", "추천 공고"**와 같은 키워드로
       이력서와 공고 매칭을 요청하면 `find_best_job_match` 툴을 호출한다.
    7. 사용자가 공고 를 지정하며 “상세 분석”, “리포트” 등을 요청하면 document_search로 공고를 검색 후, write_job_report 툴을 호출한다.
    예) "제일 적합도가 높은 공고로 리포트 작성해 줘"
    8. 위 조건에 맞지 않으면 툴을 사용하지 않고 자체 지식으로 답한다.

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

    from langchain_core.prompts import ChatPromptTemplate

    react_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    🔹 역할 & 톤
    당신은 신뢰할 수 있는 취업 전문가 AI 코치입니다.  
    한국어로 친절하게 답변하되, 채용/이력서/면접/회사 정보/채용 관련 기사 같은 질문에만 응답합니다.  

    {tools}          
    (툴 이름 목록: {tool_names}) 

    🔹 Tool 사용 규칙
    1. 문서 기반 답이 가능할 것 같으면 **반드시** `document_search`를 먼저 호출한다.
    2. 결과가 0개이거나 질문과 무관하면 `search_web`을 호출한다.
    3. '평균·그래프·통계' 등 데이터 분석 → `job_dataframe_analysis`
    4. '내 이력서' 직접 언급 → `resume_qa`
    5. '가장 적합한 공고 추천' → `find_best_job_match`
    6. 그 밖엔 Tool 없이 자체 지식으로 답한다.

    🔹 포맷 (★ ReAct 필수)
    - <모델의 내부 사고>는 **Thought:** 로 시작  
    - Tool 호출은 **Action:** Tool이름 {{ JSON 인자 }}  
    - Observation 다음 다시 Thought → Action … 또는 **Final Answer:** 로 종료

    예시)
    Thought: "채용 공고에 있을 것 같다"
    Action: document_search {{ "query": "삼성전자 백엔드 신입 연봉" }}
    Observation: "연봉 5,400만원~ …"
    Thought: "바로 답할 수 있다"
    Final Answer: "삼성전자 백엔드 신입 초봉은 …"
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
            # {tool_names}를 위에서 이미 사용했으므로 여기엔 더 안 넣어도 됨
        ]
    )

    # --------------------------------------------------
    @staticmethod
    def get_tools(*, retriever: Any, llm) -> List[Tool]:
        tools = [
            AgentTools._create_search_tool(),
            AgentTools._create_retriever_tool(retriever),
            AgentTools._create_dataframe_tool(llm),
            AgentTools._create_image_generation_tool(),
            AgentTools._create_resume_tool(llm),
            AgentTools._create_job_match_tool(retriever),
            AgentTools._create_analysis_report_tool(llm)
        ]

        # ▒ file-management 툴 (여러 개) 추가
        file_tools = AgentTools._create_file_management_tool()
        if file_tools:
            tools.extend(file_tools)

        return [t for t in tools if t is not None]

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
    def _create_job_match_tool(retriever: Any) -> Tool:
        """
        사용자의 이력서(JSON)와 채용 공고(JSON)를 읽어
        가장 적합한 공고 1~N개를 추천한다.
        query 인자는 외부에서 아무 문자열이나 넘어오므로
        실제로는 무시하거나 Top-K 개수 정도만 파싱해 써도 된다.
        """
        DATA_DIR = BASE_DIR / "Data_Files"
        JOB_POST_FILE = DATA_DIR / "wanted_detail_improve_20250616.json"
        RESUME_FILE = DATA_DIR / "resume.json"
        SBERT_MODEL_NAME = AgentTools.SBERT_MODEL_NAME  # 클래스 상수 재사용

        # ── ① 공통 리소스: 모델과 JSON을 한 번만 로드 ──────────────────
        import functools
        @functools.lru_cache(maxsize=1)
        def _load_model() -> SentenceTransformer:
            return SentenceTransformer(SBERT_MODEL_NAME)

        @functools.lru_cache(maxsize=1)
        def _load_json(path: Path) -> Any:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        # ── ② 점수 계산 유틸리티 ──────────────────────────────────────
        def _combine_resume(resume: dict) -> tuple[str, int, list[str]]:

            # 1) 프롬프트용 텍스트 한데 묶기
            text = (
                f"학력: {resume.get('university', '')} {resume.get('major', '')} "
                f"{resume.get('education_status', '')} 학점 {resume.get('gpa', '')}. "
                f"희망 직무: {', '.join(resume.get('job_interests', []))}. "
                f"희망 지역: {', '.join(resume.get('location_interests', []))}. "
                f"보유 기술: {resume.get('skills', '')}."
            ).strip()

            # 2) 경력 기간(년) – 새 스키마에 없으므로 0 으로 두거나 추가 필드가 있으면 파싱
            years = 0  # resume.get("experience_years", 0)  같은 식으로 추후 보강 가능

            # 3) 기술 스택 리스트 추출
            stacks = [s.strip().lower() for s in resume.get("skills", "").split(",") if s.strip()]
            AgentTools.RESUME_JOB_ROLE_KEYWORDS = [kw.lower() for kw in resume.get("job_interests", [])]
            return text, years, stacks

        def _combine_job(job: dict) -> str:
            return (
                f"제목: {job.get('제목', '')}. "
                f"회사 소개: {job.get('회사 소개', '')}. "
                f"주요 업무: {' '.join(job.get('주요 업무', []))}. "
                f"자격 요건: {' '.join(job.get('자격 요건', []))}. "
                f"우대 사항: {' '.join(job.get('우대 사항', []))}."
                f"지역: {', '.join(job.get('지역', []))}. "
            ).strip()

        # ── ③ 실제로 호출되는 함수(Action) ───────────────────────────
        def _find_best_job_match(query: str = "") -> str:
            """LangChain Tool로 쓰일 함수. query 값은 사실상 사용하지 않음."""
            # 데이터 로드
            job_posts: dict = _load_json(JOB_POST_FILE)
            resume_list: list = _load_json(RESUME_FILE)  # resume.json은 리스트라고 가정
            if not resume_list:
                return "이력서가 비어 있어 매칭을 수행할 수 없습니다."

            resume = resume_list[0]  # 첫 번째 이력서
            resume_text, resume_years, resume_stacks = _combine_resume(resume)

            # 모델 & 이력서 임베딩
            model = _load_model()
            resume_emb = model.encode(resume_text, convert_to_tensor=True)

            scores = []
            for job_id, job in job_posts.items():
                job_text = _combine_job(job)
                job_emb = model.encode(job_text, convert_to_tensor=True)

                # ① 의미적 유사성
                cosine_score = util.cos_sim(resume_emb, job_emb).item()

                # ② 경력
                min_exp = job.get("요구 최소 경력", 0)
                max_exp = job.get("요구 최대 경력", 999)
                if min_exp <= resume_years <= max_exp:
                    exp_score = 1.0
                elif resume_years > max_exp:
                    exp_score = 0.5
                else:
                    exp_score = 0.0

                # ③ 기술 스택
                job_stacks = [s.strip().lower() for s in job.get("기술 스택", [])]
                if job_stacks:
                    matched = sum(1 for s in resume_stacks if s in job_stacks)
                    tech_score = matched / len(job_stacks)
                else:
                    tech_score = 0.5  # 공고에 명시가 없으면 중립

                # ④ 직군
                job_role = job.get("직군", "").lower()
                job_jobs = [d.lower() for d in job.get("직무", [])]
                role_score = (
                    1.0
                    if any(k in job_role or any(k in jj for jj in job_jobs)
                           for k in AgentTools.RESUME_JOB_ROLE_KEYWORDS)
                    else 0.0
                )

                total = (
                        cosine_score * AgentTools.WEIGHT_COSINE_SIMILARITY +
                        exp_score * AgentTools.WEIGHT_EXPERIENCE_MATCH +
                        tech_score * AgentTools.WEIGHT_TECH_STACK_MATCH +
                        role_score * AgentTools.WEIGHT_JOB_ROLE_MATCH
                )

                scores.append((total, job_id, job["제목"]))

            # 상위 10개 정렬
            top = sorted(scores, reverse=True)[:5]
            if not top:
                return "채용 공고가 없습니다."

            # 출력 포맷: Markdown 표
            lines = ["|순위|공고 ID|제목|적합도|", "|---|---|---|---|"]
            for rank, (score, jid, title) in enumerate(top, 1):
                lines.append(f"|{rank}|{jid}|{title}|{score:.4f}|")
            return "\n".join(lines)

        # ── ④ LangChain Tool 래핑 ────────────────────────────────────
        return Tool.from_function(
            func=_find_best_job_match,
            name="find_best_job_match",
            description=(
                "사용자 이력서(resume.json)와 채용 공고(wanted_detail_improve_20250616.json)를 비교해 "
                "가장 적합한 공고 상위 5개를 표 형태로 반환한다."
            ),
        )

    @staticmethod
    def _create_analysis_report_tool(llm):
        DATA_DIR = BASE_DIR / "Data_Files"
        with open(DATA_DIR / "resume.json", "r", encoding="utf-8") as f:
            resume = json.load(f)

        resume_my = resume[0]

        def _write_job_report(
                depth: str = "detailed",  # 'quick' or 'detailed'
                fmt: str = "md") -> str:
            # 3) LLM 프롬프트
            prompt = f"""
            # 역할
            당신은 취업 컨설턴트이자 채용 공고 분석 전문가입니다.

            ## Resume
            {resume_my}

            # 포함 항목
            1. 공고 요약
            2. 요구 사항 표
            3. 매칭 분석 (강점 / 부족 / 중립)
            4. 개선 액션 플랜
            5. 예상 면접 질문 5개 (키워드)

            # 형식
            * Markdown
            * 분량: {"약 600자" if depth == "quick" else "약 1200~1600자"}
            """

            report_md = llm.invoke(prompt).content.strip()

            # 4) 저장 (원하면 PDF 변환도)
            from datetime import datetime
            p = Path("Reports") / f"job_report_{datetime.now():%Y%m%d_%H%M%S}.md"
            p.parent.mkdir(exist_ok=True)
            p.write_text(report_md, encoding="utf-8")

            if fmt in {"pdf", "docx"} and shutil.which("pandoc"):
                out = p.with_suffix(f".{fmt}")
                import subprocess
                subprocess.run(["pandoc", str(p), "-o", str(out)])
                p = out

            return str(p)

        return Tool.from_function(
            func=_write_job_report,
            name="write_job_report",
            description="job_id에 해당하는 공고를 RAG로 불러와 상세 분석 리포트를 작성한다.",
        )
