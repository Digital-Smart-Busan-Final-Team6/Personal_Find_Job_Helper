"""
Agent 툴을 관리합니다.
"""
import logging
import os
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

BASE_DIR = Path(__file__).resolve().parent.parent

class AgentTools:
    # ────────────────────── 공통 PROMPT 정의 ──────────────────────
    prompt_content = """
    # 롤 & 톤
    당신은 신뢰할 수 있는 취업 전문가 AI 코치입니다.
    모든 답변은 친절하지만 간결한 한국어로 작성하세요.
    필요하면 표·리스트·예시 코드를 사용해도 좋습니다.

    # 툴 사용 규칙
    1. 채용 공고, 취업 뉴스, 면접 정보 등 **문서**에서 답을 찾을 수 있을 것 같으면
       반드시 `document_search` 툴을 먼저 호출한다.
    2. `document_search`가 결과를 0개 반환하거나(embeddings empty) 내용이 질문과 무관하면
       보강 정보가 필요하다고 판단하고 `search_web` 툴을 호출한다.
    3. 사용자가 “평균”, “통계”, “그래프”, “필터” 같은 키워드로
       **데이터 분석**을 요구하면 `job_dataframe_analysis` 툴을 호출한다.
    4. 사용자가 “이미지 만들어”, “로고 그려” 같이 시각 생성 요청을 하면 `generate_image`.
    5. “내 이력서” 또는 “내 경력”을 직접 언급하면 `resume_qa`.
    6. 위 조건에 맞지 않으면 툴을 사용하지 않고 자체 지식으로 답한다.

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
    def get_tools(*, retriever: Any, llm) -> List[Tool]:
        tools = [
            AgentTools._create_search_tool(),
            AgentTools._create_retriever_tool(retriever),
            AgentTools._create_dataframe_tool(llm),
            AgentTools._create_image_generation_tool(),
            AgentTools._create_resume_tool(llm)
        ]

        # ▒ file-management 툴 (여러 개) 추가
        file_tools = AgentTools._create_file_management_tool()
        if file_tools:                      # ❷ None 체크
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
            description="업로드된 채용 공고 벡터 DB에서 관련 내용을 찾는다.",
        )

    # 3) DataFrame 분석
    @staticmethod
    def _create_dataframe_tool(llm) -> Optional[Tool]:
        if llm is None:
            logging.warning("LLM is None – dataframe 툴 비활성화.")
            return None

        data_path = BASE_DIR / "Data_Files" / "wanted_detail_improve_20250604.json"
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
            model="dall-e-3", # 생성 모델
            size="1024x1024",
            quality="standard", # 품질 설정 hd도 있음
            n=1 # 생성할 이미지 개수
        )
        return Tool.from_function(
            func=dalle_tool.run,
            name="generate_image",
            description="이미지를 생성합니다.",
        )

    @staticmethod
    def _create_file_management_tool() -> Optional[List[Tool]]:  # ❸ 타입 변경
        """
        파일 시스템을 다루는 여러 Tool(write_file, read_file …)을 돌려준다.
        """
        working_directory = BASE_DIR / "Data_Files"

        # 필요 없다면 selected_tools에서 빼거나 함수 전체를 주석 처리해도 OK
        file_tools = FileManagementToolkit(
            root_dir= str(working_directory),
            selected_tools=[
                "write_file", "read_file",
                "list_directory", "move_file",
            ],
        ).get_tools()

        return file_tools

    @staticmethod
    def _create_resume_tool(llm) -> Tool:
        """
        이력서를 LLM 컨텍스트에 넣어 질의-응답해 주는 툴.
        """
        data_path = BASE_DIR / "Data_Files" / "resume.json"
        with open(data_path, encoding="utf-8") as f:
            resume = json.load(f)

        # ▶ 필요한 형태(평문)로 합치기
        if isinstance(resume, dict):
            resume_text = "\n".join(f"{k}: {v}" for k, v in resume.items())
        elif isinstance(resume, list):
            # 리스트 요소가 dict면 합치고, 아니면 str로 변환
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

            return llm.invoke(prompt).content  # LLM 응답 텍스트

        return Tool.from_function(
            func=_answer_about_resume,
            name="resume_qa",
            description="이력서 내용에 대해 질문하면, 조언이나 답변을 해 줍니다.",
        )