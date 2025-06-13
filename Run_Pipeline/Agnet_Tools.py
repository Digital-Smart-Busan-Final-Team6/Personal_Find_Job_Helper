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
from langchain_community.agent_toolkits import FileManagementToolkit

BASE_DIR = Path(__file__).resolve().parent.parent


class AgentTools:
    # ────────────────────── 공통 PROMPT 정의 ──────────────────────
    prompt_content = """
        당신은 취업과 관련된 답변을 주는 AI 어시스턴트입니다.
        답변은 최대한 취업 전문가처럼 답변하세요.
        1) 먼저 `document_search_response` 툴을 사용해 로컬 문서에서 답을 찾고,
        2) 문서에서 답을 찾지 못했을 때만 `search` 웹 검색 툴을 사용하세요.
        3) 채용 공고에 대한 분석을 요구 할 시 'job_dataframe_analysis' 툴을 사용하세요.
        4) 이미지 생성은 'generate_image' 툴을 사용하세요.
        답변은 한국어로 작성하세요.
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