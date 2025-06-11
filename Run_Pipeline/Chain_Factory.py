# chain_factory.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any

class ChainFactory:
    # ---------- 공통 PROMPT 정의 ---------------------------------
    prompt_content = """
    당신은 취업과 관련된 답변을 주는 AI 어시스턴트입니다.
    답변은 최대한 취업 전문가처럼 답변하세요.
    1) 먼저 `document_search_response` 툴을 사용해 로컬 문서에서 답을 찾고,
    2) 문서에서 답을 찾지 못했을 때만 `search` 웹 검색 툴을 사용하세요.

    답변은 한국어로 작성하세요.
    """

    # ChatPromptTemplate 생성: 올바른 튜플 형태로 지정
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    @staticmethod
    def build_peft_chain(retriever: Any, llm: Any) -> Any:
        """
        PEFT 모델(HuggingFacePipeline 등)을 직접 RunnableLambda 방식으로 연결한 체인
        retriever → 문서 포맷팅 → PROMPT 삽입 → LLM 호출 → 후처리(generate_answer)
        """
        # 1) 문서들을 하나의 문자열로 합치는 함수
        format_docs = lambda ds: "\n\n".join(d.page_content for d in ds)

        # 2) 실제 LLM을 호출해서 대답만 추출하는 함수
        def generate_answer(full_prompt: str) -> str:
            # full_prompt: PROMPT 단계를 거친 상태
            result = llm.invoke(full_prompt)
            # '### 답변:' 뒤에 실제 답변이 온다고 가정하고 split
            return result.split("### 답변:")[-1].strip()

        # 3) 체인 구성
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | ChainFactory.PROMPT | RunnableLambda(generate_answer)

        return chain

    @staticmethod
    def build_chain(retriever: Any, llm: Any) -> Any:
        """
        비-PEFT(예: OpenAI, Ollama) LLM을 연결한 간단한 체인
        retriever → 문서 포맷팅 → PROMPT 삽입 → llm.invoke → StrOutputParser로 최종 문자열 리턴
        """
        # 1) 문서들을 하나의 문자열로 합치는 함수
        format_docs = lambda ds: "\n\n".join(d.page_content for d in ds)

        # 2) 체인 구성
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | ChainFactory.PROMPT | llm | StrOutputParser()

        return chain
