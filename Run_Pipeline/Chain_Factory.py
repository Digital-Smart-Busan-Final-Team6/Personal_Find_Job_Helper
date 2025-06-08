# chain_factory.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any

# ---------- 공통 PROMPT 정의 ---------------------------------
prompt_content = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Answer in Korean.

#Context:
{context}
"""

# 이 PROMPT를 build_chain과 build_peft_chain에서 공통으로 사용합니다.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", prompt_content),
    ("human", "{question}"),
])

class ChainFactory:
    @staticmethod
    def build_peft_chain(retriever: Any, llm: Any) -> Any:
        """
        PEFT 모델(HuggingFacePipeline 등)을 직접 RunnableLambda 방식으로 연결한 체인
        retriever → 문서 포맷팅 → PROMPT 삽입 → LLM 호출 → 후처리(generate_answer)
        """
        # 1) 문서들을 하나의 문자열로 합치는 함수 (RunnableLambda에서 사용)
        format_docs = lambda ds: "\n\n".join(d.page_content for d in ds)

        # 2) 실제 LLM을 호출해서 대답만 추출하는 함수
        def generate_answer(full_prompt: str) -> str:
            # full_prompt: 이미 PROMPT 단계를 거친 상태로, "{system}\n{human}" 형태의 문자열입니다.
            result = llm.invoke(full_prompt)
            # "### 답변:" 뒤에 실제 답변이 온다고 가정하고 split
            return result.split("### 답변:")[-1].strip()

        # 3) 체인 구성
        #    - "context" 키: retriever → format_docs
        #    - "question" 키: 사용자가 입력한 질문
        #    - PROMPT 단계: {context}와 {question}을 실제 prompt_content로 채움
        #    - RunnableLambda(generate_answer) 단계: PROMPT 결과를 LLM에 보내고 텍스트만 반환
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | PROMPT | RunnableLambda(generate_answer)

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
        #    - "context" 키: retriever → format_docs
        #    - "question" 키: 사용자가 입력한 질문
        #    - PROMPT 단계: {context}와 {question}을 실제 prompt_content로 채움
        #    - llm 단계: ChatOpenAI 또는 ChatOllama 등의 객체가 .invoke()를 자동 호출
        #    - StrOutputParser 단계: LLM이 반환한 메시지에서 본문만 추출
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | PROMPT | llm | StrOutputParser()

        return chain
