from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """
    GraphState는 그래프의 상태를 나타내는 타입입니다.
    """
    context : Annotated[List[Document], "문서 리스트"]
    answer: Annotated[List[Document], "AI의 답변"]
    question : Annotated[str, "사용자의 질문"]
    sql_query: Annotated[str, "SQL 쿼리"]
    binary_score : Annotated[str, "이진 분류 점수"]

def retreive(state : GraphState) -> GraphState:
    """
    상태에서 검색을 수행하고, 검색된 문서를 상태에 추가합니다.
    """
    documents = "검색된 문서"
    return GraphState(context = documents)

def rewrite_query(state: GraphState) -> GraphState:
    # Query Transform: 쿼리 재작성
    documents = "검색된 문서"
    return GraphState(context=documents)


def llm_gpt_execute(state : GraphState) -> GraphState:
    """
    상태에서 LLM을 사용하여 질문에 대한 답변을 생성합니다.
    """
    answer = "LLM이 생성한 답변"
    return GraphState(answer = answer)

def llm_claude_execute(state : GraphState) -> GraphState:
    """
    상태에서 Claude LLM을 사용하여 질문에 대한 답변을 생성합니다.
    """
    answer = "Claude LLM이 생성한 답변"
    return GraphState(answer = answer)

def relevance_check(state : GraphState) -> GraphState:
    """
    상태에서 검색된 문서의 관련성을 확인하고, 이진 분류 점수를 업데이트합니다.
    """
    binary_score = "관련성 점수"
    return GraphState(binary_score = binary_score)

def sum_up(state : GraphState) -> GraphState:
    """
    상태에서 최종 답변을 요약합니다.
    """
    final_answer = "최종 요약된 답변"
    return GraphState(answer = final_answer)


def search_on_web(state : GraphState) -> GraphState:
    """
    상태에서 웹 검색을 수행하고, 검색된 결과를 상태에 추가합니다.
    """
    documents = state['context'] = "기존 문서"
    web_search_results = "웹 검색 결과"
    documents += web_search_results
    return GraphState(context = documents)

def get_table_info(state: GraphState) -> GraphState:
    # Get Table Info: 테이블 정보 가져오기
    table_info = "테이블 정보"
    return GraphState(context=table_info)


def generate_sql_query(state: GraphState) -> GraphState:
    # Make SQL Query: SQL 쿼리 생성
    sql_query = "SQL 쿼리"
    return GraphState(sql_query=sql_query)


def execute_sql_query(state: GraphState) -> GraphState:
    # Execute SQL Query: SQL 쿼리 실행
    sql_result = "SQL 결과"
    return GraphState(context=sql_result)


def validate_sql_query(state: GraphState) -> GraphState:
    # Validate SQL Query: SQL 쿼리 검증
    binary_score = "SQL 쿼리 검증 결과"
    return GraphState(binary_score=binary_score)


def handle_error(state: GraphState) -> GraphState:
    # Error Handling: 에러 처리
    error = "에러 발생"
    return GraphState(context=error)


def decision(state: GraphState) -> GraphState:
    # 의사결정
    decision = "결정"
    # 로직을 추가할 수 가 있고요.

    if state["binary_score"] == "yes":
        return "종료"
    else:
        return "재검색"