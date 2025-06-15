# main_pipeline.py

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentCallbacks, AgentStreamParser
from Agent_Tool import *
from Document_Loader import DocumentLoader
from Document_Splitter import DocumentSplitter
from Embedding_DB import EmbeddingDB
from Env_Loader import EnvLoader
from LLM_Factory import LLMFactory
from Retriever_Builder import RetrieverBuilder


def main(return_chain_only: bool = False):
    # ① 환경 변수 로드 (로컬 우선)
    EnvLoader.load_local()

    # ② 파라미터 설정
    BASE_DIR = Path(__file__).resolve().parent.parent
    if return_chain_only:
        kind = "json"
        file_path = BASE_DIR / os.getenv("DATA_PATH")
        chunk_size = 1000
        overlap_size = 50
        device = "mps"
        persist_dir = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        retriever_mode = 1
        k = 3
        engine_num = 1
        backend_num = 1
    else:
        # kind           = input("파일 종류(json/txt/all): ").strip().lower()
        # file_path      = '../Data_Files'
        # chunk_size     = int(input("청크 사이즈(기본 1000): ") or 1000)
        # overlap_size   = int(input("오버랩 사이즈(기본 50): ") or 50)
        # persist_dir    = '../Data_Files'
        # device_choice  = int(input("디바이스(1:mps/2:cuda/3:cpu): ") or 1)
        # device         = {1: "mps", 2: "cuda", 3: "cpu"}[device_choice]
        # retriever_mode = int(input("리트리버 모드(1 vec / 2 bm25 / 3 ensemble): ") or 1)
        # k              = int(input("top-k 개수: ") or 3)
        # engine_num     = int(input("LLM 모델 번호(1~4): ") or 1)
        # backend_num    = int(input("LLM 백엔드(1:OpenAI/2:Ollama/3:HF): ") or 1)
        # 빠른 실행 위해 고정값 사용
        kind = "json"
        file_path = BASE_DIR / os.getenv("DATA_PATH")
        chunk_size = 1000
        overlap_size = 50
        persist_dir = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        device = "mps"
        retriever_mode = 3
        k = 10
        engine_num = 1
        backend_num = 1

    # ③ 문서 로드
    loader = DocumentLoader(file_path, kind)  # 일반 파일들을 Document형태로 변환해서 다 불러옴
    docs = loader._load_json_files()

    # ④ 문서 분할
    splitter = DocumentSplitter(chunk_size=chunk_size, overlap=overlap_size)  # 이미 분할된 문서가 있다면 로드하고 아니면 새로 스플릿함
    chunks = splitter.split(docs, cache_dir=file_path)

    # ⑤ 임베딩 & DB 생성/로드
    embed_db = EmbeddingDB(model_name="nlpai-lab/KURE-v1", device=device,
                           persist_dir=persist_dir)  # 이미 임베딩된 DB가 있다면 로드하고 아니면 새로 임베딩함
    db = embed_db.get_or_create_db(chunks)

    # ⑦ LLM 로드
    llm = LLMFactory.load_llm(engine=engine_num, backend=backend_num, device=device)  # LLM 로드

    # ⑥ 리트리버 빌드
    retriever_builder = RetrieverBuilder(
        db=db,
        docs=chunks,
        k=k,
        mode=retriever_mode
    )
    retriever = retriever_builder.build()

    tools = AgentTools.get_tools(retriever=retriever, llm=llm)  # 도구 목록 생성 (retriever 포함)

    agent = create_tool_calling_agent(llm, tools, AgentTools.PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 도구 호출 시 실행되는 콜백 함수입니다.
    def tool_callback(tool) -> None:
        print("<<<<<<< 도구 호출 >>>>>>")
        print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
        print("<<<<<<< 도구 호출 >>>>>>")

    # 관찰 결과를 출력하는 콜백 함수입니다.
    def observation_callback(observation) -> None:
        print("<<<<<<< 관찰 내용 >>>>>>")
        print(
            f"Observation: {observation.get('observation')[0]}"
        )  # 관찰 내용을 출력합니다.
        print("<<<<<<< 관찰 내용 >>>>>>")

    # 최종 결과를 출력하는 콜백 함수입니다.
    def result_callback(result: str) -> None:
        print("<<<<<<< 최종 답변 >>>>>>")
        print(result)  # 최종 답변을 출력합니다.
        print("<<<<<<< 최종 답변 >>>>>>")

    # AgentCallbacks 객체를 생성하여 각 단계별 콜백 함수를 설정합니다.
    agent_callbacks = AgentCallbacks(
        tool_callback=tool_callback,
        observation_callback=observation_callback,
        result_callback=result_callback,
    )

    # AgentStreamParser 객체를 생성하여 에이전트의 실행 과정을 파싱합니다.
    agent_stream_parser = AgentStreamParser(agent_callbacks)

    store = {}

    def get_session_history(session_ids):
        if session_ids not in store:  # session_id 가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            store[session_ids] = ChatMessageHistory()
        return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    # ─── 여기에 while 루프 추가 ───
    print("질문을 입력하세요. 종료하려면 'exit'를 입력하세요.")
    while True:
        query = input(">> ").strip()
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break

        # 에이전트에 질의 전송
        result = agent_with_chat_history.stream(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
        )
        for step in result:
            agent_stream_parser.process_agent_steps(step)
        print()  # 다음 질문을 위해 줄바꿈


if __name__ == "__main__":
    main()
