# main_pipeline.py
import os
from pathlib import Path
from typing import Any

from Env_Loader import EnvLoader
from Document_Loader import DocumentLoader
from Document_Splitter import DocumentSplitter
from Embedding_DB import EmbeddingDB
from Retiever_Builder import RetrieverBuilder
from LLM_Factory import LLMFactory
from Chain_Factory import ChainFactory

def main(return_chain_only: bool = False):
    # ① 환경 변수 로드 (로컬 우선)
    EnvLoader.load_local()

    # ② 파라미터 설정
    BASE_DIR = Path(__file__).resolve().parent.parent
    if return_chain_only:
        kind           = "json"
        file_path      = BASE_DIR / os.getenv("DATA_PATH")
        chunk_size     = 1000
        overlap_size   = 50
        device         = "mps"
        persist_dir    = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        retriever_mode = 1
        k              = 3
        engine_num     = 1
        backend_num    = 1
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
        kind           = "json"
        file_path      = BASE_DIR / os.getenv("DATA_PATH")
        chunk_size     = 1000
        overlap_size   = 50
        persist_dir    = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        device         = "mps"
        retriever_mode = 1
        k              = 3
        engine_num     = 1
        backend_num    = 1


    # ③ 문서 로드
    loader = DocumentLoader(file_path, kind) # 일반 파일들을 Document형태로 변환해서 다 불러옴
    docs = loader.load()

    # ④ 문서 분할
    splitter = DocumentSplitter(chunk_size=chunk_size, overlap=overlap_size) # 이미 분할된 문서가 있다면 로드하고 아니면 새로 스플릿함
    chunks = splitter.split(docs, cache_dir=file_path)

    # ⑤ 임베딩 & DB 생성/로드
    embed_db = EmbeddingDB(model_name="nlpai-lab/KURE-v1", device=device, persist_dir=persist_dir) # 이미 임베딩된 DB가 있다면 로드하고 아니면 새로 임베딩함
    db = embed_db.get_or_create_db(chunks)

    # ⑥ 리트리버 빌드
    retriever_builder = RetrieverBuilder(db=db, docs=chunks, k=k, mode=retriever_mode) # 리트리버 빌더 생성
    retriever = retriever_builder.build()

    # ⑦ LLM 로드
    llm = LLMFactory.load_llm(engine=engine_num, backend=backend_num, device=device) # LLM 로드

    # ⑧ 체인 생성
    if backend_num == 3:  # HF-Pipeline (PEFT) 모드
        chain = ChainFactory.build_peft_chain(retriever=retriever, llm=llm)
    else:
        chain = ChainFactory.build_chain(retriever=retriever, llm=llm)

    if return_chain_only:
        return chain

    # ⑨ 인터랙티브 질의 루프
    while True:
        q = input("\n질문(종료: exit): ").strip()
        if q.lower() == "exit":
            break
        if backend_num == 3:
            # PEFT 체인 호출 (RunnableLambda 체인 형태)
            result = chain.invoke({"context": retriever.get_relevant_documents(q), "text": q})
            print(result)
        else:
            # RetrievalQA (RetrievalQA 체인)
            from langchain.schema import Document
            print(chain.invoke(q))


if __name__ == "__main__":
    main()
