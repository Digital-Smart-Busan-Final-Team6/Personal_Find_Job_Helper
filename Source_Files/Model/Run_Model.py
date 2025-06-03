# -------- import & helpers ------------------------------------
from pathlib import Path
import json, os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from konlpy.tag import Okt
import pandas as pd
from Source_Files.Model.setting_metadata import *


def load_env_variables_for_Local():
    """
    .env 파일에서 환경 변수를 로드합니다.
    """
    # .env 파일의 경로를 지정합니다.
    # 현재 스크립트의 상대 경로로 설정합니다.
    # load_dotenv(dotenv_path='../../.env')
    # 절대 경로로 설정합니다.
    load_dotenv(dotenv_path='../../.env')

    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
    os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
    os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
    os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')


def load_env_variables_for_Colab():
    from google.colab import userdata
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
    os.environ['LANGSMITH_ENDPOINT'] = userdata.get('LANGSMITH_ENDPOINT2')
    os.environ['LANGSMITH_PROJECT'] = userdata.get('LANGSMITH_PROJECT2')
    os.environ['LANGSMITH_TRACING'] = userdata.get('LANGSMITH_TRACING')
    os.environ['LANGSMITH_API_KEY'] = userdata.get('LANGSMITH_API_KEY2')


# ---------- tokenizers ----------------------------------------
# okt = Okt()
#
#
# def len_okt(t):  return len(okt.morphs(t))
#
#
# def okt_tokenize(t): return okt.morphs(t)

# konlpy Okt 를 지연 초기화하기 위한 래퍼
_okt = None


def get_okt():
    global _okt
    if _okt is None:
        from konlpy.tag import Okt
        _okt = Okt()
    return _okt


# 토크나이저 함수들
def len_okt(t):
    return len(get_okt().morphs(t))


def okt_tokenize(t):
    return get_okt().morphs(t)


# ---------- 1. 로더 -------------------------------------------


import os, json
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader


def load_files(file_path: str, kind: str) -> list[Document]:
    docs = []

    # 1) JSON 처리
    if kind in ("json", "all"):
        json_files = [f for f in os.listdir(file_path) if f.endswith(".json")]
        analysis_files = [f for f in json_files if "Analysis" in f]
        post_files = [f for f in json_files if "improve" in f]

        if analysis_files:
            with open(os.path.join(file_path, analysis_files[0]), encoding="utf-8") as f:
                data_analysis = json.load(f)
            docs += make_analysis_docs(data_analysis)

        if post_files:
            with open(os.path.join(file_path, post_files[0]), encoding="utf-8") as f:
                raw = json.load(f)

            # ◀ 리스트면 dict로 변환 ▶
            if isinstance(raw, list):
                # 리스트 요소에 'id' 키가 있다면 그걸, 없다면 인덱스를 key로
                post_dict = {}
                for idx, item in enumerate(raw):
                    pid = item.get("id")
                    if pid is None:
                        pid = str(idx)
                    post_dict[str(pid)] = item
            elif isinstance(raw, dict):
                post_dict = raw
            else:
                raise ValueError("Post JSON은 list 또는 dict 이어야 합니다.")

            docs += make_post_docs(post_dict)

    # 2) TXT 처리
    if kind in ("txt", "all"):
        loader = DirectoryLoader(
            file_path,
            glob="**/*.txt",
            show_progress=True
        )
        docs += loader.load()

    if not docs:
        raise ValueError(f"'{kind}'에 해당하는 문서를 찾지 못했습니다.")

    return docs


# ---------- 2. 스플리터 ---------------------------------------
from langchain_text_splitters import RecursiveJsonSplitter

from tqdm.auto import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


# def split_docs_with_progress(
#         docs: list[Document],
#         filepath : str,
#         chunk: int = 1000,
#         overlap: int = 100,
# ) -> list[Document]:
#     """
#     각 Document를 순회하며 TextSplitter를 적용하고,
#     tqdm으로 진행률을 표시합니다.
#     """
#
#     # 만약 docs 파일이 존재하면
#     if os.path.exists(filepath+f'/{chunk}_{overlap}_docs'):
#         with open(filepath+f'/{chunk}_{overlap}_docs', 'r', encoding='utf-8') as f:
#             loaded = []
#             for line in f:
#                 record = json.loads(line)
#                 loaded.append(Document(page_content=record["page_content"], metadata=record["metadata"]))
#         return loaded
#
#     else:
#
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk,
#             chunk_overlap=overlap,
#             length_function=len_okt
#         )
#         all_chunks: list[Document] = []
#
#         for doc in tqdm(docs, desc="문서 분할 중....", unit="doc"):
#             # 각 Document마다 split_documents를 호출해도 되고,
#             # 더 세밀하게는 splitter.split_text(doc.page_content)
#             chunks = splitter.split_documents([doc])
#             all_chunks.extend(chunks)
#
#         with open(filepath+f'/{chunk}_{overlap}_docs', 'w', encoding='utf-8') as f:
#             for doc in all_chunks:
#                 record = {
#                     "page_content": doc.page_content,
#                     "metadata": doc.metadata
#                 }
#                 f.write(json.dumps(record, ensure_ascii=False))
#                 f.write("\n")
#     return all_chunks

def split_docs_with_progress(
        docs: list[Document],
        filepath: str | Path,
        chunk: int = 1000,
        overlap: int = 100,
) -> list[Document]:
    """
    각 Document를 순회하며 TextSplitter를 적용하고,
    tqdm으로 진행률을 표시합니다.
    캐시 파일(<filepath>/<chunk>_<overlap>_docs)이 있으면 그걸 로드합니다.
    """
    # 1) Path 객체로 통일
    base = Path(filepath)
    cache_file = base / f"{chunk}_{overlap}_docs_chunk"

    # 2) 캐시가 있으면 로드
    if cache_file.exists():
        loaded = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                loaded.append(Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"]
                ))
        return loaded

    # 3) 없으면 새로 split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=overlap,
        length_function=len_okt,  # 기존 토크나이저 함수 사용
    )
    all_chunks: list[Document] = []
    for doc in tqdm(docs, desc="문서 분할 중....", unit="doc"):
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    # 4) 캐시 파일로 저장
    base.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성
    with open(cache_file, 'w', encoding='utf-8') as f:
        for doc in all_chunks:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    return all_chunks


# ---------- 3. 임베딩 & DB ------------------------------------
# 1) 임베딩 ─────────────────────────────────────────────
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings


def load_embed(device: str, model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# 2) Chroma DB ─────────────────────────────────────────
from langchain_chroma import Chroma
from pathlib import Path

from pathlib import Path
from langchain_chroma import Chroma

from pathlib import Path
from tqdm.auto import tqdm
from langchain_chroma import Chroma


def get_db(docs, embed, persist_dir: str):
    """
    ▶ docs: list[Document]
    ▶ embed: embedding function
    ▶ persist_dir: 저장할 폴더 경로
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # 1) 기존 DB 로드
    if any(persist_path.iterdir()):
        print("▶ 기존 Chroma DB 로드")
        db = Chroma(persist_directory=str(persist_path), embedding_function=embed)
        print("▶ 기존 DB 로드 완료")
    # 2) 새 DB 생성
    else:
        print("▶ 새 Chroma DB 생성 (임베딩 + 저장)")
        db = Chroma(persist_directory=str(persist_path), embedding_function=embed)

        # tqdm 으로 프로그레스 바 표시하며 문서 추가
        for doc in tqdm(docs, desc="문서 추가 중", unit="doc"):
            db.add_documents([doc])

        # langchain_chroma 에서는 db.persist() 가 없으므로
        # 내부 클라이언트에 persist() 를 호출합니다.
        # db._client.persist()
        print('▶ 새 DB 저장 완료')
    return db


# 사용 예
# db = get_db(documents, embedding_fn, "my_chroma_db")
# 이후 추가 삽입 등 변경이 있으면:
# db.add_documents(new_docs)
# db.persist()


# ---------- 4. Retriever --------------------------------------
def build_retriever(mode: int, k: int, db, docs):
    # 1) Vector retriever (Chroma)
    vec = db.as_retriever(search_kwargs={"k": k})

    # 2) BM25 retriever with tqdm progress
    if mode in (2, 3):
        print("▶ BM25 색인 중…")
        # tqdm 으로 docs 순회 모습을 보여준 뒤 from_documents 로 색인
        bm = BM25Retriever.from_documents(
            tqdm(docs, desc="BM25 문서 인덱싱", unit="doc"),
            preprocess_func=okt_tokenize
        )
        bm.k = k
    else:
        bm = None

    # 3) 최종 리턴
    if mode == 1:
        return vec
    if mode == 2:
        return bm
    if mode == 3:
        return EnsembleRetriever(retrievers=[vec, bm], weights=[0.5, 0.5])

    raise ValueError("retriever_num 은 1~3")


# ---------- 5. LLM --------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer


from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
from pathlib import Path
import os
from dotenv import load_dotenv

def load_peft_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    gen_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id or 2,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=gen_config
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def load_llm(engine: int, backend: int, device: str = "cuda"):
    # 1) engine 번호에 따른 모델 ID 결정
    if engine == 1:
        name = "gpt-4o-mini"
    elif engine == 2:
        name = "gemma3:4b"
    elif engine == 3:
        name = "qwen3:4b"
    elif engine == 4:
        name = "SiniDSBA/8800QA_SET"
    else:
        raise ValueError("engine_num 은 1~4")

    # 2) backend별 LLM 반환
    if backend == 1:
        return ChatOpenAI(
            model=name,
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    elif backend == 2:
        return ChatOllama(
            model=name,
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    elif backend == 3:
        # HuggingFacePipeline을 통한 로딩
        pipe = pipeline(
            "text-generation",
            model=name,
            device=device,  # GPU 사용
            max_new_tokens=256,
            temperature=0.0
        )  # :contentReference[oaicite:7]{index=7}
        return HuggingFacePipeline(pipeline=pipe)  # :contentReference[oaicite:8]{index=8}
    else:
        raise ValueError("backend는 1, 2, 또는 3 중 하나여야 합니다.")


# ---------- 6. Chain ------------------------------------------
prompt_content = """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.

            Answer in Korean.

            #Context:
            {context}
            """

PROMPT = ChatPromptTemplate.from_messages([

    (
        "system",
        prompt_content,
    ), ("human", "{text}"),

])
def build_peft_chain(retriever, llm):

    format_docs = lambda ds: "\n\n".join(d.page_content for d in ds)

    def generate_answer(prompt):
        result = llm.invoke(prompt)
        answer = result.split("### 답변:")[-1].strip()
        return answer

    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    } | PROMPT | RunnableLambda(generate_answer)

    return chain

def build_chain(retriever, llm):
    format_docs = lambda ds: "\n\n".join(d.page_content for d in ds)
    chain = {
                "context": retriever | RunnableLambda(format_docs),
                "text": RunnablePassthrough(),
            } | PROMPT | llm | StrOutputParser()
    return chain


from langchain.chains import RetrievalQA



def main(return_chain_only: bool = False):
    # ① 환경 변수 로드
    if os.path.exists("../../.env"):
        load_env_variables_for_Local()
    # else:
    #     load_env_variables_for_Colab()
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # serve 모드면 input() 건너뛰고, ENV에서 설정을 읽어들임
    if return_chain_only:
        # kind          = os.getenv("KIND","json")  # json/txt/all
        # file_path     = BASE_DIR / os.getenv("DATA_PATH")
        # chunk_size    = int(os.getenv("CHUNK_SIZE",    "1000"))
        # overlap_size  = int(os.getenv("OVERLAP_SIZE",    "50"))
        # device        = { "mps": "mps", "cuda": "cuda", "cpu": "cpu" }\
        #                   [os.getenv("DEVICE", "cuda")]
        # persist_dir   = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        # retriever_num = int(os.getenv("RETR_MODE",    "1"))  # 1/2/3
        # k             = int(os.getenv("RETR_K",       "3"))  # top-k
        # llm_model_num = int(os.getenv("LLM_MODEL",    "1"))  # 1:gpt-4o-mini,2:gemma3,3:qwen3
        # llm_backend   = int(os.getenv("LLM_BACKEND",  "1"))  # 1:OpenAI,2:Ollama

        kind = 'json'  # json/txt/all
        file_path = BASE_DIR / os.getenv("DATA_PATH")
        chunk_size = 1000
        overlap_size = 50
        device = 'mps'
        persist_dir = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
        retriever_num = 1  # 1/2/3
        k = 3  # top-k
        llm_model_num = 1  # 1:gpt-4o-mini,2:gemma3,3:qwen3
        llm_backend = 1  # 1:OpenAI,2:Ollama

    else:
        # 기존 인터랙티브
        kind = input("파일 종류(json/txt/all): ").strip().lower()
        file_path = "../../Data_Files"
        chunk_size = int(input("청크 사이즈(기본 1000): "))
        overlap_size = int(input("오버랩 사이즈(기본 50): "))
        persist_dir = f'../../Data_Files/{kind}_{chunk_size}'
        device = {1: "mps", 2: "cuda", 3: "cpu"}[int(input("디바이스(1:mps/2:cuda/3:cpu): "))]
        retriever_num = int(input("retriever (1 vec /2 bm25 /3 ensemble): "))
        k = int(input("k 개수를 입력해 주세요: "))
        llm_model_num = int(input("LLM 모델 번호(1: gpt-4o-mini / 2: gemma3:4b / 3:qwen3:4b / 4:Private Model): "))
        llm_backend = int(input("LLM (1:OpenAI / 2:Ollama / 3: HuggingFace): "))

    # ② 파일 로드 → docs
    docs = load_files(file_path, kind)

    # ③ 분할
    chunks = split_docs_with_progress(docs, file_path, chunk=chunk_size, overlap=overlap_size)

    # ④ 임베딩 & DB
    embed = load_embed(device, "nlpai-lab/KURE-v1")
    db = get_db(chunks, embed, persist_dir)

    # ⑤ retriever
    retr = build_retriever(retriever_num, k=k, db=db, docs=chunks)

    # ⑦ Chain 생성
    if llm_backend == 3:
        # PEFT 모델일 경우
        llm = load_peft_model('SiniDSBA/8800QA_SET', device=device)

        chain = build_peft_chain(retr, llm)
    else:
        # 일반 LLM일 경우
        llm = load_llm(llm_model_num, backend=llm_backend, device=device)
        chain = build_chain(retr, llm)

    if return_chain_only:
        return chain

    # ── 인터랙티브 질의 루프 ──────────────────
    while True:
        q = input("\n질문(종료 exit): ")
        if q.lower() == "exit":
            break
        print(chain.invoke(q))


if __name__ == "__main__":
    main()
