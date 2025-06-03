# ✦ pipeline.py  ───────────────────────────────────────────────
from pathlib import Path
import os, json
from dotenv import load_dotenv
from langchain.schema import Document
from konlpy.tag import Okt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 고정 파라미터 (원한다면 .env 로 빼도 됨) -----------------
DATA_DIR           = "../../Data_Files"
FILE_KIND          = "json"               # 또는 "txt"
CHUNK, OVERLAP     = 1000, 100
DEVICE, EMB_MODEL  = "mps", "nlpai-lab/KURE-v1"
RETRIEVER_MODE, K  = 3, 5                 # 1 vec / 2 bm25 / 3 ensemble
LLM_ENGINE, BACKEND = 1, 1               # 1 gpt-4o , 2 gemma3 … / backend 1 OpenAI 2 Ollama
# --------------------------------------------------------------

okt = Okt()
len_okt = lambda t: len(okt.morphs(t))
okt_tokenize = lambda t: okt.morphs(t)

def load_docs(folder: str, kind: str):
    docs = []
    if kind == "json":
        for p in Path(folder).rglob("*.json"):
            txt = json.dumps(json.load(p.open()), ensure_ascii=False, indent=2)
            docs.append(Document(page_content=txt, metadata={"source": p.as_posix()}))
    else:
        from langchain_community.document_loaders import DirectoryLoader
        docs.extend(DirectoryLoader(folder, glob="**/*.txt").load())
    return docs

def build_chain() -> "Runnable":
    # 1) 파일 로드 & 청크
    docs   = load_docs(DATA_DIR, FILE_KIND)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK,
                                              chunk_overlap=OVERLAP,
                                              length_function=len_okt)
    texts  = splitter.create_documents([d.page_content for d in docs])

    # 2) 임베딩 & Chroma
    embed = HuggingFaceEmbeddings(model_name=EMB_MODEL,
                                  model_kwargs={"device": DEVICE},
                                  encode_kwargs={"normalize_embeddings": True})
    db    = Chroma.from_documents(texts, embed, persist_directory=f"./{FILE_KIND}_{CHUNK}")

    # 3) Retriever
    vec = db.as_retriever(search_kwargs={"k": K})
    bm  = BM25Retriever.from_documents(texts, preprocess_func=okt_tokenize); bm.k = K
    if RETRIEVER_MODE == 1: retr = vec
    elif RETRIEVER_MODE == 2: retr = bm
    else: retr = EnsembleRetriever(retrievers=[vec, bm], weights=[0.5,0.5])

    # 4) LLM
    name_map = {1: "gpt-4o-mini", 2: "gemma3:4b", 3: "quen3:4b"}
    if BACKEND == 1:
        llm = ChatOpenAI(model=name_map[LLM_ENGINE], temperature=0)
    else:
        llm = ChatOllama(model=name_map[LLM_ENGINE], temperature=0)

    # 5) Chain
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use context to answer in Korean.\n#Context:{context}"),
        ("human", "{question}")
    ])
    fmt = lambda ds: "\n\n".join(d.page_content for d in ds)

    return {
        "context": retr | RunnableLambda(fmt),
        "question": RunnablePassthrough()
    } | PROMPT | llm | StrOutputParser()
