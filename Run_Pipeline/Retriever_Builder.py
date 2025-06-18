# retriever_builder.py
from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from typing import Any
from langchain.storage import InMemoryStore


class RetrieverBuilder:
    def __init__(self, db, full_docs: List[Document],
                 chunks: List[Document], k: int = 3, mode: int = 1):
        """
        db: Chroma 인스턴스 (벡터 DB)
        docs: 분할된 Document 리스트 (BM25 인덱싱용)
        k: top-k 검색 개수
        mode: 1=벡터, 2=BM25, 3=앙상블(벡터+BM25)
        """
        self.db = db
        self.full_docs = full_docs
        self.chunks = chunks
        self.k = k
        self.mode = mode

    def build(self) -> Any:
        # 1) 벡터 리트리버
        vec_retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 2, "lambda_mult": 0.5,
                           "score_threshold": 0.7},
        )

        # 2) BM25 리트리버 (모드 2 또는 3일 때만)
        bm_retriever = None
        if self.mode in (2, 3):
            print("▶ BM25 색인 중…")
            bm = BM25Retriever.from_documents(
                tqdm(self.docs, desc="BM25 문서 인덱싱", unit="doc"),
                preprocess_func=lambda t: t.split()  # 혹은 konlpy 형태소 분석 함수
            )
            bm.k = self.k
            bm_retriever = bm

        parent_store = InMemoryStore()
        parent_store.mset([(doc.metadata["id"], doc) for doc in self.full_docs])

        if self.mode == 4:
            parent_retreiver = ParentDocumentRetriever(
                vectorstore=self.db,
                docstore=parent_store,
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50),  # 더미값임
                search_kwargs={"k": self.k},
                id_key="id",
            )


        # 3) 최종 리트리버 반환
        if self.mode == 1:
            return vec_retriever
        if self.mode == 2:
            return bm_retriever
        if self.mode == 3:
            from langchain.retrievers import EnsembleRetriever
            return EnsembleRetriever(retrievers=[vec_retriever, bm_retriever], weights=[0.5, 0.5])
        if self.mode == 4:
            return parent_retreiver

        raise ValueError("mode는 1, 2, ,3 또는 4이어야 합니다.")
