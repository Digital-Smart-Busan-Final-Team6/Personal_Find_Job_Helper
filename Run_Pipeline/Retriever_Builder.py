# retriever_builder.py (수정된 버전)
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
                 chunks: List[Document], k: int = 3):
        """
        db: Chroma 인스턴스 (벡터 DB)
        full_docs: 원본 문서들 (Parent Document Retriever용)
        chunks: 분할된 Document 리스트 (BM25 및 벡터 인덱싱용)
        k: top-k 검색 개수
        """
        self.db = db
        self.full_docs = full_docs
        self.chunks = chunks  # 수정: self.docs -> self.chunks
        self.k = k

    def build(self, mode: int = 1) -> Any:
        """
        mode: 1=벡터, 2=BM25, 3=앙상블(벡터+BM25), 4=ParentDocument
        """
        return self.build_with_mode(mode)

    def build_with_mode(self, mode: int) -> Any:
        # 1) 벡터 리트리버
        vec_retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 2, "lambda_mult": 0.5,
                           "score_threshold": 0.7},
        )

        # 2) BM25 리트리버 (모드 2 또는 3일 때만)
        bm_retriever = None
        if mode in (2, 3):
            print("▶ BM25 색인 중…")
            bm = BM25Retriever.from_documents(
                tqdm(self.chunks, desc="BM25 문서 인덱싱", unit="doc"),  # 수정: self.docs -> self.chunks
                preprocess_func=lambda t: t.split()  # 혹은 konlpy 형태소 분석 함수
            )
            bm.k = self.k
            bm_retriever = bm

        # 3) Parent Document Retriever
        if mode == 4:
            parent_store = InMemoryStore()
            parent_store.mset([(doc.metadata["id"], doc) for doc in self.full_docs])

            parent_retriever = ParentDocumentRetriever(  # 수정: 오타 수정
                vectorstore=self.db,
                docstore=parent_store,
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50),
                search_kwargs={"k": self.k},
                id_key="id",
            )
            return parent_retriever

        # 4) 최종 리트리버 반환
        if mode == 1:
            return vec_retriever
        elif mode == 2:
            return bm_retriever
        elif mode == 3:
            from langchain.retrievers import EnsembleRetriever
            return EnsembleRetriever(retrievers=[vec_retriever, bm_retriever], weights=[0.5, 0.5])

        raise ValueError("mode는 1, 2, 3 또는 4이어야 합니다.")  # 수정: 오타 수정