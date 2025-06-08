# retriever_builder.py
from typing import List
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from typing import Any
class RetrieverBuilder:
    def __init__(self, db, docs: List[Document], k: int = 3, mode: int = 1):
        """
        db: Chroma 인스턴스 (벡터 DB)
        docs: 분할된 Document 리스트 (BM25 인덱싱용)
        k: top-k 검색 개수
        mode: 1=벡터, 2=BM25, 3=앙상블(벡터+BM25)
        """
        self.db   = db
        self.docs = docs
        self.k    = k
        self.mode = mode

    def build(self) -> Any:
        # 1) 벡터 리트리버
        vec_retriever = self.db.as_retriever(search_kwargs={"k": self.k})

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

        # 3) 최종 리트리버 반환
        if self.mode == 1:
            return vec_retriever
        if self.mode == 2:
            return bm_retriever
        if self.mode == 3:
            return EnsembleRetriever(retrievers=[vec_retriever, bm_retriever], weights=[0.5, 0.5])

        raise ValueError("mode는 1, 2, 또는 3이어야 합니다.")
