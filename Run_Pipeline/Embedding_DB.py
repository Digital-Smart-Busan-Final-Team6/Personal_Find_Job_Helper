# embedding_db.py
from pathlib import Path
from typing import List
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class EmbeddingDB:
    def __init__(self, model_name: str, device: str, persist_dir: str):
        """
        model_name: HuggingFace 임베딩 모델명
        device: "cuda", "mps", "cpu" 등
        persist_dir: Chroma DB를 저장/로드할 경로
        """
        self.model_name  = model_name
        self.device      = device
        self.persist_dir = Path(persist_dir)

    def _get_embedding_function(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_or_create_db(self, docs: List[Document]) -> Chroma:
        self.persist_dir.mkdir(parents=True, exist_ok=True) # 폴더 이미 있으면 넘어가라.
        embed_fn = self._get_embedding_function()

        # 기존 DB가 있으면 로드
        if any(self.persist_dir.iterdir()):
            print("▶ 기존 Chroma DB 로드 중...")
            db = Chroma(persist_directory=str(self.persist_dir), embedding_function=embed_fn)
            print("▶ 로드 완료")
        else:
            print("▶ 새 Chroma DB 생성 중 (임베딩 + 저장)")
            db = Chroma(persist_directory=str(self.persist_dir), embedding_function=embed_fn)
            for doc in tqdm(docs, desc="문서 임베딩 중", unit="doc"):
                db.add_documents([doc])
            # (langchain_chroma 내부적으로 persist 처리)
            print("▶ 새 DB 저장 완료")
        return db

    def get_or_create_db_simple(self, texts: List[str]) -> Chroma:
        self.persist_dir.mkdir(parents=True, exist_ok=True)  # 폴더 이미 있으면 넘어가라.
        embed_fn = self._get_embedding_function()

        # 기존 DB가 있으면 로드
        if any(self.persist_dir.iterdir()):
            print("▶ 기존 Chroma DB (Simple) 로드 중...")
            db = Chroma(persist_directory=str(self.persist_dir), embedding_function=embed_fn)
            print("▶ 로드 완료")
        else:
            print("▶ 새 Chroma DB (Simple) 생성 중 (임베딩 + 저장)")
            db = Chroma(persist_directory=str(self.persist_dir), embedding_function=embed_fn)

            # 텍스트를 Document 객체로 변환 (metadata 완전히 없음)
            simple_docs = []
            for text in texts:
                doc = Document(
                    page_content=text,
                    metadata={}  # 완전히 빈 metadata
                )
                simple_docs.append(doc)

            # 배치로 한번에 추가 (더 효율적)
            for doc in tqdm(simple_docs, desc="텍스트 임베딩 중", unit="text"):
                db.add_documents([doc])

            print("▶ 새 DB (Simple) 저장 완료")
        return db
