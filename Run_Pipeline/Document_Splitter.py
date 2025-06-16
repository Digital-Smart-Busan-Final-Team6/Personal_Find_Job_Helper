# document_splitter.py
import json
from pathlib import Path
from typing import List
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from konlpy.tag import Okt


class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.okt = Okt()

    def _len_okt(self, text: str) -> int:
        return len(self.okt.morphs(text))

    def _okt_tokenize(self, text: str):
        return self.okt.morphs(text)

    def split(self, docs: List[Document], cache_dir: str) -> List[Document]:
        cache_dir = Path(cache_dir)

        cache_file = cache_dir / f"{self.chunk_size}_{self.overlap}_docs_chunk"

        # 캐시 존재 시 로드
        if cache_file.exists():
            loaded: List[Document] = []
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    loaded.append(Document(page_content=record["page_content"], metadata=record["metadata"]))
            print(f"캐시에서 {len(loaded)}개의 문서 청크 로드 완료")
            return loaded

        # 캐시 없으면 분할 수행
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=self._len_okt
        )

        all_chunks: List[Document] = []
        for doc in tqdm(docs, desc="문서 분할 중", unit="doc"):
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        # 캐시에 저장
        with open(cache_file, 'w', encoding='utf-8') as f:
            for doc in all_chunks:
                record = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

        return all_chunks
