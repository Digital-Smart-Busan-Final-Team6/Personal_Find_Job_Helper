import os
import json
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, file_path: str, kind: str = "all"):
        self.file_path = Path(file_path)
        self.kind = kind.lower()


    def load(self) -> List[Document]:
        docs: List[Document] = []
        if self.kind in ("json", "all"):
            docs += self._load_json_files()
        if not docs:
            raise ValueError(f"No documents found for kind='{self.kind}' in '{self.file_path}'")
        return docs

    def _load_json_files(self) -> List[Document]:
        docs: List[Document] = []
        # 1) JSON 파일 전체 스캔
        json_paths = list(Path(self.file_path).glob("*.json"))

        # 2) 'improve' 포함된 파일 → 게시글 Document
        for path in json_paths:
            if "improve" in path.name:
                with path.open(encoding="utf-8") as f:
                    raw = json.load(f)

                # 리스트/딕셔너리 형태에 맞춰 post_dict 생성
                for post_id, post in raw.items():
                    meta = {
                        "id": post_id,
                        "source": "채용 공고"
                    }
                    # dict를 "Key: Value" 형태의 문자열로 변환
                    content = "\n".join(f"{k}: {v}" for k, v in post.items())
                    docs.append(Document(page_content=content, metadata=meta))

                # (이후에도 company 파일을 함께 처리하려면 continue 없이 두 루프 모두 실행)

        # 3) 'Company' 포함된 파일 → 회사 정보 Document
        for path in json_paths:
            if "company" in path.name:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                for company in data:
                    company_name = company.get("회사명", "")
                    metadata = {
                        "id": company_name,
                        "source": "회사 정보"
                    }

                    # 나머지 필드는 key: value 형태로 page_content 생성
                    lines = []
                    for k, v in company.items():
                        if isinstance(v, list):
                            joined = "\n".join(v)
                        else:
                            joined = str(v)
                        lines.append(f"{k}: {joined}")

                    page_content = "\n".join(lines)
                    docs.append(Document(page_content=page_content, metadata=metadata))

        if not docs:
            raise FileNotFoundError("`improve` 또는 `Company` JSON 파일을 찾지 못했습니다.")
        return docs
