import os
import json
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from setting_metadata import *

class DocumentLoader:
    def __init__(self, file_path: str, kind: str = "all"):
        """
        kind: "json", "txt", "all"
        """
        self.file_path = Path(file_path)
        self.kind = kind.lower()

    def load(self) -> List[Document]:
        docs: List[Document] = []
        if self.kind in ("json", "all"):
            docs += self._load_json_files()
        if self.kind in ("txt", "all"):
            docs += self._load_txt_files()
        if not docs:
            raise ValueError(f"No documents found for kind='{self.kind}' in '{self.file_path}'")
        return docs

    def _load_json_files(self) -> List[Document]:
        json_files = [f for f in os.listdir(self.file_path) if f.endswith(".json")]
        analysis_files = [f for f in json_files if "Analysis" in f]
        post_files     = [f for f in json_files if "improve"  in f]
        company_files = [f for f in json_files if "Company" in f]

        docs: List[Document] = []
        # 예시: "Analysis" 파일이 있으면 분석 Document 생성
        if analysis_files:
            with open(self.file_path / analysis_files[0], encoding="utf-8") as f:
                data_analysis = json.load(f)
            docs += make_analysis_docs(data_analysis)

        # "improve" 파일이 있으면 게시글 형태 Document 생성
        if post_files:
            with open(self.file_path / post_files[0], encoding="utf-8") as f:
                raw = json.load(f)
            post_dict = {}
            if isinstance(raw, list):
                for idx, item in enumerate(raw):
                    pid = item.get("id", str(idx))
                    post_dict[str(pid)] = item
            elif isinstance(raw, dict):
                post_dict = raw
            else:
                raise ValueError("Post JSON must be list or dict.")
            docs += make_post_docs(post_dict)

        return docs



    def _load_txt_files(self) -> List[Document]:
        loader = DirectoryLoader(str(self.file_path), glob="**/*.txt", show_progress=True)
        return loader.load()
