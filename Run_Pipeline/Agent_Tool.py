import logging
import os
from pathlib import Path
from typing import List, Optional, Any
import pandas as pd
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch  # torch 임포트 유지 (SBERT 내부 사용)

BASE_DIR = Path(__file__).resolve().parent.parent


class AgentTools:
    # --- Configuration for Job Matching Tool ---
    SBERT_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
    # 가중치 설정 (모든 이력서에 대해 고정)
    WEIGHT_COSINE_SIMILARITY = 0.6
    WEIGHT_EXPERIENCE_MATCH = 0.2
    WEIGHT_TECH_STACK_MATCH = 0.15
    WEIGHT_JOB_ROLE_MATCH = 0.05
    # 이력서의 직무 관련 키워드 (고정된 가중치를 사용하므로 여기서 직접 정의)
    # 이 부분은 이력서 파싱 후 LLM으로 동적 추출하는 방향으로 고도화될 수 있습니다.
    RESUME_JOB_ROLE_KEYWORDS = ["ai", "데이터 분석", "엔지니어", "개발자", "소프트웨어"]  # 새로 추가/변경: 클래스 변수로 이동
    # --- End Configuration ---

    # ────────────────────── 공통 PROMPT 정의 ──────────────────────
    prompt_content = """
    # 롤 & 톤
    당신은 신뢰할 수 있는 취업 전문가 AI 코치입니다.
    모든 답변은 친절하지만 간결한 한국어로 작성하세요.
    필요하면 표·리스트·예시 코드를 사용해도 좋습니다.

    # 툴 사용 규칙
    1. 채용 공고, 취업 뉴스, 면접 정보 등 **문서**에서 답을 찾을 수 있을 것 같으면
       반드시 `document_search` 툴을 먼저 호출한다.
    2. `document_search`가 결과를 0개 반환하거나(embeddings empty) 내용이 질문과 무관하면
       보강 정보가 필요하다고 판단하고 `search_web` 툴을 호출한다.
    3. 사용자가 “평균”, “통계”, “그래프”, “필터” 같은 키워드로
       **데이터 분석**을 요구하면 `job_dataframe_analysis` 툴을 호출한다.
    4. 사용자가 “이미지 만들어”, “로고 그려” 같이 시각 생성 요청을 하면 `generate_image`.
    5. “내 이력서” 또는 “내 경력”을 직접 언급하면 `resume_qa`.
    6. **"가장 적합한 공고", "내 이력서에 맞는 공고", "추천 공고"**와 같은 키워드로
       이력서와 공고 매칭을 요청하면 `find_best_job_match` 툴을 호출한다.
    7. 위 조건에 맞지 않으면 툴을 사용하지 않고 자체 지식으로 답한다.

    # 워크플로 예시 (Few-shot)
    ## 예시 1 – 문서 검색 성공
    User: "삼성전자 백엔드 신입 연봉 정보 알려줘"
    Assistant:
    - Thought: "채용 공고 문서에 있을 가능성이 높다"
    - Action: document_search {{ "query": "삼성전자 백엔드 신입 연봉" }}

    ## 예시 2 – 문서에 없음 → 웹 검색
    User: "내년 공무원 채용 일정 알려줘"
    Assistant:
    - Thought: "사내 문서엔 없음, 웹 최신 정보 필요"
    - Action: search_web {{ "query": "2025 공무원 채용 일정" }}

    ## 예시 3 – 데이터프레임 통계
    User: "지난 3개월 동안 서울에서 올라온 프론트엔드 평균 연봉 그래프로 보여줘"
    Assistant:
    - Thought: "DataFrame 분석 필요"
    - Action: job_dataframe_analysis {{ "query": "지난 3개월 서울 프론트엔드 평균 연봉 그래프" }}

    ## 예시 4 – 이력서 피드백
    User: "내 이력서에서 개선할 점 알려줘"
    Assistant:
    - Thought: "resume_qa 툴 사용"
    - Action: resume_qa {{ "question": "내 이력서 개선 포인트" }}

    ## 예시 5 – 이력서 기반 공고 추천
    User: "제 이력서에 가장 적합한 채용 공고를 찾아 추천해주세요."
    Assistant:
    - Thought: "이력서 기반 공고 매칭이 필요하므로 find_best_job_match 툴을 사용한다."
    - Action: find_best_job_match {{ "query": "이력서에 가장 적합한 공고 추천" }} # 변경: question -> query로 (이 툴은 사용자 질문에 관계없이 항상 이력서 기반 추천을 하므로)

    대답을 작성할 땐 최종적으로
    **"Answer:"** 섹션에서 사용자에게 보이는 답을 작성하고,
    필요하면 **"Sources:"** 항목에 참고 툴 결과를 간단히 인용한다.
    """

    # ChatPromptTemplate 생성: 올바른 튜플 형태로 지정
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # --------------------------------------------------
    @staticmethod
    def get_tools(*, retriever: Any, llm) -> List[Tool]:
        tools = [
            AgentTools._create_search_tool(),
            AgentTools._create_retriever_tool(retriever),
            AgentTools._create_dataframe_tool(llm),
            AgentTools._create_image_generation_tool(),
            AgentTools._create_resume_tool(llm),
            AgentTools._create_job_match_tool(retriever)  # 새로 추가/변경: retriever를 인수로 전달
        ]

        # ▒ file-management 툴 (여러 개) 추가
        file_tools = AgentTools._create_file_management_tool()
        if file_tools:
            tools.extend(file_tools)

        return [t for t in tools if t is not None]

    # ────────────────────── tool builders ──────────────────────
    # 1) 웹 검색
    @staticmethod
    def _search_web(query: str) -> str:
        return TavilySearchResults(k=6).run(query)

    @staticmethod
    def _create_search_tool() -> Tool:
        return Tool.from_function(
            func=AgentTools._search_web,
            name="search_web",
            description="문서에 없으면 최신 웹 검색을 수행한다.",
        )

    # 2) RAG 리트리버
    @staticmethod
    def _create_retriever_tool(retriever: Any) -> Optional[Tool]:
        if retriever is None:
            logging.warning("Retriever is None – document_search 툴 비활성화.")
            return None

        return create_retriever_tool(
            retriever=retriever,
            name="document_search",
            description="업로드된 채용 공고 벡터 DB에서 관련 내용을 찾는다.",
        )

    # 3) DataFrame 분석
    @staticmethod
    def _create_dataframe_tool(llm) -> Optional[Tool]:
        if llm is None:
            logging.warning("LLM is None – dataframe 툴 비활성화.")
            return None

        data_path = BASE_DIR / "Data_Files" / "wanted_detail_improve_20250604.json"
        if not data_path.exists():
            logging.warning("DataFrame 파일을 찾을 수 없음 – dataframe 툴 비활성화.")
            return None

        df = pd.read_json(data_path).T

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            prefix="너는 채용공고 데이터 분석 전문가다.",
        )

        return Tool.from_function(
            func=lambda q: pandas_agent.run(q),
            name="job_dataframe_analysis",
            description="채용공고 DataFrame을 이용해 통계·시각화를 수행한다.",
        )

    # 4) 이미지 생성
    @staticmethod
    def _create_image_generation_tool() -> Optional[Tool]:
        """
        DALL-E 이미지 생성 툴을 생성합니다.
        현재는 사용하지 않지만, 필요시 활성화할 수 있습니다.
        """
        if not os.getenv("OPENAI_API_KEY"):
            logging.warning("DALL-E API 키가 설정되지 않았습니다. 이미지 생성 툴 비활성화.")
            return None

        dalle_tool = DallEAPIWrapper(
            model="dall-e-3",  # 생성 모델
            size="1024x1024",
            quality="standard",  # 품질 설정 hd도 있음
            n=1  # 생성할 이미지 개수
        )
        return Tool.from_function(
            func=dalle_tool.run,
            name="generate_image",
            description="이미지를 생성합니다.",
        )

    # 5) 파일 관리 툴
    @staticmethod
    def _create_file_management_tool() -> Optional[List[Tool]]:
        """
        파일 시스템을 다루는 여러 Tool(write_file, read_file …)을 돌려준다.
        """
        working_directory = BASE_DIR / "Data_Files"

        file_tools = FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=[
                "write_file", "read_file",
                "list_directory", "move_file",
            ],
        ).get_tools()

        return file_tools

    # 6) 이력서 QA 툴
    @staticmethod
    def _create_resume_tool(llm) -> Tool:
        """
        이력서를 LLM 컨텍스트에 넣어 질의-응답해 주는 툴.
        """
        data_path = BASE_DIR / "Data_Files" / "resume.json"
        try:
            with open(data_path, encoding="utf-8") as f:
                resume = json.load(f)
        except FileNotFoundError:
            logging.error(f"Resume file not found at {data_path}. Resume QA tool will not function.")
            return Tool.from_function(
                func=lambda q: "이력서 파일을 찾을 수 없어 이력서 관련 질문에 답변할 수 없습니다.",
                name="resume_qa",
                description="이력서 파일이 없을 경우 답변할 수 없습니다."
            )
        except json.JSONDecodeError:
            logging.error(f"Error decoding resume JSON from {data_path}. Resume QA tool will not function.")
            return Tool.from_function(
                func=lambda q: "이력서 파일 형식이 올바르지 않아 이력서 관련 질문에 답변할 수 없습니다.",
                name="resume_qa",
                description="이력서 파일 형식이 올바르지 않을 경우 답변할 수 없습니다."
            )

        # ▶ 필요한 형태(평문)로 합치기
        if isinstance(resume, dict):
            resume_text = "\n".join(f"{k}: {v}" for k, v in resume.items())
        elif isinstance(resume, list):
            lines = []
            for idx, item in enumerate(resume, 1):
                if isinstance(item, dict):
                    lines.append(f"--- 항목 {idx} ---")
                    lines.extend(f"{k}: {v}" for k, v in item.items())
                else:
                    lines.append(str(item))
            resume_text = "\n".join(lines)
        else:
            resume_text = str(resume)

        def _answer_about_resume(question: str) -> str:
            prompt = f"""너는 구직자의 이력서 코치야.
            이력서 원문:
            ---
            {resume_text}
            ---
            사용자의 질문: {question}
            """
            return llm.invoke(prompt).content

        return Tool.from_function(
            func=_answer_about_resume,
            name="resume_qa",
            description="이력서 내용에 대해 질문하면, 조언이나 답변을 해 줍니다.",
        )





    # # 7) 이력서-채용 공고 적합도 계산 툴 (RAG Retriever와 연동) # 새로 추가/변경
    # @staticmethod
    # def _create_job_match_tool(retriever: Any) -> Optional[Tool]:  # retriever 인수를 받도록 변경
    #     if retriever is None:
    #         logging.warning("Retriever is None – find_best_job_match 툴 비활성화.")
    #         return None
    #
    #     # --------------------- 새로 추가/변경 시작 ---------------------
    #     # 1. 채용 공고 원본 데이터 전체를 미리 로드
    #     # 이 데이터는 ID를 키로 하여 빠르게 접근할 수 있도록 딕셔너리로 만듭니다.
    #     full_job_post_data_path = BASE_DIR / "Data_Files" / "wanted_detail_improve_20250604.json"
    #     full_job_post_data = {}
    #     try:
    #         with open(full_job_post_data_path, 'r', encoding='utf-8') as f:
    #             # wanted_detail_improve_20250604.json이 이미 ID: {데이터} 형태라면 바로 로드
    #             # 그렇지 않고 리스트 형태라면 ID를 키로 하는 딕셔너리로 변환
    #             raw_data = json.load(f)
    #             if isinstance(raw_data, dict):  # 이미 ID: {} 형태
    #                 full_job_post_data = raw_data
    #             elif isinstance(raw_data, list):  # [{공고1}, {공고2}, ...] 형태라면 변환
    #                 for job_entry in raw_data:
    #                     # 'id' 필드를 사용하여 키로 만듭니다. 'id' 필드가 없으면 건너뛸 수 있습니다.
    #                     job_id = str(job_entry.get('id'))  # ID는 문자열로 취급
    #                     if job_id and job_id != 'None':  # 'id'가 유효한 값인 경우에만 추가
    #                         full_job_post_data[job_id] = job_entry
    #                     else:
    #                         logging.warning(f"Job entry with no valid 'id' found: {job_entry.get('제목', '제목 없음')}")
    #             else:
    #                 logging.error(f"Unexpected format for job postings file: {type(raw_data)}. Expected dict or list.")
    #                 return None  # 툴 비활성화
    #         logging.info(f"Loaded {len(full_job_post_data)} job postings from {full_job_post_data_path}.")
    #     except FileNotFoundError:
    #         logging.warning(f"Full job postings file not found at {full_job_post_data_path}. Job match tool disabled.")
    #         return None
    #     except json.JSONDecodeError:
    #         logging.warning(
    #             f"Error decoding full job postings JSON from {full_job_post_data_path}. Job match tool disabled.")
    #         return None
    #     # --------------------- 새로 추가/변경 끝 ---------------------
    #
    #     # resume.json 파일 로드 (이전과 동일)
    #     resume_path = BASE_DIR / "Data_Files" / "resume.json"
    #     try:
    #         with open(resume_path, 'r', encoding='utf-8') as f:
    #             resume_data = json.load(f)[0]  # resume.json은 리스트의 첫 번째 요소라고 가정
    #     except FileNotFoundError:
    #         logging.warning(f"Resume file not found at {resume_path}. Job match tool disabled.")
    #         return None
    #     except json.JSONDecodeError:
    #         logging.warning(f"Error decoding resume JSON from {resume_path}. Job match tool disabled.")
    #         return None
    #
    #     # Sentence-BERT 모델 로드 (이전과 동일)
    #     try:
    #         sbert_model = SentenceTransformer(AgentTools.SBERT_MODEL_NAME)
    #     except Exception as e:
    #         logging.error(f"Failed to load SBERT model {AgentTools.SBERT_MODEL_NAME}: {e}. Job match tool disabled.")
    #         return None
    #
    #     # 이력서 텍스트 결합 및 정보 추출 (이전과 동일)
    #     resume_combined_text = AgentTools._combine_resume_text_for_match(resume_data)
    #     resume_experience_match = re.search(r'(\d+)\s*년', resume_data.get('경력', ''))
    #     resume_experience_years = int(resume_experience_match.group(1)) if resume_experience_match else 0
    #     resume_tech_stacks = [ts.strip().lower() for ts in resume_data.get("기술 스택", "").split(',') if ts.strip()]
    #
    #     # 이력서 임베딩 생성 (이전과 동일)
    #     resume_embedding = sbert_model.encode(resume_combined_text, convert_to_tensor=True)
    #
    #     # 툴이 실행될 때 호출될 함수
    #     def _find_best_job_match(query: str) -> str:
    #         # Retriever에 전달할 검색 쿼리. 사용자 질문이나 이력서 내용을 활용
    #         search_query_for_retriever = query if query and query != "이력서에 가장 적합한 공고 추천" else resume_combined_text
    #
    #         # --- RAG의 Retriever를 사용하여 문서 검색 ---
    #         # DeprecationWarning 해결: .get_relevant_documents() 대신 .invoke() 사용
    #         #retrieved_documents = retriever.invoke(search_query_for_retriever)
    #         retrieved_documents = retriever.get_relevant_documents(
    #             search_query_for_retriever,
    #             k=30  # ← 이 툴에서만 원하는 k 값
    #         )
    #
    #         job_posts_for_matching = {}  # 매칭에 사용할 채용 공고 딕셔너리
    #
    #         # --------------------- 새로 추가/변경 시작 ---------------------
    #         # 검색된 Document에서 ID를 추출하여 원본 full_job_post_data에서 해당 공고를 찾습니다.
    #         for doc in retrieved_documents:
    #             # Document의 metadata에 'id'가 직접 저장되어 있다고 가정합니다.
    #             # 또는 doc.page_content에서 정규표현식 등으로 ID를 추출해야 할 수도 있습니다.
    #             job_id = doc.metadata.get('id', None)  # ID는 보통 int나 string으로 저장될 수 있으므로 str 변환 고려
    #
    #             # 만약 metadata에 ID가 없다면, page_content에서 ID를 추출하는 로직을 추가할 수 있습니다.
    #             # 예시: 'ID: 12345' 형태라면 re.search(r'ID:\s*(\d+)', doc.page_content)
    #             if not job_id:
    #                 # 대안: page_content에 JSON 문자열이 있다면 이를 파싱하여 ID를 얻기 시도
    #                 try:
    #                     # doc.page_content가 실제 JSON 문자열이라고 가정 (로그에 나온 "제목: HyperCLOVA X..."는 JSON이 아님)
    #                     # 만약 RAG 파이프라인이 Document.page_content에 JSON을 저장한다면 이 코드 사용
    #                     # parsed_content = json.loads(doc.page_content)
    #                     # job_id = parsed_content.get('id')
    #                     pass  # 현재는 page_content가 텍스트이므로 이 로직은 건너뜁니다.
    #                 except json.JSONDecodeError:
    #                     logging.debug(f"Document page_content is not valid JSON: {doc.page_content[:50]}...")
    #                     # JSON이 아니므로 다음 ID 추출 시도 또는 건너뛰기
    #
    #             # 추출된 ID가 유효하고, full_job_post_data에 해당 공고가 있는지 확인
    #             if job_id and str(job_id) in full_job_post_data:  # full_job_post_data의 키는 문자열 ID
    #                 job_posts_for_matching[str(job_id)] = full_job_post_data[str(job_id)]
    #             else:
    #                 logging.warning(
    #                     f"Retrieved document with ID '{job_id}' not found in full_job_post_data or ID missing. Document metadata: {doc.metadata.get('source', '')[:50]}")
    #         # --------------------- 새로 추가/변경 끝 ---------------------
    #
    #         if not job_posts_for_matching:  # 변경: job_posts_from_retriever -> job_posts_for_matching
    #             return "이력서와 관련된 채용 공고를 찾을 수 없었습니다. 검색 범위를 넓히거나 이력서를 보완해 보세요."
    #
    #         logging.info(f"Retrieved {len(job_posts_for_matching)} job postings from RAG system for matching.")  # 변경
    #
    #         best_match_job_id = None
    #         highest_score = -1.0
    #         job_post_scores = {}
    #
    #         # 이제 RAG가 검색해온 공고들(job_posts_for_matching)에 대해서만 점수 계산
    #         for job_id, job_post in job_posts_for_matching.items():  # 변경: job_posts_from_retriever -> job_posts_for_matching
    #             job_post_combined_text = AgentTools._combine_job_post_text_for_match(job_post)
    #             job_post_embedding = sbert_model.encode(job_post_combined_text, convert_to_tensor=True)
    #
    #             # 코사인 유사도
    #             cosine_similarity_score = util.cos_sim(resume_embedding, job_post_embedding).item()
    #
    #             # 경력 매칭 점수 (이전과 동일)
    #             required_min_exp = job_post.get("요구 최소 경력", 0)
    #             required_max_exp = job_post.get("요구 최대 경력", 999)
    #
    #             experience_match_score = 0.0
    #             if required_min_exp <= resume_experience_years <= required_max_exp:
    #                 experience_match_score = 1.0
    #             elif resume_experience_years > required_max_exp:
    #                 experience_match_score = 0.5
    #             elif resume_experience_years < required_min_exp:
    #                 experience_match_score = 0.0
    #
    #             # 기술 스택 매칭 점수 (이전과 동일)
    #             job_tech_stacks = [ts.strip().lower() for ts in job_post.get("기술 스택", []) if ts.strip()]
    #             tech_stack_score = 0.0
    #             if job_tech_stacks:
    #                 matched_tech_count = sum(1 for tech in resume_tech_stacks if tech in job_tech_stacks)
    #                 tech_stack_score = matched_tech_count / len(job_tech_stacks)
    #             else:
    #                 tech_stack_score = 0.5  # Neutral if job has no explicit tech stacks
    #
    #             # 직군 매칭 점수 (AgentTools.RESUME_JOB_ROLE_KEYWORDS 사용) (이전과 동일)
    #             job_직군 = job_post.get("직군", "").lower()
    #             직무_list = [d.lower() for d in job_post.get("직무", []) if d.strip()]
    #             job_role_match_score = 0.0
    #
    #             if any(keyword in job_직군 for keyword in AgentTools.RESUME_JOB_ROLE_KEYWORDS) or \
    #                     any(any(keyword in specific_job for keyword in AgentTools.RESUME_JOB_ROLE_KEYWORDS) for
    #                         specific_job in 직무_list):
    #                 job_role_match_score = 1.0
    #             elif "마케팅" in job_직군 or "광고" in job_직군 or "기획" in job_직군:
    #                 job_role_match_score = 0.0
    #             else:
    #                 job_role_match_score = 0.3
    #
    #             # 최종 가중치 합산 점수 (이전과 동일)
    #             current_score = (
    #                     cosine_similarity_score * AgentTools.WEIGHT_COSINE_SIMILARITY +
    #                     experience_match_score * AgentTools.WEIGHT_EXPERIENCE_MATCH +
    #                     tech_stack_score * AgentTools.WEIGHT_TECH_STACK_MATCH +
    #                     job_role_match_score * AgentTools.WEIGHT_JOB_ROLE_MATCH
    #             )
    #
    #             job_post_scores[job_id] = {
    #                 "title": job_post.get("제목", "제목 없음"),
    #                 "score": current_score,
    #                 "details": {
    #                     "cosine_similarity": cosine_similarity_score,
    #                     "experience_match": experience_match_score,
    #                     "tech_stack_match": tech_stack_score,
    #                     "job_role_match": job_role_match_score
    #                 }
    #             }
    #
    #             if current_score > highest_score:
    #                 highest_score = current_score
    #                 best_match_job_id = job_id
    #
    #         # 결과 포맷팅 (이전과 동일)
    #         if not best_match_job_id:
    #             return "검색된 채용 공고 중에서 이력서에 가장 적합한 공고를 찾을 수 없습니다."
    #
    #         best_match_data = job_post_scores[best_match_job_id]
    #         sorted_jobs = sorted(job_post_scores.items(), key=lambda item: item[1]['score'], reverse=True)
    #         top_n_results = []
    #         for i, (job_id, data) in enumerate(sorted_jobs[:3]):  # 상위 3개까지
    #             top_n_results.append(f"{i + 1}. '{data['title']}' (ID: {job_id})\n   - 점수: {data['score']:.4f}")
    #             for detail_key, detail_value in data['details'].items():
    #                 top_n_results.append(f"     - {detail_key.replace('_', ' ').title()}: {detail_value:.4f}")
    #
    #         result_str = (
    #             f"사용자 이력서에 가장 적합한 채용 공고를 찾았습니다:\n\n"
    #             f"최고 매칭 공고: '{best_match_data['title']}' (ID: {best_match_job_id})\n"
    #             f"종합 적합도 점수: {best_match_data['score']:.4f}\n"
    #             f"상세 점수:\n"
    #         )
    #         for detail_key, detail_value in best_match_data['details'].items():
    #             result_str += f"  - {detail_key.replace('_', ' ').title()}: {detail_value:.4f}\n"
    #
    #         if len(sorted_jobs) > 1:
    #             result_str += "\n--- 상위 추천 공고 목록 ---\n" + "\n".join(top_n_results)
    #
    #         return result_str
    #
    #     return Tool.from_function(
    #         func=_find_best_job_match,
    #         name="find_best_job_match",
    #         description="사용자의 이력서와 관련 채용 공고를 비교하여 가장 적합한 공고를 찾아 추천합니다. '가장 적합한 공고', '내 이력서에 맞는 공고', '추천 공고' 등의 질문에 사용합니다.",
    #     )
    #
    # # 텍스트 결합 함수 재사용 (툴 내부에서만 사용할 Private 함수)
    # @staticmethod
    # def _combine_resume_text_for_match(resume):
    #     combined_text = ""
    #     combined_text += f"자기소개: {resume.get('자기소개', '')}. "
    #     combined_text += f"경력: {resume.get('경력', '')}. "
    #     combined_text += f"기술 스택: {resume.get('기술 스택', '')}. "
    #     combined_text += f"프로젝트 경험: {resume.get('프로젝트 경험', '')}. "
    #     combined_text += f"수상 경력: {resume.get('수상 경력', '')}. "
    #     combined_text += f"기타: {resume.get('기타', '')}."
    #     return combined_text.strip()
    #
    # @staticmethod
    # def _combine_job_post_text_for_match(job_post):
    #     combined_text = ""
    #     combined_text += f"제목: {job_post.get('제목', '')}. "
    #     combined_text += f"회사 소개: {job_post.get('회사 소개', '')}. "
    #     combined_text += f"주요 업무: {' '.join(filter(None, job_post.get('주요 업무', [])))}. "
    #     combined_text += f"자격 요건: {' '.join(filter(None, job_post.get('자격 요건', [])))}. "
    #     combined_text += f"우대 사항: {' '.join(filter(None, job_post.get('우대 사항', [])))}. "
    #     return combined_text.strip()


# """
# Agent 툴을 관리합니다.
# """
# import logging
# import os
# from pathlib import Path
# from typing import List, Optional, Any
# import pandas as pd
# from langchain.agents import Tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
# from langchain_community.agent_toolkits import FileManagementToolkit
# import json
#
# BASE_DIR = Path(__file__).resolve().parent.parent
#
# class AgentTools:
#     # ────────────────────── 공통 PROMPT 정의 ──────────────────────
#     prompt_content = """
#     # 롤 & 톤
#     당신은 신뢰할 수 있는 취업 전문가 AI 코치입니다.
#     모든 답변은 친절하지만 간결한 한국어로 작성하세요.
#     필요하면 표·리스트·예시 코드를 사용해도 좋습니다.
#
#     # 툴 사용 규칙
#     1. 채용 공고, 취업 뉴스, 면접 정보 등 **문서**에서 답을 찾을 수 있을 것 같으면
#        반드시 `document_search` 툴을 먼저 호출한다.
#     2. `document_search`가 결과를 0개 반환하거나(embeddings empty) 내용이 질문과 무관하면
#        보강 정보가 필요하다고 판단하고 `search_web` 툴을 호출한다.
#     3. 사용자가 “평균”, “통계”, “그래프”, “필터” 같은 키워드로
#        **데이터 분석**을 요구하면 `job_dataframe_analysis` 툴을 호출한다.
#     4. 사용자가 “이미지 만들어”, “로고 그려” 같이 시각 생성 요청을 하면 `generate_image`.
#     5. “내 이력서” 또는 “내 경력”을 직접 언급하면 `resume_qa`.
#     6. 위 조건에 맞지 않으면 툴을 사용하지 않고 자체 지식으로 답한다.
#
#     # 워크플로 예시 (Few-shot)
#     ## 예시 1 – 문서 검색 성공
#     User: "삼성전자 백엔드 신입 연봉 정보 알려줘"
#     Assistant:
#     - Thought: "채용 공고 문서에 있을 가능성이 높다"
#     - Action: document_search {{ "query": "삼성전자 백엔드 신입 연봉" }}
#
#     ## 예시 2 – 문서에 없음 → 웹 검색
#     User: "내년 공무원 채용 일정 알려줘"
#     Assistant:
#     - Thought: "사내 문서엔 없음, 웹 최신 정보 필요"
#     - Action: search_web {{ "query": "2025 공무원 채용 일정" }}
#
#     ## 예시 3 – 데이터프레임 통계
#     User: "지난 3개월 동안 서울에서 올라온 프론트엔드 평균 연봉 그래프로 보여줘"
#     Assistant:
#     - Thought: "DataFrame 분석 필요"
#     - Action: job_dataframe_analysis {{ "query": "지난 3개월 서울 프론트엔드 평균 연봉 그래프" }}
#
#     ## 예시 4 – 이력서 피드백
#     User: "내 이력서에서 개선할 점 알려줘"
#     Assistant:
#     - Thought: "resume_qa 툴 사용"
#     - Action: resume_qa {{ "question": "내 이력서 개선 포인트" }}
#
#     대답을 작성할 땐 최종적으로
#     **"Answer:"** 섹션에서 사용자에게 보이는 답을 작성하고,
#     필요하면 **"Sources:"** 항목에 참고 툴 결과를 간단히 인용한다.
#     """
#
#     # ChatPromptTemplate 생성: 올바른 튜플 형태로 지정
#     PROMPT = ChatPromptTemplate.from_messages([
#         ("system", prompt_content),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ])
#     # --------------------------------------------------
#     @staticmethod
#     def get_tools(*, retriever: Any, llm) -> List[Tool]:
#         tools = [
#             AgentTools._create_search_tool(),
#             AgentTools._create_retriever_tool(retriever),
#             AgentTools._create_dataframe_tool(llm),
#             AgentTools._create_image_generation_tool(),
#             AgentTools._create_resume_tool(llm)
#         ]
#
#         # ▒ file-management 툴 (여러 개) 추가
#         file_tools = AgentTools._create_file_management_tool()
#         if file_tools:                      # ❷ None 체크
#             tools.extend(file_tools)
#
#         return [t for t in tools if t is not None]
#
#     # ────────────────────── tool builders ──────────────────────
#     # 1) 웹 검색
#     @staticmethod
#     def _search_web(query: str) -> str:
#         return TavilySearchResults(k=6).run(query)
#
#     @staticmethod
#     def _create_search_tool() -> Tool:
#         return Tool.from_function(
#             func=AgentTools._search_web,
#             name="search_web",
#             description="문서에 없으면 최신 웹 검색을 수행한다.",
#         )
#
#     # 2) RAG 리트리버
#     @staticmethod
#     def _create_retriever_tool(retriever: Any) -> Optional[Tool]:
#         if retriever is None:
#             logging.warning("Retriever is None – document_search 툴 비활성화.")
#             return None
#
#         return create_retriever_tool(
#             retriever=retriever,
#             name="document_search",
#             description="업로드된 채용 공고 벡터 DB에서 관련 내용을 찾는다.",
#         )
#
#     # 3) DataFrame 분석
#     @staticmethod
#     def _create_dataframe_tool(llm) -> Optional[Tool]:
#         if llm is None:
#             logging.warning("LLM is None – dataframe 툴 비활성화.")
#             return None
#
#         data_path = BASE_DIR / "Data_Files" / "wanted_detail_improve_20250604.json"
#         if not data_path.exists():
#             logging.warning("DataFrame 파일을 찾을 수 없음 – dataframe 툴 비활성화.")
#             return None
#
#         df = pd.read_json(data_path).T
#
#         pandas_agent = create_pandas_dataframe_agent(
#             llm,
#             df,
#             verbose=False,
#             allow_dangerous_code=True,
#             prefix="너는 채용공고 데이터 분석 전문가다.",
#         )
#
#         return Tool.from_function(
#             func=lambda q: pandas_agent.run(q),
#             name="job_dataframe_analysis",
#             description="채용공고 DataFrame을 이용해 통계·시각화를 수행한다.",
#         )
#
#     @staticmethod
#     def _create_image_generation_tool() -> Optional[Tool]:
#         """
#         DALL-E 이미지 생성 툴을 생성합니다.
#         현재는 사용하지 않지만, 필요시 활성화할 수 있습니다.
#         """
#         if not os.getenv("OPENAI_API_KEY"):
#             logging.warning("DALL-E API 키가 설정되지 않았습니다. 이미지 생성 툴 비활성화.")
#             return None
#
#         dalle_tool = DallEAPIWrapper(
#             model="dall-e-3", # 생성 모델
#             size="1024x1024",
#             quality="standard", # 품질 설정 hd도 있음
#             n=1 # 생성할 이미지 개수
#         )
#         return Tool.from_function(
#             func=dalle_tool.run,
#             name="generate_image",
#             description="이미지를 생성합니다.",
#         )
#
#     @staticmethod
#     def _create_file_management_tool() -> Optional[List[Tool]]:  # ❸ 타입 변경
#         """
#         파일 시스템을 다루는 여러 Tool(write_file, read_file …)을 돌려준다.
#         """
#         working_directory = BASE_DIR / "Data_Files"
#
#         # 필요 없다면 selected_tools에서 빼거나 함수 전체를 주석 처리해도 OK
#         file_tools = FileManagementToolkit(
#             root_dir= str(working_directory),
#             selected_tools=[
#                 "write_file", "read_file",
#                 "list_directory", "move_file",
#             ],
#         ).get_tools()
#
#         return file_tools
#
#     @staticmethod
#     def _create_resume_tool(llm) -> Tool:
#         """
#         이력서를 LLM 컨텍스트에 넣어 질의-응답해 주는 툴.
#         """
#         data_path = BASE_DIR / "Data_Files" / "resume.json"
#         with open(data_path, encoding="utf-8") as f:
#             resume = json.load(f)
#
#         # ▶ 필요한 형태(평문)로 합치기
#         if isinstance(resume, dict):
#             resume_text = "\n".join(f"{k}: {v}" for k, v in resume.items())
#         elif isinstance(resume, list):
#             # 리스트 요소가 dict면 합치고, 아니면 str로 변환
#             lines = []
#             for idx, item in enumerate(resume, 1):
#                 if isinstance(item, dict):
#                     lines.append(f"--- 항목 {idx} ---")
#                     lines.extend(f"{k}: {v}" for k, v in item.items())
#                 else:
#                     lines.append(str(item))
#             resume_text = "\n".join(lines)
#         else:
#             resume_text = str(resume)
#
#         def _answer_about_resume(question: str) -> str:
#             prompt = f"""너는 구직자의 이력서 코치야.
#             이력서 원문:
#             ---
#             {resume_text}
#             ---
#             사용자의 질문: {question}
#             """
#
#             return llm.invoke(prompt).content  # LLM 응답 텍스트
#
#         return Tool.from_function(
#             func=_answer_about_resume,
#             name="resume_qa",
#             description="이력서 내용에 대해 질문하면, 조언이나 답변을 해 줍니다.",
#         )