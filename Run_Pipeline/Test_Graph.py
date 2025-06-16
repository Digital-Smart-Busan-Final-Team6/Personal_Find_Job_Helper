from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph
from Run_Pipeline.Document_Loader import DocumentLoader
from Run_Pipeline.Document_Splitter import DocumentSplitter
from Run_Pipeline.Embedding_DB import EmbeddingDB
from Run_Pipeline.Retriever_Builder import RetrieverBuilder
from State import *
from Env_Loader import EnvLoader
from pathlib import Path
import os

EnvLoader.load_local()
BASE_DIR = Path(__file__).resolve().parent.parent

kind = "json"
file_path = BASE_DIR / os.getenv("DATA_PATH")
chunk_size = 1000
overlap_size = 50
persist_dir = BASE_DIR / os.getenv("DATA_PATH") / f"{kind}_{chunk_size}"
device = "mps"
retriever_mode = 1
k = 3
engine_num = 1
backend_num = 1
# ③ 문서 로드
loader = DocumentLoader(file_path, kind)  # 일반 파일들을 Document형태로 변환해서 다 불러옴
docs = loader.load()

# ④ 문서 분할
splitter = DocumentSplitter(chunk_size=chunk_size, overlap=overlap_size)  # 이미 분할된 문서가 있다면 로드하고 아니면 새로 스플릿함
chunks = splitter.split(docs, cache_dir=file_path)

# ⑤ 임베딩 & DB 생성/로드
embed_db = EmbeddingDB(model_name="nlpai-lab/KURE-v1", device=device,
                       persist_dir=persist_dir)  # 이미 임베딩된 DB가 있다면 로드하고 아니면 새로 임베딩함
db = embed_db.get_or_create_db(chunks)

# ⑥ 리트리버 빌드
retriever_builder = RetrieverBuilder(db=db, docs=chunks, k=k, mode=retriever_mode)  # 리트리버 빌더 생성
retriever = retriever_builder.build()

workflow = StateGraph(GraphState)

workflow.add_node('retreive', retreive)

workflow.add_node('rewrite_query', rewrite_query)

workflow.add_node('gpt 요청', llm_gpt_execute)
workflow.add_node('claude 요청', llm_claude_execute)
workflow.add_node('GPT_relevance_check', relevance_check)
workflow.add_node('Claude_relevance_check', relevance_check)
workflow.add_node("결과 종합", sum_up)

# 노드 연결
workflow.add_edge('retreive', 'gpt 요청')
workflow.add_edge('retreive', 'claude 요청')

workflow.add_edge('rewrite_query', 'retreive')

workflow.add_edge('gpt 요청', 'GPT_relevance_check')
workflow.add_edge('GPT_relevance_check', '결과 종합')
workflow.add_edge('claude 요청', 'Claude_relevance_check')
workflow.add_edge('Claude_relevance_check', '결과 종합')

# workflow.add_edge('결과 종합', END)

# workflow.add_conditional_edges(
#     "결과 종합",
#     decision,
#     {
#         "재검색": "retreive",
#         "종료": END,
#     }
# )

workflow.add_conditional_edges(
    '결과 종합',
    decision,
    {
        '재검색': 'rewrite_query',
        '종료': END,
    },
)

workflow.set_entry_point('retreive')

# 워크플로우 시각화
memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

visualize_graph(app)
