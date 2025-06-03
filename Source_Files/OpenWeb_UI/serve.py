# Source_Files/OpenWeb_UI/serve.py
import sys, pathlib, os
from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes

# 1) PROJECT_ROOT 를 PYTHONPATH 에 추가
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 2) 체인만 반환하도록 빌드
from Source_Files.Model.Run_Model import main as build_pipeline
chain = build_pipeline(return_chain_only=True)

# 3) FastAPI 인스턴스 생성 & LangServe 라우트 붙이기
app = FastAPI()
add_routes(app, chain, path="/v1")

# 4) gpt-4o-mini 모델 등록
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "gpt-4o-mini", "object": "model", "owned_by": "local"}]
    }

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {"id": model_id, "object": "model", "owned_by": "local"}

# 5) Chat Completions 핸들러
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    question = req.messages[-1].content
    answer = chain.invoke(question)
    return {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
            "index": 0
        }],
        "usage": {}
    }

import debugpy
# 0.0.0.0:5678 포트로 디버거 연결 대기
debugpy.listen(("0.0.0.0", 5678))
print("⏳ Waiting for debugger to attach on port 5678...")
# 클라이언트가 붙을 때까지 블록
debugpy.wait_for_client()

# 6) 로컬 개발용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
