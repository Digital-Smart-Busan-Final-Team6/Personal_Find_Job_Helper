# main.py
# 이 파일은 제가 작성한 코드입니다.
# FastAPI를 사용하여 챗봇 API를 제공합니다.

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

# 1단계에서 만든 Agent 생성 함수를 import 합니다.
from .Agent_Manager import create_agent_chain
# 팀원분이 만드신 스트림 파서를 그대로 활용합니다.
from langchain_teddynote.messages import AgentCallbacks, AgentStreamParser

# FastAPI 앱 생성
app = FastAPI()

# 서버 시작 시, 단 한번만 Agent 체인을 생성하여 전역 변수에 저장
agent_chain = create_agent_chain()

# 클라이언트로부터 받을 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    session_id: str  # 각 사용자를 구분하기 위한 세션 ID
    message: str     # 사용자가 보낸 메시지

# 스트리밍 응답을 위한 비동기 제너레이터 함수
async def stream_generator(request: ChatRequest):
    """
    Agent의 스트리밍 출력을 받아서 클라이언트에게 전달하는 제너레이터
    """
    # 팀원분의 AgentStreamParser를 여기서 활용합니다.
    # 단, print하는 대신 yield로 데이터를 전송하도록 콜백을 수정합니다.
    
    # queue를 사용하여 stream의 출력을 비동기적으로 처리합니다.
    queue = asyncio.Queue()

    # 콜백 함수 정의: Agent의 각 단계 결과를 queue에 넣습니다.
    def result_callback(result: str) -> None:
        # 최종 답변을 queue에 넣습니다.
        queue.put_nowait(f"data: {result}\n\n")

    def tool_callback(tool) -> None:
        # 도구 사용 정보를 queue에 넣습니다. (디버깅 또는 UI 표시에 유용)
        # queue.put_nowait(f"data: [도구 사용: {tool.get('tool')}]\n\n")
        pass # 실제 서비스에서는 보통 생략

    def observation_callback(observation) -> None:
        # 관찰 정보를 queue에 넣습니다. (디버깅 또는 UI 표시에 유용)
        # queue.put_nowait(f"data: [관찰: {observation.get('observation')[0]}]\n\n")
        pass # 실제 서비스에서는 보통 생략

    agent_callbacks = AgentCallbacks(
        tool_callback=tool_callback,
        observation_callback=observation_callback,
        result_callback=result_callback,
    )
    agent_stream_parser = AgentStreamParser(agent_callbacks)

    # 백그라운드에서 agent_chain.stream을 실행하고 결과를 파싱하여 queue에 넣습니다.
    async def run_agent_in_background():
        # agent_chain.stream은 동기 함수이므로 asyncio.to_thread로 실행
        async for step in await asyncio.to_thread(
            agent_chain.stream,
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        ):
            agent_stream_parser.process_agent_steps(step)
        await queue.put(None) # 스트림 종료 신호

    asyncio.create_task(run_agent_in_background())

    # queue에서 결과를 꺼내 클라이언트로 스트리밍합니다.
    while True:
        item = await queue.get()
        if item is None: # 종료 신호를 받으면
            break
        yield item

# 챗봇 엔드포인트 정의
@app.post("/chat")
def chat(request: ChatRequest):
    """
    사용자 메시지를 받아 Agent의 답변을 실시간 스트리밍으로 반환합니다.
    """
    return StreamingResponse(stream_generator(request), media_type="text/event-stream")

# 서버 실행 방법:
# 터미널에서 uvicorn main:app --reload --host 0.0.0.0 --port 8000 입력