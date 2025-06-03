import os
import time
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from openai import AsyncOpenAI
from langserve import add_routes
from app import get_chain  # 자신의 get_chain() 경로에 맞게 조정

# ── 1. 환경 변수 로드 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ── 3. FastAPI 앱 생성 ────────────────────────────────────────────
app = FastAPI()

# ── 6. LangServe Playground 등록 (→ /playground/) ─────────────────
chain = get_chain()

from fastapi.staticfiles import StaticFiles

# 1) custom_playground 디렉토리를 "/playground" 경로에 먼저 마운트
# app.mount(
#     "/playground",
#     StaticFiles(directory="Custom_Playground/dist", html=True),
#     name="Custom_Playground",
# )

# 2) 그 다음에 LangServe 기본 라우트 마운트
add_routes(app, chain, path="/playground")


add_routes(app, get_chain())
