from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .model_service import ModelService
from .schemas import ChatRequest, ChatResponse

service = ModelService(config.MODEL_PATH, config.ADAPTER_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()  # load once at startup, not per request
    yield


app = FastAPI(title="Private AI Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": service.model is not None}


@app.get("/api/model-info")
def model_info():
    return service.info()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    return service.chat(messages, req.max_tokens, req.temperature)
