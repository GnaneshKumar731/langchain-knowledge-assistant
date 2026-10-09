import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import ingest_pdf, initialize_agent, sync_deleted_pdfs

load_dotenv()

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app):
    print("Starting AI Knowledge Assistant...")

    agent, vector_store = initialize_agent()

    sync_deleted_pdfs(vector_store)
    
    app.state.agent = agent
    app.state.vector_store = vector_store

    print("AI Knowledge Assistant started successfully.")

    yield

    print("Shutting down AI Knowledge Assistant...")

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_request_time(request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)

        process_time = time.time() - start_time

        print(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} "
            f"in {process_time:.2f}s"
        )

        return response

    except Exception as e:
        process_time = time.time() - start_time

        print(
            f"{request.method} {request.url.path} "
            f"-> ERROR "
            f"in {process_time:.2f}s: {e}"
        )

        raise

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    decode_responses=True
)



def get_conversation_history(session_id):
    data = redis_client.get(f"chat:{session_id}")

    if data:
        return json.loads(data)

    return []


def save_conversation_history(session_id, history):
    redis_client.set(
        f"chat:{session_id}",
        json.dumps(history)
    )


class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def home():

    return {
        "message": "AI Knowledge Assistant API is running"
    }


@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        filename = Path(file.filename).name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        file_path = DOCUMENTS_DIR / filename

        with open(file_path, "wb") as output_file:
            content = await file.read()
            output_file.write(content)

        ingest_pdf(
            file_path,
            request.app.state.vector_store
        )

        return {
            "message": f"{file.filename} uploaded and ingested successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed."
        )
    

@app.post("/chat", response_model=ChatResponse)
def chat(
    chat_request: ChatRequest,
    request: Request
):
    try:
        history = get_conversation_history(chat_request.session_id)

        history.append({
            "role": "user",
            "content": chat_request.message
        })

        save_conversation_history(chat_request.session_id, history)

        response = request.app.state.agent.invoke({
            "messages": history
        })

        structured_response = response["structured_response"]

        answer = structured_response.answer
        sources = structured_response.sources

        history.append({
            "role": "assistant",
            "content": answer
        })

        save_conversation_history(
            chat_request.session_id,
            history
        )

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print(f"Chat error: {e}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request."
        )
