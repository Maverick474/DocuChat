import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .rag import answer_question, ingest_document
    from .vector_store import VectorStore
except ImportError:
    from rag import answer_question, ingest_document
    from vector_store import VectorStore


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

app = FastAPI(title="DocuChat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    source: str
    page: str
    chunk: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_stored: int


def get_store():
    try:
        return VectorStore()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def validate_file(file, content):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="The uploaded file type is invalid.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit.",
        )


@app.get("/health")
def health():
    configured = all(
        os.getenv(name)
        for name in ("OPENROUTER_API_KEY", "SUPABASE_URL", "SUPABASE_API_KEY")
    )
    return {"status": "ok", "services_configured": configured}


@app.post("/upload", response_model=UploadResponse)
async def upload(file=File(...)):
    content = await file.read()
    validate_file(file, content)

    try:
        result = ingest_document(content, file.filename, get_store())
        return UploadResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {error}") from error


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = answer_question(
            request.question.strip(),
            get_store(),
            request.top_k,
            request.document_id,
        )
        return ChatResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {error}") from error
