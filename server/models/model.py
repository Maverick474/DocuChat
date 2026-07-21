from pydantic import BaseModel, Field

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
