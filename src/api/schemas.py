"""
schema.py
----------
Pydantic models for API request and response validation.
FastAPI uses these to automatically validate incoming requests and serialize outgoing responses.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    pipeline: str = "custom"
    top_k: int = 3


class ChunkResult(BaseModel):
    text: str
    score: float
    source: str
    reranker_score: float
    chunk_id: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    pipeline: str
    retrieved_chunks: list[ChunkResult]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    score: float
    pipeline: str
    comment: str = ""
    retrieved_chunks: list = []
