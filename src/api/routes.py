"""
routes.py
---------
API route definitions for the RAG MLOps system.

Endpoints:
    GET     /health     - check if the service is running
    POST    /query      - send a question and get an answer
    POST    /feedback   - submit a score for an answer
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, ChunkResult, FeedbackRequest
from src.rag.pipeline import RAGPipeline
from src.llamaindex.pipeline import LlamaIndexPipeline

router = APIRouter()

FILE_PATH = "data/11OSproject.docx"
custom_pipeline = RAGPipeline(file_path=FILE_PATH)
llama_index_pipeline = LlamaIndexPipeline(file_path=FILE_PATH)


@router.get("/health")
def health_check():
    return {"message": "OK"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        if request.pipeline == "custom":
            result = custom_pipeline.query(request.question)
        elif request.pipeline == "llamaindex":
            result = llama_index_pipeline.query(request.question)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pipeline: {request.pipeline}. Use 'custom' or 'llamaindex'."
            )
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            pipeline=request.pipeline,
            retrieved_chunks=[
                ChunkResult(text=c["text"], score=c["score"])
                for c in result["retrieved_chunks"]
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.score < 1 or request.score > 5:
        raise HTTPException(
            status_code=400,
            detail="Score must be between 1 and 5."
        )
    print(f"Feedback received: question='{request.question}' score={request.score}")
    return {"status": "ok", "message": "Feedback received"}
