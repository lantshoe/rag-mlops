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
from src.feedback.collector import save_feedback, get_feedback_count
from src.rag.pipeline import RAGPipeline
from src.llamaindex.pipeline import LlamaIndexPipeline
from src.training.scheduler import check_threshold_trigger

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
                ChunkResult(text=c["text"], score=c["score"],
                            source=c["source"], reranker_score=c.get("reranker_score"),
                            chunk_id=c["chunk_id"])
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
    save_feedback(
        question=request.question,
        answer=request.answer,
        score=request.score,
        pipeline=request.pipeline,
        comment=request.comment,
        retrieved_chunks = request.retrieved_chunks
    )
    check_threshold_trigger()

    count = get_feedback_count()
    return {
        "status": "ok",
        "message": "Feedback saved",
        "total_feedback_count": count
    }