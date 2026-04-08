"""
routes.py
---------
API route definitions for the RAG MLOps system.

Endpoints:
    GET     /health     - check if the service is running
    POST    /query      - send a question and get an answer
    POST    /feedback   - submit a score for an answer
"""

import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from src.api.schemas import QueryRequest, QueryResponse, ChunkResult, FeedbackRequest
import os
from src.rag.pipeline import RAGPipeline
from src.llamaindex.pipeline import LlamaIndexPipeline
from src.training.scheduler import check_threshold_trigger
from src.feedback.collector import (
    save_feedback, get_feedback_count,
    save_document, get_all_documents,
    delete_document, document_exists,get_feedback_stats
)
from src.rag.loader import load_documents, split_documents
from src.training.scheduler import get_last_trained_count  # we'll add this below


router = APIRouter()
DATA_DIR = "data"
custom_pipeline = RAGPipeline(data_dir=DATA_DIR)
llama_index_pipeline = LlamaIndexPipeline(data_dir=DATA_DIR)

@router.get("/feedback/stats")
def feedback_stats():
    stats = get_feedback_stats()
    last_trained = get_last_trained_count()
    stats["since_last_training"] = stats["total"] - last_trained
    return stats

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
    except HTTPException:
        raise
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

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Only allow supported file types
    allowed_extensions = {".docx", ".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Check if file already exists
    if document_exists(file.filename):
        raise HTTPException(status_code=400, detail=f"'{file.filename}' is already uploaded.")

    # Save file to data/ folder
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Index the new document
    try:
        docs = load_documents(file_path)
        nodes = split_documents(docs)
        custom_pipeline.indexer.build(nodes, custom_pipeline.embedder, source=file.filename)
        custom_pipeline.indexer.save()
        llama_index_pipeline.build_from_disk()
    except Exception as e:
        # Clean up file if indexing fails
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    # Save metadata to PostgreSQL
    save_document(
        filename=file.filename,
        file_path=file_path,
        chunk_count=len(nodes),
    )

    return {
        "status": "ok",
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "chunk_count": len(nodes),
    }


@router.get("/documents")
def list_documents():
    docs = get_all_documents()
    return [
        {
            "filename": d.filename,
            "file_path": d.file_path,
            "chunk_count": d.chunk_count,
            "uploaded_at": d.updated_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/documents/{filename}")
def delete_document_endpoint(filename: str):
    # Check it exists in DB
    if not document_exists(filename):
        raise HTTPException(status_code=404, detail=f"'{filename}' not found.")

    # Remove from FAISS index and rebuild
    removed = custom_pipeline.indexer.delete_by_source(filename, custom_pipeline.embedder)

    # Delete physical file
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete from PostgreSQL
    delete_document(filename)

    return {
        "status": "ok",
        "message": f"'{filename}' deleted successfully.",
        "chunks_removed": removed,
    }