"""
main.py
-------
FastAPI application query point.
registers all routes and starts the server.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""


from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="RAG MLOps System",
    description="A RAG pipeline with feedback loop and MLOps integration",
    version="0.1.0"
)

app.include_router(router)