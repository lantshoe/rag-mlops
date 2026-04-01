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
from contextlib import asynccontextmanager
from src.training.scheduler import start_schedule_trigger
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts
    start_schedule_trigger()
    yield

app = FastAPI(
    title="RAG MLOps System",
    description="A RAG pipeline with feedback loop and MLOps integration",
    version="0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)