import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_embedder():
    """A fake embedder that returns random vectors."""
    embedder = MagicMock()
    embedder.embed.return_value = np.random.rand(3, 384).astype("float32")
    return embedder

@pytest.fixture
def sample_chunks():
    return [
        {"text": "Virtual memory allows programs to use more memory than physically available.",
         "source": "test.docx", "chunk_id": 0, "score": 0.9},
        {"text": "Page tables map virtual addresses to physical addresses.",
         "source": "test.docx", "chunk_id": 1, "score": 0.8},
        {"text": "FreeRTOS is a real-time operating system for embedded systems.",
         "source": "test.docx", "chunk_id": 2, "score": 0.7},
    ]

@pytest.fixture
def sample_feedback():
    feedback = MagicMock()
    feedback.question = "What is virtual memory?"
    feedback.score = 5.0
    feedback.retrieved_chunks = '[{"text": "Virtual memory allows...", "source": "test.docx", "chunk_id": 0}]'
    return feedback

@pytest.fixture
def app_client():
    """FastAPI test client with all external dependencies mocked."""
    with patch("src.api.routes.custom_pipeline") as mock_custom, \
         patch("src.api.routes.llama_index_pipeline") as mock_llama, \
         patch("src.api.routes.save_feedback"), \
         patch("src.api.routes.get_feedback_count", return_value=10), \
         patch("src.api.routes.check_threshold_trigger"):

        mock_custom.query.return_value = {
            "question": "What is virtual memory?",
            "answer": "Virtual memory is a memory management technique.",
            "retrieved_chunks": [
                {"text": "Virtual memory allows...", "score": 0.9,
                 "reranker_score": 0.95, "source": "test.docx", "chunk_id": 0}
            ]
        }
        mock_llama.query.return_value = {
            "question": "What is virtual memory?",
            "answer": "Virtual memory is a memory management technique.",
            "retrieved_chunks": [
                {"text": "Virtual memory allows...", "score": 0.9,
                 "reranker_score": None, "source": "test.docx", "chunk_id": 0}
            ]
        }
        # must import after patch
        from src.api.main import app
        yield TestClient(app)