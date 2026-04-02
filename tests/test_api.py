import pytest
import io


def test_health_check(app_client):
    """GET /health should return 200 OK."""
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_query_custom_pipeline(app_client):
    """POST /query with custom pipeline should return answer and chunks."""
    response = app_client.post("/query", json={
        "question": "What is virtual memory?",
        "pipeline": "custom"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is virtual memory?"
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert data["pipeline"] == "custom"


def test_query_llamaindex_pipeline(app_client):
    """POST /query with llamaindex pipeline should return answer and chunks."""
    response = app_client.post("/query", json={
        "question": "What is virtual memory?",
        "pipeline": "llamaindex"
    })
    print("-----------",response.json())
    assert response.status_code == 200
    assert response.json()["pipeline"] == "llamaindex"


def test_query_invalid_pipeline(app_client):
    """POST /query with unknown pipeline should return 400."""
    response = app_client.post("/query", json={
        "question": "What is virtual memory?",
        "pipeline": "unknown"
    })
    assert response.status_code == 400


def test_feedback_valid(app_client):
    """POST /feedback with valid score should return 200."""
    response = app_client.post("/feedback", json={
        "question": "What is virtual memory?",
        "answer": "Virtual memory is...",
        "score": 5,
        "pipeline": "custom",
        "comment": "",
        "retrieved_chunks": []
    })
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_feedback_score_too_low(app_client):
    """POST /feedback with score < 1 should return 400."""
    response = app_client.post("/feedback", json={
        "question": "Q", "answer": "A",
        "score": 0, "pipeline": "custom",
        "comment": "", "retrieved_chunks": []
    })
    assert response.status_code == 400


def test_feedback_score_too_high(app_client):
    """POST /feedback with score > 5 should return 400."""
    response = app_client.post("/feedback", json={
        "question": "Q", "answer": "A",
        "score": 6, "pipeline": "custom",
        "comment": "", "retrieved_chunks": []
    })
    assert response.status_code == 400


def test_upload_invalid_file_type(app_client):
    """POST /upload with unsupported file type should return 400."""
    from unittest.mock import patch
    with patch("src.api.routes.document_exists", return_value=False):
        file = io.BytesIO(b"fake content")
        response = app_client.post(
            "/upload",
            files={"file": ("test.exe", file, "application/octet-stream")}
        )
    assert response.status_code == 400


def test_upload_duplicate_file(app_client):
    """POST /upload with already existing filename should return 400."""
    from unittest.mock import patch
    with patch("src.api.routes.document_exists", return_value=True):
        file = io.BytesIO(b"fake content")
        response = app_client.post(
            "/upload",
            files={"file": ("test.docx", file, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
    assert response.status_code == 400


def test_delete_nonexistent_document(app_client):
    """DELETE /documents/{filename} for missing file should return 404."""
    from unittest.mock import patch
    with patch("src.api.routes.document_exists", return_value=False):
        response = app_client.delete("/documents/missing.docx")
    assert response.status_code == 404