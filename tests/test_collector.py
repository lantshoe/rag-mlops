import pytest
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
from src.feedback.models import Base, Feedback, Document


@pytest.fixture
def db_session():
    """In-memory SQLite database for testing — no PostgreSQL needed."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def patched_session(db_session):
    """Patch the collector's Session to use our in-memory DB."""
    with patch("src.feedback.collector.Session", return_value=db_session):
        yield db_session


def test_save_feedback(patched_session):
    """save_feedback() should persist a feedback entry to the database."""
    from src.feedback.collector import save_feedback

    save_feedback(
        question="What is virtual memory?",
        answer="Virtual memory is...",
        score=5.0,
        pipeline="custom",
        comment="great answer",
        retrieved_chunks=[{"text": "chunk", "source": "test.docx", "chunk_id": 0}]
    )

    result = patched_session.query(Feedback).first()
    assert result is not None
    assert result.question == "What is virtual memory?"
    assert result.score == 5.0
    assert result.pipeline == "custom"


def test_save_feedback_serializes_chunks(patched_session):
    """save_feedback() should store retrieved_chunks as a JSON string."""
    from src.feedback.collector import save_feedback

    chunks = [{"text": "chunk text", "source": "test.docx", "chunk_id": 0}]
    save_feedback("Q?", "A.", 4.0, "custom", "", chunks)

    result = patched_session.query(Feedback).first()
    parsed = json.loads(result.retrieved_chunks)
    assert parsed[0]["text"] == "chunk text"


def test_get_feedback_count(patched_session):
    """get_feedback_count() should return the correct total count."""
    from src.feedback.collector import get_feedback_count

    for i in range(3):
        patched_session.add(Feedback(
            question=f"Q{i}", answer="A", score=4.0,
            pipeline="custom", comment="", retrieved_chunks="[]"
        ))
    patched_session.commit()

    count = get_feedback_count()
    assert count == 3


def test_get_all_feedback(patched_session):
    """get_all_feedback() should return all entries."""
    from src.feedback.collector import get_all_feedback

    patched_session.add(Feedback(
        question="Q1", answer="A1", score=5.0,
        pipeline="custom", comment="", retrieved_chunks="[]"
    ))
    patched_session.commit()

    results = get_all_feedback()
    assert len(results) == 1
    assert results[0].question == "Q1"


def test_save_document(patched_session):
    """save_document() should persist document metadata."""
    from src.feedback.collector import save_document

    save_document("test.docx", "data/test.docx", 42)

    result = patched_session.query(Document).first()
    assert result is not None
    assert result.filename == "test.docx"
    assert result.chunk_count == 42


def test_document_exists(patched_session):
    """document_exists() should return True if the file is in the DB."""
    from src.feedback.collector import document_exists

    patched_session.add(Document(
        filename="existing.docx",
        file_path="data/existing.docx",
        chunk_count=10
    ))
    patched_session.commit()

    assert document_exists("existing.docx") is True
    assert document_exists("missing.docx") is False


def test_delete_document(patched_session):
    """delete_document() should remove the entry from the database."""
    from src.feedback.collector import delete_document

    patched_session.add(Document(
        filename="to_delete.docx",
        file_path="data/to_delete.docx",
        chunk_count=5
    ))
    patched_session.commit()

    result = delete_document("to_delete.docx")
    assert result is True
    assert patched_session.query(Document).count() == 0


def test_get_feedback_stats(patched_session):
    """get_feedback_stats() should return correct total and avg_score."""
    from src.feedback.collector import get_feedback_stats

    for score in [5.0, 4.0, 3.0]:
        patched_session.add(Feedback(
            question="Q", answer="A", score=score,
            pipeline="custom", comment="", retrieved_chunks="[]"
        ))
    patched_session.commit()

    stats = get_feedback_stats()
    assert stats["total"] == 3
    assert stats["avg_score"] == 4.0