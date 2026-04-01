"""
collector.py
------------
Handles all database operations for the feedback system.

Functions:
- save_feedback:       store a new feedback entry in the database
- get_all_feedback:    retrieve all feedback entries
- get_low_scores:      retrieve feedback with score <= threshold
                       used to identify poor retrievals for retraining
- get_feedback_count:  return total number of feedback entries
                       used to trigger retraining when count reaches threshold
"""
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.feedback.models import Base, Feedback
from dotenv import load_dotenv
from sqlalchemy import func

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_feedback_stats() -> dict:
    session = Session()
    try:
        total = session.query(Feedback).count()
        avg_score = session.query(func.avg(Feedback.score)).scalar()

        last_feedback = (
            session.query(Feedback)
            .order_by(Feedback.created_at.desc())
            .first()
        )

        return {
            "total": total,
            "avg_score": round(float(avg_score), 2) if avg_score else 0.0,
            "last_feedback_at": last_feedback.created_at.isoformat() if last_feedback else None,
        }
    finally:
        session.close()

def save_feedback(question: str,
                  answer: str,
                  score: float,
                  pipeline: str,
                  comment: str = "",
                  retrieved_chunks: list = None) -> Feedback:
    session = Session()
    try:
        feedback = Feedback(
            question=question,
            answer=answer,
            score=score,
            pipeline=pipeline,
            comment=comment,
            retrieved_chunks=json.dumps(retrieved_chunks or [])
        )
        session.add(feedback)
        session.commit()
        session.refresh(feedback)
        print(f"Feedback saved: question='{question[:50]}' score={score}")
        return feedback
    finally:
        session.close()

def get_all_feedback() -> list:
    session = Session()
    try:
        return session.query(Feedback).all()
    finally:
        session.close()

def get_low_scores(threshold: int = 3) -> list:
    session = Session()
    try:
        return session.query(Feedback).filter(Feedback.score <= threshold).all()
    finally:
        session.close()

def get_feedback_count() -> int:
    session = Session()
    try:
        return session.query(Feedback).count()
    finally:
        session.close()