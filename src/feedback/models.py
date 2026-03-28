"""
models.py
---------
Database table definitions for the feedback system.
Uses SQLAlchemy ORM to define the feedback table structure.

Each row represents one user interaction:
- the question asked
- the answer generated
- the user's score (1-5)
- which pipeline generated the answer
- the retrieved chunks used to generate the answer
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    pipeline = Column(String(50), nullable=False)
    comment = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)