import json
import pytest
from unittest.mock import MagicMock
from src.training.train_reranker import prepare_training_data

def make_feedback(question, score, chunks):
    entry = MagicMock()
    entry.question = question
    entry.score = score
    entry.retrieved_chunks = json.dumps(chunks)
    return entry


def test_positive_examples_from_high_scores():
    """Scores >= 4 should produce label 1.0 (positive examples)."""
    chunks = [{"text": "relevant chunk"}]
    feedback = [make_feedback("What is memory?", 5, chunks)]

    samples = prepare_training_data(feedback)

    assert len(samples) == 1
    assert samples[0].label == 1.0


def test_negative_examples_from_low_scores():
    """Scores <= 2 should produce label 0.0 (negative examples)."""
    chunks = [{"text": "irrelevant chunk"}]
    feedback = [make_feedback("What is memory?", 1, chunks)]

    samples = prepare_training_data(feedback)

    assert len(samples) == 1
    assert samples[0].label == 0.0


def test_neutral_scores_are_ignored():
    """Score == 3 should be ignored and produce no training samples."""
    chunks = [{"text": "some chunk"}]
    feedback = [make_feedback("What is memory?", 3, chunks)]

    samples = prepare_training_data(feedback)

    assert len(samples) == 0


def test_empty_chunks_are_skipped():
    """Feedback entries with no chunks should be skipped."""
    feedback = [make_feedback("What is memory?", 5, [])]

    samples = prepare_training_data(feedback)

    assert len(samples) == 0


def test_empty_chunk_text_is_skipped():
    """Chunks with empty text should be skipped."""
    chunks = [{"text": ""}]
    feedback = [make_feedback("What is memory?", 5, chunks)]

    samples = prepare_training_data(feedback)

    assert len(samples) == 0


def test_multiple_chunks_produce_multiple_samples():
    """Each chunk in a feedback entry should become a separate training sample."""
    chunks = [{"text": "chunk one"}, {"text": "chunk two"}, {"text": "chunk three"}]
    feedback = [make_feedback("What is memory?", 4, chunks)]

    samples = prepare_training_data(feedback)

    assert len(samples) == 3


def test_texts_are_question_chunk_pairs():
    """Each InputExample should contain [question, chunk_text]."""
    chunks = [{"text": "virtual memory explanation"}]
    feedback = [make_feedback("What is virtual memory?", 5, chunks)]

    samples = prepare_training_data(feedback)

    assert samples[0].texts[0] == "What is virtual memory?"
    assert samples[0].texts[1] == "virtual memory explanation"


def test_mixed_scores():
    """Mixed scores should produce correct labels for each entry."""
    feedback = [
        make_feedback("Q1", 5, [{"text": "good chunk"}]),
        make_feedback("Q2", 3, [{"text": "neutral chunk"}]),
        make_feedback("Q3", 1, [{"text": "bad chunk"}]),
    ]

    samples = prepare_training_data(feedback)

    assert len(samples) == 2
    assert samples[0].label == 1.0
    assert samples[1].label == 0.0