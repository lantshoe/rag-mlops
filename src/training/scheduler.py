"""
scheduler.py
------------
Handles automated retraining of the CrossEncoder reranker.

Two triggers:
    1. Threshold-based: retrain when new feedback count reaches FEEDBACK_THRESHOLD
    2. Schedule-based: retrain every RETRAIN_INTERVAL_HOURS hours

Both run without blocking the main FastAPI server.
"""

import threading
import time
from src.feedback.collector import get_feedback_count
from src.training.train_reranker import train_reranker

FEEDBACK_THRESHOLD = 50
RETRAIN_INTERVAL_HOURS = 24
_last_trained_feedback_count = 0
_lock = threading.Lock()
_retrain_lock = threading.Lock()

def _do_retrain(reason:str):
    if not _retrain_lock.acquire(blocking=False):
        print(f"Retrain already running, skipping trigger: {reason}")
        return
    try:
        global _last_trained_feedback_count
        print(f"Retraining triggered: {reason}")
        try:
            train_reranker(min_samples=10)
            with _lock:
                _last_trained_feedback_count = get_feedback_count()
            print("Retraining completed successfully.")
        except Exception as e:
            print(f"Retraining failed with exception: {e}")
    finally:
        _retrain_lock.release()

def check_threshold_trigger():
    global _last_trained_feedback_count
    current_feedback_count = get_feedback_count()
    with _lock:
        new_feedback_count = current_feedback_count - _last_trained_feedback_count
    if new_feedback_count >= FEEDBACK_THRESHOLD:
        thread = threading.Thread(
            target=_do_retrain,
            args=(f"feedback count reached threshold of {FEEDBACK_THRESHOLD}. ",),
            daemon=True
        )
        thread.start()
def get_last_trained_count() -> int:
    with _lock:
        return _last_trained_feedback_count

def start_schedule_trigger():
    def _loop():
        while True:
            time.sleep(RETRAIN_INTERVAL_HOURS * 3600)
            _do_retrain(f"scheduled every {RETRAIN_INTERVAL_HOURS} hours.")

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    print(f"Scheduled retraining started — every {RETRAIN_INTERVAL_HOURS} hours.")
