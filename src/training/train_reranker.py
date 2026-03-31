"""
train_reranker.py
-----------------
Trains a CrossEncoder re-ranker model using feedback data from PostgreSQL.

A CrossEncoder takes a (question, chunk) pair as input and outputs
a relevance score. Unlike the embedder which encodes question and chunk
separately, the CrossEncoder reads them together — giving much more
accurate relevance scoring.

Training data comes from user feedback:
- High score (4-5) → the retrieved chunks were relevant  → positive examples
- Low score  (1-2) → the retrieved chunks were not relevant → negative examples

The trained model is saved to disk and logged to MLflow for version tracking.

User gives score 4 to a good answer
     ↓
Saved to PostgreSQL with retrieved chunks
     ↓
train_reranker() reads this feedback
     ↓
Teaches CrossEncoder: "these chunks were relevant for this question"
     ↓
Next time user asks similar question
     ↓
CrossEncoder ranks the relevant chunks higher
     ↓
LLM gets better context → generates better answer
     ↓
User gives score 5 → even more training data
"""
import json
import os

import mlflow
from sentence_transformers import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

from src.feedback.collector import get_all_feedback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# path to save the trained re-ranker model
RERANKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "reranker")

MLFLOW_DB_PATH = "mlflow.db"

# base model to fine-tune from
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"




def prepare_training_data(feedback_list) -> list:
    """
    Convert feedback entries into training pairs for the CrossEncoder.

    For each feedback entry:
    - score >= 4 → positive example (question, chunk) pair
    - score <= 2 → negative example (question, chunk) pair
    - score == 3 → ignored (neutral, not useful for training)

    Returns a list of InputExample objects.
    """
    training_samples = []
    for entry in feedback_list:
        question = entry.question
        score = entry.score
        chunks = json.loads(entry.retrieved_chunks)

        if not chunks:
            continue
        #  good, label as positive
        if score >= 4:
            label = 1.0
        #   bad, label as negative
        elif score <= 2:
            label = 0.0
        else:
            continue

        for chunk in chunks:
            text = chunk["text"]
            if not text:
                continue
            # if a user gave score 4 to a query, we tell the model that
            # when you see the question and chunk together, output score close to 1
            training_samples.append(
                # question and chunk together, so that the self attention encoder can read question and answer together to have a better understand
                InputExample(texts=[question, text], label=label)
            )
    return training_samples

def train_reranker(min_samples: int = 10):
    os.makedirs(RERANKER_MODEL_PATH, exist_ok=True)

    feedback_list = get_all_feedback()
    if not feedback_list:
        print("No feedback entries found.")
        return

    training_samples = prepare_training_data(feedback_list)
    if len(training_samples) < min_samples:
        print(f"Not enough training samples: {len(training_samples)} < {min_samples}. Skipping.")
        return

    print(f"Training samples: {len(training_samples)}")
    print(f"Reranking model: {RERANKER_MODEL_PATH}")
    # We start from `cross-encoder/ms-marco-MiniLM-L-6-v2`
    # — a pre-trained CrossEncoder that already understands relevance scoring.
    # We fine-tune it on our specific domain (OS concepts) using feedback data.
    model = CrossEncoder(BASE_MODEL, num_labels=1)
    train_dataloader = DataLoader(training_samples, shuffle=True, batch_size=8)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    experiment_name = "reranker_training"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="reranker_training"):
        mlflow.log_param("base_model", BASE_MODEL)
        mlflow.log_param("num_samples", len(training_samples))
        mlflow.log_param("num_epochs", 3)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("model_saved_to", RERANKER_MODEL_PATH)

        print("Training reranker...")
        model.fit(
            train_dataloader=train_dataloader,
            epochs=3,
            warmup_steps=10
        )

        model.save(RERANKER_MODEL_PATH)
        mlflow.log_artifacts(RERANKER_MODEL_PATH, artifact_path="reranker_model")

    print(f"Re-ranker trained and saved to {RERANKER_MODEL_PATH}")

if __name__ == '__main__':
    train_reranker(min_samples=1)
