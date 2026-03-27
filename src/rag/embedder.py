"""
embedder.py
--------
Responsible for converting text into vector embeddings.

Use the sentence-transformers (all-MiniLM-L6-v2) to encode text into 384-dimensional vectors
that capture semantic meaning.
These vectors are used by the FAISS indexer for similarity search.

"""
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)
