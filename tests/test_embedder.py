import numpy as np
import pytest
from unittest.mock import patch, MagicMock

def test_embed_returns_numpy_array():
    with patch("src.rag.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype("float32")
        mock_st.return_value = mock_model

        from src.rag.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)

def test_embed_single_text():
    """Embedder.embed() should handle a single text input."""
    with patch("src.rag.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype("float32")
        mock_st.return_value = mock_model

        from src.rag.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed(["single text"])

        assert result.shape[0] == 1


def test_embed_called_with_correct_input():
    """Embedder.embed() should pass texts directly to the model."""
    with patch("src.rag.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype("float32")
        mock_st.return_value = mock_model

        from src.rag.embedder import Embedder
        embedder = Embedder()
        texts = ["test input"]
        embedder.embed(texts)

        mock_model.encode.assert_called_once_with(texts, show_progress_bar=True)