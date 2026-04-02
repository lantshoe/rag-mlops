import pytest
import numpy as np
import json
import os
from unittest.mock import MagicMock, patch


@pytest.fixture
def indexer(tmp_path):
    """Create a FAISSIndexer with temp paths so nothing touches disk."""
    from src.rag.indexer import FAISSIndexer
    return FAISSIndexer(
        dimension=384,
        index_path=str(tmp_path / "faiss.index"),
        chunks_path=str(tmp_path / "chunks.json"),
    )

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.random.rand(3, 384).astype("float32")
    return embedder


@pytest.fixture
def mock_nodes():
    nodes = []
    for i in range(3):
        node = MagicMock()
        node.text = f"chunk text {i}"
        nodes.append(node)
    return nodes


def test_build_adds_chunks(indexer, mock_nodes, mock_embedder):
    """build() should add chunks to the index."""
    indexer.build(mock_nodes, mock_embedder, source="test.docx")

    assert len(indexer.chunks) == 3
    assert indexer.index.ntotal == 3

def test_build_sets_correct_source(indexer, mock_nodes, mock_embedder):
    """build() should tag each chunk with the correct source."""
    indexer.build(mock_nodes, mock_embedder, source="test.docx")

    for chunk in indexer.chunks:
        assert chunk["source"] == "test.docx"

def test_save_and_load(indexer, mock_nodes, mock_embedder):
    """save() then load() should restore chunks and index correctly."""
    indexer.build(mock_nodes, mock_embedder, source="test.docx")
    indexer.save()

    from src.rag.indexer import FAISSIndexer
    new_indexer = FAISSIndexer(
        dimension=384,
        index_path=indexer.index_path,
        chunks_path=indexer.chunks_path,
    )
    loaded = new_indexer.load()

    assert loaded is True
    assert len(new_indexer.chunks) == 3
    assert new_indexer.index.ntotal == 3

def test_search_returns_top_k(indexer, mock_nodes, mock_embedder):
    """search() should return exactly top_k results."""
    indexer.build(mock_nodes, mock_embedder, source="test.docx")

    mock_embedder.embed.return_value = np.random.rand(1, 384).astype("float32")
    results = indexer.search("test query", mock_embedder, top_k=2)

    assert len(results) == 2


def test_search_result_has_required_fields(indexer, mock_nodes, mock_embedder):
    """search() results should have text, source, chunk_id, and score."""
    indexer.build(mock_nodes, mock_embedder, source="test.docx")
    mock_embedder.embed.return_value = np.random.rand(1, 384).astype("float32")

    results = indexer.search("test query", mock_embedder, top_k=1)

    assert "text" in results[0]
    assert "source" in results[0]
    assert "chunk_id" in results[0]
    assert "score" in results[0]


def test_delete_by_source_removes_chunks(indexer, mock_nodes, mock_embedder):
    """delete_by_source() should remove all chunks from that source."""
    indexer.build(mock_nodes, mock_embedder, source="remove_me.docx")

    extra_node = MagicMock()
    extra_node.text = "keep this"
    mock_embedder.embed.return_value = np.random.rand(1, 384).astype("float32")
    indexer.build([extra_node], mock_embedder, source="keep_me.docx")

    mock_embedder.embed.return_value = np.random.rand(1, 384).astype("float32")
    removed = indexer.delete_by_source("remove_me.docx", mock_embedder)

    assert removed == 3
    assert all(c["source"] != "remove_me.docx" for c in indexer.chunks)


def test_delete_nonexistent_source(indexer, mock_embedder):
    """delete_by_source() should return 0 if source doesn't exist."""
    removed = indexer.delete_by_source("nonexistent.docx", mock_embedder)
    assert removed == 0


def test_load_returns_false_when_no_files(indexer):
    """load() should return False when index files don't exist."""
    result = indexer.load()
    assert result is False