"""
indexer.py
--------
Responsible for building and researching a FAISS vector index.

Uses IndexFlatIP with L2 normalization,
which is equivalent to cosine similarity search - measuring semantic
direction between vectors rather than their magnitude.

- build: embed all chunks -> normalize -> add to FAISS index
- search: embed query -> normalize -> find top-k similar chunks
- save: save index to FAISS index and save chunks to disk
- load: load index from FAISS index and load chunks from disk
- delete_by_source: remove chunks from a specific file and rebuild index
"""
import faiss
from src.rag.embedder import Embedder
import os
import json


class FAISSIndexer:
    def __init__(self, dimension: int = 384, index_path: str = "indexes/faiss.index",
                 chunks_path: str = "indexes/chunks.json"):
        self.dimension = dimension
        self.chunks = []
        self.index = faiss.IndexFlatIP(dimension)
        self.index_path = index_path
        self.chunks_path = chunks_path

    def build(self, nodes, embedder: Embedder, source: str = "unknown"):
        new_chunks = [
            {"text":node.text, "source":source, "chunk_id":i}
            for i, node in enumerate(nodes)
        ]
        embeddings = embedder.embed([c["text"] for c in new_chunks])
        faiss.normalize_L2(embeddings)

        self.chunks.extend(new_chunks)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors from '{source}'")

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "w") as f:
            json.dump(self.chunks, f)
        print(f"Index saved to {self.index_path}")

    def load(self):
        if not os.path.exists(self.index_path) or \
                not os.path.exists(self.chunks_path):
            return False

        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "r") as f:
            self.chunks = json.load(f)
        print(f"Index loaded from {self.index_path} with {self.index.ntotal} vectors")
        return True

    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> list:
        query_embedding = embedder.embed([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        result = []
        for score, index in zip(scores[0], indices[0]):
            chunk = self.chunks[index]
            result.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score)
            })
        return result

    def delete_by_source(self, source:str, embedder: Embedder):
        before = len(self.chunks)
        remaining_chunks = [c for c in self.chunks if c["source"]!=source]
        removed_count = before - len(remaining_chunks)
        if removed_count==0:
            print(f"No chunks found for '{source}'")
            return 0

        self.chunks = []
        self.index = faiss.IndexFlatIP(self.dimension)
        if remaining_chunks:
            texts = [c["text"] for c in remaining_chunks]
            embeddings = embedder.embed(texts)
            faiss.normalize_L2(embeddings)
            self.chunks = remaining_chunks
            self.index.add(embeddings)
        self.save()
        print(f"Deleted {removed_count} chunks from '{source}'. Index rebuilt with {self.index.ntotal} vectors.")
        return removed_count