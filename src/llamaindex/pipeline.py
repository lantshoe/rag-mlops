"""
pipeline.py
--------
LlamaIndex-based RAG pipeline, parallel to the custom FAISS pipeline.

Uses LlamaIndex's built-in components for document indexing and retrieval,
with Llama3.1:8b via Ollama for answer generation.

This pipeline is compared against the custom FAISS pipeline in notebooks/
to evaluate retrieval quality and answer quality.


Custom Pipeline:                    LlamaIndex Pipeline:

loader.py                           loader.py (shared)
    ↓                                   ↓
embedder.py (manual)                Settings.embed_model (auto)
    ↓                                   ↓
indexer.py (FAISS manual)           VectorStoreIndex (auto)
    ↓                                   ↓
_build_prompt() (manual)            as_query_engine() (auto)
    ↓                                   ↓
client.chat() (manual)              query_engine.query() (auto)
"""
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from src.rag.loader import load_documents, split_documents
from llama_index.core.settings import Settings


class LlamaIndexPipeline:
    def __init__(self, file_path: str, top_k: int = 3):
        Settings.llm = Ollama(model="llama3.1:8b", request_timeout=300)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="all-MiniLM-L6-v2"
        )
        self.top_k = top_k
        docs = load_documents(file_path)
        nodes = split_documents(docs)
        self.index = VectorStoreIndex(nodes)
        self.query_engine = self.index.as_query_engine(similarity_topk=self.top_k)

    def query(self, question: str):
        response = self.query_engine.query(question)
        retrieved_chunks = [
            {
                "text": node.text,
                "score": float(node.score) if node.score else 0.0
            }
            for node in response.source_nodes
        ]
        return {
            "question": question,
            "answer": str(response),
            "retrieved_chunks": retrieved_chunks
        }
