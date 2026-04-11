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
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from src.rag.loader import load_documents, split_documents
from llama_index.core.settings import Settings
import os
from src.rag.prompts import RAG_PROMPT_TEMPLATE
from llama_index.core import StorageContext, load_index_from_storage

LLAMA_INDEX_PATH = "indexes/llama_index"
class LlamaIndexPipeline:
    def __init__(self, data_dir: str, top_k: int = 3):
        Settings.llm = Ollama(
            model="llama3.1:8b",
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            request_timeout=300,
            temperature=0.1,
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="all-MiniLM-L6-v2"
        )
        self.top_k = top_k
        self.data_dir = data_dir

        if os.path.exists(LLAMA_INDEX_PATH):
            print("LlamaIndex: loading index from disk...")
            storage_context = StorageContext.from_defaults(persist_dir=LLAMA_INDEX_PATH)
            self.index = load_index_from_storage(storage_context)
        else:
            print("LlamaIndex: building index from scratch...")
            self.build_from_disk()

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            text_qa_template=PromptTemplate(RAG_PROMPT_TEMPLATE),
        )

    def build_from_disk(self):
        all_nodes = self._load_all_nodes(self.data_dir)
        self.index = VectorStoreIndex(all_nodes)
        self.index.storage_context.persist(persist_dir=LLAMA_INDEX_PATH)

    def _load_all_nodes(self, data_dir: str) -> list:
        """Load and chunk all supported files from the data folder."""
        supported = {".docx", ".pdf", ".txt"}
        all_nodes = []

        if not os.path.exists(data_dir):
            print(f"Data folder '{data_dir}' not found, starting with empty index.")
            return all_nodes

        files = [
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in supported
        ]

        if not files:
            print(f"No supported files found in '{data_dir}'.")
            return all_nodes

        for filename in files:
            file_path = os.path.join(data_dir, filename)
            print(f"LlamaIndex: indexing '{filename}'...")
            docs = load_documents(file_path)
            nodes = split_documents(docs)
            all_nodes.extend(nodes)

        print(f"LlamaIndex: indexed {len(all_nodes)} chunks from {len(files)} file(s).")
        return all_nodes

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
