"""
pipeline.py
--------
The main custom RAG pipeline that connects all components end to end.

Flow:
    1. Load and chunk the document(loader.py)
    2. Build FAISS index from chunks(embedder.py and indexer.py)
    3. On each query:
        - Retrieve top-k relevant chunks via FAISS
        - Build a prompt with retrieved thunks as context
        - send prompt to Llama3.1:8b via Ollama
        - return the generated answer and retrieved chunks

"""
from ollama import Client
from src.rag.embedder import Embedder
from src.rag.indexer import FAISSIndexer
from src.rag.loader import load_documents, split_documents
import os

class RAGPipeline:
    def __init__(self, file_path: str, top_k: int = 3):
        self.embedder = Embedder()
        self.top_k = top_k
        self.indexer = FAISSIndexer()
        self.client = Client()
        self.PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based only on the context provided below.
                                If the answer is not found in the context, say "I don't know based on the provided document."

                                Context:
                                {context}

                                Question: {query}

                                Answer:"""
        # load
        if not self.indexer.load():
            print("No saved index found, building from scratch...")
            docs = load_documents(file_path)
            nodes = split_documents(docs)
            source = os.path.basename(file_path)
            # embedder converts chunks and query into vectors
            # indexer stores vectors and retrieves efficiently
            self.indexer.build(nodes, self.embedder,source=source)
            self.indexer.save()
        else:
            print("Index loaded from disk, skipping rebuild")

    def _build_prompt(self, query: str, chucks: list) -> str:
        context = "\n\n".join(
            [f"[Chunk {i + 1}]:\n{c['text']}" for i, c in enumerate(chucks)]
        )
        prompt = self.PROMPT_TEMPLATE.format(context=context, query=query)
        return prompt

    def query(self, question: str) -> dict:
        chunks = self.indexer.search(question, self.embedder, self.top_k)

        prompt = self._build_prompt(question, chunks)

        response = self.client.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            # With `temperature=0.1`, the model will almost always pick the highest probability path, giving much more consistent answers while still sounding natural.
            # temperature = 0.0  → always picks the highest probability word
            # temperature = 1.0  → full randomness (default)
            options = {
                "temperature": 0.1
            }
        )
        return {
            "question": question,
            "answer": response["message"]["content"],
            "retrieved_chunks": chunks
        }
