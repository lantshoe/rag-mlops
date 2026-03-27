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


class RAGPipeline:
    def __init__(self, file_path:str, top_k:int = 3):
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
        docs = load_documents(file_path)
        nodes = split_documents(docs)
        # embedder converts chunks and query into vectors
        # indexer stores vectors and retrieves efficiently
        self.indexer.build(nodes, self.embedder)

    def _build_prompt(self, query:str, chucks:list) -> str:
        context = "\n\n".join(
            [f"[Chunk {i+1}]:\n{c['text']}" for i, c in enumerate(chucks)]
        )
        prompt = self.PROMPT_TEMPLATE.format(context=context, query=query)
        return prompt
    
    def query(self, question:str) -> dict:
        chunks = self.indexer.search(question, self.embedder, self.top_k)

        prompt = self._build_prompt(question, chunks)

        response = self.client.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "question": question,
            "answer": response["message"]["content"],
            "retrieved_chunks": chunks
        }

