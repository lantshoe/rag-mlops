
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

        # load
        docs = load_documents(file_path)
        nodes = split_documents(docs)
        self.indexer.build(nodes, self.embedder)

    def _build_prompt(self, query:str, chucks:list) -> str:
        context = "\n\n".join(
            [f"[Chunk {i+1}]:\n{c['text']}" for i, c in enumerate(chucks)]
        )
        return f"""You are a helpful assistant. Answer the question based only on the context provided below.
                If the answer is not found in the context, say "I don't know based on the provided document."
                Context:
                {context}
                Question: {query}
                Answer:"""

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

