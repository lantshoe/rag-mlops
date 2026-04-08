"""
pipeline.py
--------
The main custom RAG pipeline that connects all components end to end.

Flow:
    1. Load and chunk the document(loader.py)
    2. Build FAISS index from chunks(embedder.py and indexer.py)
    3. On each query:
        - Retrieve top-k relevant chunks via FAISS
        - Re-rank chunks using CrossEncoder(reranker)
        - Build a prompt with retrieved thunks as context
        - send prompt to Llama3.1:8b via Ollama
        - return the generated answer and retrieved chunks

"""
from collections import defaultdict

from ollama import Client
from sentence_transformers import CrossEncoder
from src.rag.prompts import RAG_PROMPT_TEMPLATE
from src.rag.embedder import Embedder
from src.rag.indexer import FAISSIndexer
from src.rag.loader import load_documents, split_documents
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RERANKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "reranker")

class RAGPipeline:
    def __init__(self, data_dir: str, top_k: int = 10, top_n = 3):
        self.embedder = Embedder()
        # retrieve from FAISS
        self.top_k = top_k
        # keep only the best after rerank
        self.top_n = top_n
        self.indexer = FAISSIndexer()
        self.client = Client()
        # load the CrossEncoder reranker
        if os.path.exists(RERANKER_MODEL_PATH):
            print(f"Loading model from {RERANKER_MODEL_PATH}")
            self.reranker = CrossEncoder(model_name_or_path=RERANKER_MODEL_PATH, num_labels=1)
        else:
            print("Fine-tuned reranker not found, using base model...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

        self.PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE
        # load
        if not self.indexer.load():
            print("No saved index found, building from scratch...")
            self._build_from_folder(data_dir)
        else:
            print("Index loaded from disk, skipping rebuild")

    def _build_from_folder(self, data_dir:str):
        supported = {".docx",".pdf",".txt"}
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found, skipping build")
            return
        files = [
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in supported
        ]
        if not files:
            print(f"No supported files found in '{data_dir}'.")
            return

        for filename in files:
            file_path = os.path.join(data_dir, filename)
            docs = load_documents(file_path)
            nodes = split_documents(docs)
            self.indexer.build(nodes, self.embedder,source=filename)
        self.indexer.save()
        print(f"Built index from {len(files)} file(s) in '{data_dir}'.")


    def _rerank(self, question:str, chunks:list) -> list:
        """
        re-rank chunks using CrossEncoder
        return
        ------
        [{"text":"","chunk_id":"", "score":0, "reranker_score":0}]
        """
        paris = [[question, chunk["text"]] for chunk in chunks]

        scores = self.reranker.predict(paris)
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:self.top_n]

    def _build_prompt(self, query: str, chunks: list) -> str:
        context = "\n\n".join(
            [f"[Chunk {i + 1}]:\n{c['text']}" for i, c in enumerate(chunks)]
        )
        prompt = self.PROMPT_TEMPLATE.format(context_str=context, query_str=query)
        return prompt

    def _expand_chunks(self, reranked_chunks, chunks, window) -> list:
        source_chunks = defaultdict(list)
        for chunk in chunks:
            source_chunks[chunk["source"]].append(chunk)

        # sort chunks in every source in order to apply window
        for source in source_chunks:
            source_chunks[source] = sorted(source_chunks[source], key=lambda x: x["chunk_id"])

        selected_ids = {(c["source"], c["chunk_id"]) for c in reranked_chunks}

        expanded = list(reranked_chunks)

        for chunk in reranked_chunks:
            source = chunk["source"]
            cid = chunk["chunk_id"]
            neighbors = source_chunks[source]
            for neighbor in neighbors:
                nid = neighbor["chunk_id"]
                key = (source, nid)
                if abs(nid - cid) <= window and key not in selected_ids:
                    selected_ids.add(key)
                    expanded.append({**neighbor, "reranker_score": None, "is_context": True})

        expanded.sort(key=lambda x: (x["source"], x["chunk_id"]))
        return expanded

    def query(self, question: str) -> dict:
        # retrieve top_k from FAISS
        chunks = self.indexer.search(question, self.embedder, self.top_k)

        # rerank using CrossEncoder, keep top_n
        reranked_chunks = self._rerank(question, chunks)
        
        expand_chunks = self._expand_chunks(
            reranked_chunks,
            self.indexer.chunks,
            window  = 1
        )

        # build prompt with reranked chunks
        prompt = self._build_prompt(question, expand_chunks)

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
            # return reranked_chunks for feedback not all expand_chunks
            "retrieved_chunks": reranked_chunks
        }




