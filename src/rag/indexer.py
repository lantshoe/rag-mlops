import faiss

from src.rag.embedder import Embedder


class FAISSIndexer:
    def __init__(self, dimension:int = 384):
        self.dimension = dimension
        self.chunks = []
        self.index = faiss.IndexFlatIP(dimension)

    def build(self, nodes, embedder:Embedder):
        self.chunks = [node.text for node in nodes]
        embeddings = embedder.embed(self.chunks)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index build with {self.index.ntotal} vectors")

    def search(self, query:str, embedder: Embedder, top_k:int = 5) -> list:
        query_embedding = embedder.embed([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        result = []
        for score, index in zip(scores[0], indices[0]):
            result.append({
                "text":self.chunks[index],
                "score":float(score)
            })
        return result
