from sentence_transformers import CrossEncoder
from database.models import Chunk


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [
            (question, chunk.embedding_text)
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        results = list(zip(chunks, scores))

        results.sort(
            key=lambda result: result[1],
            reverse=True,
        )

        return [c[0] for c in results[:top_k]]