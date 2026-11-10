"""Classement des chunks candidats à l'aide d'un CrossEncoder."""

from sentence_transformers import CrossEncoder
from database.models import Chunk


class Reranker:
    """Réordonne les chunks selon leur pertinence sémantique pour une question."""

    def __init__(self, model_name: str):
        """Charge le modèle CrossEncoder à utiliser pour le reclassement.

        Args:
            model_name: Identifiant local ou distant du modèle CrossEncoder.

        Returns:
            None: Le constructeur conserve le modèle chargé dans l'instance.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Classe les chunks par score CrossEncoder et conserve les meilleurs.

        Args:
            question: Question de l'utilisateur utilisée comme requête de classement.
            chunks: Chunks candidats à évaluer.
            top_k: Nombre maximal de chunks à retourner.

        Returns:
            list[Chunk]: Chunks classés par score décroissant, tronqués à ``top_k``;
                une liste vide si aucun candidat n'est fourni.
        """
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