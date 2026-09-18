"""Service RAG : embedding de la question, recherche vectorielle + BM25, fusion RRF et génération via Ollama."""

import asyncio
import os
import json
import requests

from database.chunk_repository import ChunkRepository
from database.database import get_session
from database.models import Chunk

from logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

CHUNK_LIMIT_RRF = int(os.getenv("CHUNK_LIMIT_RRF", "10"))
CHUNK_LIMIT_CROSS_ENCODER = int(os.getenv("CHUNK_LIMIT_CROSS_ENCODER", "10"))


SYSTEM_PROMPT = """\
Tu es un assistant technique expert pour la marque Electrodomus, spécialisé dans le dépannage,
l'installation et l'entretien des appareils électroménagers (fours, lave-linge, lave-vaisselle).

Règles absolues :
- Réponds UNIQUEMENT à partir des extraits de documentation fournis dans le CONTEXTE.
- Si la réponse ne figure pas dans le contexte, réponds exactement :
  "Je n'ai pas trouvé cette information dans la documentation Electrodomus."
- Ne complète jamais avec tes connaissances générales.
- Cite TOUJOURS la source (section / page) entre parenthèses après chaque information.
- Si plusieurs extraits se contredisent, signale la contradiction.
- Réponds en français, de façon claire, structurée et concise.
"""

class ChatService:
    """Orchestre le pipeline RAG : recherche de contexte pertinent puis génération de réponse."""

    def __init__(self, embed_model, reranker):
        self.embed_model = embed_model
        self.reranker = reranker

    async def _embed(self, question: str):
        """Calcule l'embedding normalisé de la question, dans un thread séparé."""
        return await asyncio.to_thread(
            self.embed_model.encode, question, normalize_embeddings=True
        )

    @staticmethod
    def _get_nearest_vectorized_chunks(
        embedding, model_id: int | None = None, error_code: str | None = None
    ) -> list[Chunk]:
        """Recherche les chunks les plus proches par similarité vectorielle (cosinus)."""
        with get_session() as session:
            return ChunkRepository(session).get_nearest(embedding.tolist(), model_id=model_id, error_code=error_code)

    @staticmethod
    def _get_nearest_bm25_chunks(
        question: str, model_id: int | None = None, error_code: str | None = None
    ) -> list[Chunk]:
        """Recherche les chunks les plus pertinents par correspondance lexicale (BM25)."""
        with get_session() as session:
            return ChunkRepository(session).search_bm25(question, model_id=model_id, error_code=error_code)

    @staticmethod
    def _chat(messages: list[dict], stream: bool = True) -> str:
        """Envoie les messages à Ollama et retourne la réponse complète (streaming ou non)."""
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": stream,
        }

        logger.debug("Envoi de %d message(s) à Ollama (modèle=%s, stream=%s)", len(messages), MODEL, stream)
        response = requests.post(OLLAMA_URL, json=payload, stream=stream)
        if not response.ok:
            logger.error("Erreur Ollama %s : %s", response.status_code, response.text)
            raise RuntimeError(f"Ollama {response.status_code}: {response.text}")

        full_response = ""

        if stream:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full_response += token
                if chunk.get("done"):
                    break
            print()
        else:
            data = response.json()
            full_response = data["message"]["content"]

        return full_response


    @staticmethod
    def _build_rag_message(user_input: str, chunks: list) -> str:
        """Construit le message utilisateur contenant le contexte des chunks et la question."""
        if not chunks:
            context = "Aucun extrait pertinent trouvé."
        else:
            context = "\n\n---\n\n".join(
                f"[Source : chunk {c.id}, document {c.document_id or '?'}, {c.section or 'inconnue'}, page {c.page or '?'}]\n{c.content}"
                for c in chunks
            )
        return f"CONTEXTE :\n{context}\n\nQUESTION : {user_input}"

    @staticmethod
    def _build_messages(question: str, history: list, rag_content: str) -> list[dict]:
        """Assemble les messages Ollama : prompt système, historique puis message RAG."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": rag_content})
        return messages

    @staticmethod
    def _build_sources(chunks: list) -> list[dict]:
        """Extrait les métadonnées de source (section, page, document) des chunks retenus."""
        return [
            {"section": c.section, "page": c.page, "document_id": c.document_id}
            for c in chunks
        ]

    @staticmethod
    def _reciprocal_rank_fusion(vectorial_list: list[Chunk], bm_list: list[Chunk], k: int = 60) -> list[Chunk]:
        """Effectue la fusion par rang réciproque (RRF) de deux listes de résultats, dédoublonnée par id de chunk.

        Args:
            vectorial_list (list[Chunk]): Liste des chunks triés par similarité vectorielle.
            bm_list (list[Chunk]): Liste des chunks triés par score BM25.
            k (int, optional): Paramètre de décalage pour le calcul des scores RRF. 60 par défaut.

        Returns:
            list[Chunk]: Liste des chunks triée par score RRF décroissant.
        """
        scores: dict[int, float] = {}
        chunks_by_id: dict[int, Chunk] = {}

        for rank, chunk in enumerate(vectorial_list):
            chunks_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank + 1)

        for rank, chunk in enumerate(bm_list):
            chunks_by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank + 1)

        ordered_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_id[chunk_id] for chunk_id in ordered_ids[:CHUNK_LIMIT_RRF]]
    
    async def ask(
        self,
        question: str,
        history: list,
        model_id: int | None = None,
        error_code: str | None = None,
    ) -> dict:
        """Retourne une réponse RAG complète : {answer, sources}."""
        logger.info("Requête /chat reçue : %s (model_id=%s, error_code=%s)", question, model_id, error_code)
        embedding = await self._embed(question)

        nearest_vectorized_chunks = self._get_nearest_vectorized_chunks(embedding, model_id=model_id, error_code=error_code)
        logger.debug("%d chunk(s) trouvé(s) pour la requête /chat : %s", len(nearest_vectorized_chunks), [c.id for c in nearest_vectorized_chunks])

        nearest_bm25_chunks = self._get_nearest_bm25_chunks(question, model_id=model_id, error_code=error_code)
        logger.debug("%d chunk(s) BM25 trouvé(s) pour la requête /chat : %s", len(nearest_bm25_chunks), [c.id for c in nearest_bm25_chunks])

        nearest_rrf_chunks = self._reciprocal_rank_fusion(nearest_vectorized_chunks, nearest_bm25_chunks)
        logger.debug("%d chunk(s) fusionné(s) pour la requête /chat : %s", len(nearest_rrf_chunks), [c.id for c in nearest_rrf_chunks])

        reranked_nearest_chunks = self.reranker.rerank(
            question,
            nearest_rrf_chunks,
            top_k=5,
        )
        logger.debug("%d chunk(s) reranké(s) pour la requête /chat : %s", len(reranked_nearest_chunks), [c.id for c in reranked_nearest_chunks])

        rag_content = self._build_rag_message(question, reranked_nearest_chunks)
        full_message = self._build_messages(question, history, rag_content)
        print(full_message)
        answer = await asyncio.to_thread(self._chat, full_message, False)
        logger.info("Réponse /chat générée (%d caractères) : %s", len(answer), answer[:100] + "..." if len(answer) > 100 else answer)

        return {"answer": answer, "sources": self._build_sources(reranked_nearest_chunks)}
