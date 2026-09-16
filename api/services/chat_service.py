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

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
CHUNK_NB_LIMIT = int(os.getenv("CHUNK_NB_LIMIT", "30"))

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

    def __init__(self, embed_model):
        self.embed_model = embed_model

    async def _embed(self, question: str):
        """Calcule l'embedding normalisé de la question, dans un thread séparé."""
        return await asyncio.to_thread(
            self.embed_model.encode, question, normalize_embeddings=True
        )

    @staticmethod
    def _get_nearest_vectorized_chunks(embedding, limit: int = CHUNK_NB_LIMIT) -> list[Chunk]:
        """Recherche les chunks les plus proches par similarité vectorielle (cosinus)."""
        with get_session() as session:
            return ChunkRepository(session).get_nearest(embedding.tolist(), limit=limit)

    @staticmethod
    def _get_nearest_bm25_chunks(question: str, limit: int = CHUNK_NB_LIMIT) -> list[Chunk]:
        """Recherche les chunks les plus pertinents par correspondance lexicale (BM25)."""
        with get_session() as session:
            return ChunkRepository(session).search_bm25(question, limit=limit)

    @staticmethod
    def _merge_scores(vector_chunks: list[Chunk], bm25_chunks: list[Chunk], limit: int = CHUNK_NB_LIMIT, k: int = 60) -> list[Chunk]:
        """Fusionne les résultats vectoriels et BM25 par Reciprocal Rank Fusion (RRF), sur l'id du chunk."""
        scores: dict[int, float] = {}
        chunks_by_id: dict[int, object] = {}

        for rang, chunk in enumerate(vector_chunks, start=1):
            chunks_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rang)

        for rang, chunk in enumerate(bm25_chunks, start=1):
            chunks_by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rang)

        ordered_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_id[chunk_id] for chunk_id in ordered_ids[:limit]]

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
        """Effectue la fusion par rang réciproque (RRF) de deux listes de résultats.

        Args:
            vectorial_list (list[Chunk]): Liste des chunks triés par similarité vectorielle.
            bm_list (list[Chunk]): Liste des chunks triés par score BM25.
            k (int, optional): Paramètre de décalage pour le calcul des scores RRF. 60 par défaut.

        Returns:
            list[tuple[Chunk, float]]: Liste des tuples (chunk, score RRF) triée par score décroissant.
        """
        scores = {}

        for rank, element in enumerate(vectorial_list):
            scores[element] = 1 / (k + rank + 1)

        for rank, element in enumerate(bm_list):
            scores[element] = scores.get(element, 0) + 1 / (k + rank + 1)

        sorted_chunks =sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [c[0] for c in sorted_chunks]
    
    async def ask(self, question: str, history: list, limit: int = CHUNK_NB_LIMIT) -> dict:
        """Retourne une réponse RAG complète : {answer, sources}."""
        logger.info("Requête /chat reçue : %s", question)
        embedding = await self._embed(question)

        nearest_vectorized_chunks = self._get_nearest_vectorized_chunks(embedding, limit)
        logger.debug("%d chunk(s) trouvé(s) pour la requête /chat (limite : %d): %s", len(nearest_vectorized_chunks), limit, [c.id for c in nearest_vectorized_chunks])

        nearest_bm25_chunks = self._get_nearest_bm25_chunks(question, limit)
        logger.debug("%d chunk(s) BM25 trouvé(s) pour la requête /chat (limite : %d): %s", len(nearest_bm25_chunks), limit, [c.id for c in nearest_bm25_chunks])

        nearest_chunks = self._reciprocal_rank_fusion(nearest_vectorized_chunks, nearest_bm25_chunks)
        logger.debug("%d chunk(s) fusionné(s) pour la requête /chat : %s", len(nearest_chunks), [c.id for c in nearest_chunks])

        rag_content = self._build_rag_message(question, nearest_chunks)
        full_message = self._build_messages(question, history, rag_content)
    
        answer = await asyncio.to_thread(self._chat, full_message, False)
        logger.info("Réponse /chat générée (%d caractères) : %s", len(answer), answer[:100] + "..." if len(answer) > 100 else answer)

        return {"answer": answer, "sources": self._build_sources(nearest_chunks)}
