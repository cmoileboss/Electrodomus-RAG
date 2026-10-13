"""Service RAG : embedding de la question, recherche vectorielle + BM25, fusion RRF et génération via Groq."""

import asyncio
import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from database.chunk_repository import ChunkRepository
from database.database import get_session
from database.models import Chunk

from logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PROMPT_LOG_DIR = Path("logs/prompts")
RESPONSE_LOG_DIR = Path("logs/responses")

CHUNK_LIMIT_RRF = int(os.getenv("CHUNK_LIMIT_RRF", "10"))
CHUNK_LIMIT_CROSS_ENCODER = int(os.getenv("CHUNK_LIMIT_CROSS_ENCODER", "10"))
TOP_K_RERANKER = int(os.getenv("TOP_K_RERANKER", "3"))

SYSTEM_PROMPT = """\
Tu es un assistant technique expert pour la marque Electrodomus, spécialisé dans le dépannage,
l'installation et l'entretien des appareils électroménagers (fours, lave-linge, lave-vaisselle).

Règles absolues :
- Réponds UNIQUEMENT à partir des extraits de documentation fournis dans le CONTEXTE.
- Si la réponse ne figure pas dans le contexte, réponds exactement :
  "Je n'ai pas trouvé cette information dans la documentation Electrodomus."
- Ne complète jamais avec tes connaissances générales.
- Cite TOUJOURS la source (section / page) entre parenthèses après chaque information.
- Si plusieurs extraits se contredisent, privilégie les informations les plus récentes.
- Réponds en français, de façon claire, structurée et concise.
- Utilise du Markdown standard : chaque ligne d'un tableau doit être sur sa propre ligne.
- N'utilise pas de balises HTML (notamment <br>) ; dans un tableau, garde les cellules concises ou préfère une liste si le contenu est détaillé.
"""


def _write_prompt_to_file(messages: list[dict]) -> Path:
    """Écrit uniquement les rôles et contenus du prompt dans un fichier lisible."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"prompt_{timestamp}_{uuid4().hex[:8]}.txt"
    prompt_path = PROMPT_LOG_DIR / filename
    prompt_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    for message in messages:
        sections.append(f"[{message['role'].upper()}]\n\n{message['content']}")

    prompt_path.write_text(("\n\n" + "-" * 80 + "\n\n").join(sections) + "\n", encoding="utf-8")
    return prompt_path


def _write_response_to_file(response: str) -> Path:
    """Écrit le texte brut de la réponse du LLM dans un fichier distinct."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"response_{timestamp}_{uuid4().hex[:8]}.txt"
    response_path = RESPONSE_LOG_DIR / filename
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response, encoding="utf-8")
    return response_path


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
        """Envoie les messages à Groq et retourne la réponse complète (streaming ou non)."""
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY n'est pas configurée.")
        if not MODEL:
            raise RuntimeError("GROQ_MODEL n'est pas configuré.")

        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        logger.debug("Envoi de %d message(s) à Groq (modèle=%s, stream=%s)", len(messages), MODEL, stream)
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=120,
        )
        if not response.ok:
            logger.error("Erreur Groq %s : %s", response.status_code, response.text)
            raise RuntimeError(f"Groq {response.status_code}: {response.text}")

        full_response = ""

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                event_data = line.removeprefix("data:").strip()
                if event_data == "[DONE]":
                    break
                chunk = json.loads(event_data)
                choices = chunk.get("choices", [])
                token = choices[0].get("delta", {}).get("content", "") if choices else ""
                print(token, end="", flush=True)
                full_response += token
            print()
        else:
            data = response.json()
            full_response = data["choices"][0]["message"]["content"]

        return full_response


    @staticmethod
    def _build_rag_message(user_input: str, chunks: list) -> str:
        """Construit le message utilisateur contenant le contexte des chunks et la question."""
        if not chunks:
            context = "Aucun extrait pertinent trouvé."
        else:
            context = "\n\n---\n\n".join(
                f"[Source : chunk {c.id}, document {c.document_id or 'inconnu'}, {c.section or 'inconnue'}, page {c.page or 'inconnu'}, date : {c.date or 'inconnue'}]\n{c.content}"
                for c in chunks
            )
        return f"CONTEXTE :\n{context}\n\nQUESTION : {user_input}"

    @staticmethod
    def _build_messages(question: str, history: list, rag_content: str) -> list[dict]:
        """Assemble les messages : prompt système, historique puis message RAG."""
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
            top_k=TOP_K_RERANKER,
        )
        logger.debug("%d chunk(s) reranké(s) pour la requête /chat : %s", len(reranked_nearest_chunks), [c.id for c in reranked_nearest_chunks])

        rag_content = self._build_rag_message(question, reranked_nearest_chunks)
        full_message = self._build_messages(question, history, rag_content)
        prompt_path = _write_prompt_to_file(full_message)
        logger.debug("Prompt envoyé au LLM écrit dans %s", prompt_path)
        logger.debug("Prompt complet : %s", full_message)
        answer = await asyncio.to_thread(self._chat, full_message, False)
        response_path = _write_response_to_file(answer)
        logger.debug("Réponse du LLM enregistrée dans %s", response_path)
        logger.info("Réponse /chat générée (%d caractères) : %s", len(answer), answer[:100] + "..." if len(answer) > 100 else answer)

        return {"answer": answer, "sources": self._build_sources(reranked_nearest_chunks)}
