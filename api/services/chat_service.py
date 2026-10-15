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
from database.document_repository import DocumentRepository
from database.models import Chunk

from prompts import STANDARD_PROMPT, ADVANCED_PROMPT

from api.services.rag_config import (
    CHUNK_LIMIT_RRF,
    DEFAULT_MODEL as MODEL,
    TOP_K_RERANKER,
    RagConfig,
)
from logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
PROMPT_LOG_DIR = Path("logs/prompts")
RESPONSE_LOG_DIR = Path("logs/responses")

CHUNK_LIMIT_CROSS_ENCODER = int(os.getenv("CHUNK_LIMIT_CROSS_ENCODER", "10"))

SYSTEM_PROMPT = STANDARD_PROMPT


def _write_prompt_to_file(messages: list[dict]) -> Path:
    """Enregistre le prompt sous forme textuelle, sans autre métadonnée de requête.

    Args:
        messages: Messages contenant chacun un rôle et un contenu textuel.

    Returns:
        Path: Chemin du fichier de prompt créé.
    """
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
    """Enregistre le texte brut du modèle dans un fichier de réponse distinct.

    Args:
        response: Réponse textuelle complète à archiver.

    Returns:
        Path: Chemin du fichier de réponse créé.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"response_{timestamp}_{uuid4().hex[:8]}.txt"
    response_path = RESPONSE_LOG_DIR / filename
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response, encoding="utf-8")
    return response_path


class ChatService:
    """Orchestre le pipeline RAG : recherche de contexte pertinent puis génération de réponse."""

    def __init__(self, embed_model, reranker):
        """Construit le service avec les modèles partagés par l'application.

        Args:
            embed_model: Modèle fournissant la méthode ``encode`` pour les questions.
            reranker: Composant qui reclasse les chunks candidats.

        Returns:
            None: Le constructeur conserve les dépendances pour les appels suivants.
        """
        self.embed_model = embed_model
        self.reranker = reranker

    async def _embed(self, question: str):
        """Calcule l'embedding normalisé de la question hors de la boucle asynchrone.

        Args:
            question: Texte de la question à convertir en vecteur.

        Returns:
            Embedding: Vecteur produit par ``embed_model.encode`` avec normalisation.
        """
        return await asyncio.to_thread(
            self.embed_model.encode, question, normalize_embeddings=True
        )

    @staticmethod
    def _get_nearest_vectorized_chunks(
        embedding, model_id: int | None = None, error_code: str | None = None, limit: int | None = None
    ) -> list[Chunk]:
        """Recherche les chunks les plus proches selon la distance cosinus.

        Args:
            embedding: Vecteur de la question, exposant la méthode ``tolist``.
            model_id: Identifiant facultatif du modèle d'appareil à filtrer.
            error_code: Code d'erreur facultatif à utiliser comme filtre.
            limit: Nombre maximal de chunks; valeur par défaut du dépôt si absent.

        Returns:
            list[Chunk]: Chunks classés par proximité vectorielle.
        """
        with get_session() as session:
            return ChunkRepository(session).get_nearest(
                embedding.tolist(), model_id=model_id, error_code=error_code, limit=limit
            )

    @staticmethod
    def _get_nearest_bm25_chunks(
        question: str, model_id: int | None = None, error_code: str | None = None, limit: int | None = None
    ) -> list[Chunk]:
        """Recherche des chunks par correspondance lexicale BM25.

        Args:
            question: Texte soumis à la recherche BM25.
            model_id: Identifiant facultatif du modèle d'appareil à filtrer.
            error_code: Code d'erreur facultatif à utiliser comme filtre.
            limit: Nombre maximal de chunks; valeur par défaut du dépôt si absent.

        Returns:
            list[Chunk]: Chunks classés par pertinence lexicale.
        """
        with get_session() as session:
            return ChunkRepository(session).search_bm25(
                question, model_id=model_id, error_code=error_code, limit=limit
            )

    @staticmethod
    def _chat(messages: list[dict], stream: bool = True, model: str | None = None) -> str:
        """Envoie une conversation à Groq et rassemble le texte généré.

        Args:
            messages: Messages de conversation au format rôle/contenu de l'API.
            stream: Active la lecture et l'affichage progressifs des tokens si vrai.
            model: Modèle Groq à utiliser; ``MODEL`` si absent.

        Returns:
            str: Réponse complète produite par le modèle.

        Raises:
            RuntimeError: Si la clé API ou le nom du modèle manque, ou si Groq
                retourne une réponse HTTP en erreur.
            requests.RequestException: Si la requête HTTP échoue au niveau transport.
        """
        model = model or MODEL
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY n'est pas configurée.")
        if not model:
            raise RuntimeError("GROQ_MODEL n'est pas configuré.")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        logger.debug("Envoi de %d message(s) à Groq (modèle=%s, stream=%s)", len(messages), model, stream)
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
        """Formate le contexte documentaire et la question pour le modèle de langage.

        Args:
            user_input: Question originale de l'utilisateur.
            chunks: Chunks retenus et leurs métadonnées de source.

        Returns:
            str: Message combinant contexte, références de source et question.
        """
        if not chunks:
            context = "Aucun extrait pertinent trouvé."
        else:
            # Ajout de la date du document
            for c in chunks:
                with get_session() as session:
                    doc_repo = DocumentRepository(session)
                    document = doc_repo.get_by_id(c.document_id) if c.document_id else None
                    c.date = document.date if document else None
            context = "\n\n---\n\n".join(
                f"[Source : chunk {c.id}, document {c.document_id or 'inconnu'}, {c.section or 'inconnue'}, page {c.page or 'inconnu'}, date {c.date or 'inconnue'}]\n{c.content}"
                for c in chunks
            )
        return f"CONTEXTE :\n{context}\n\nQUESTION : {user_input}"

    @staticmethod
    def _build_messages(question: str, history: list, rag_content: str) -> list[dict]:
        """Assemble le prompt système, l'historique et le contexte RAG.

        Args:
            question: Question originale (conservée dans l'interface, le contexte
                question étant déjà intégré à ``rag_content``).
            history: Messages antérieurs dotés des attributs ``role`` et ``content``.
            rag_content: Message utilisateur contenant le contexte et la question.

        Returns:
            list[dict]: Messages au format rôle/contenu attendus par l'API Groq.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": rag_content})
        return messages

    @staticmethod
    def _build_sources(chunks: list) -> list[dict]:
        """Extrait les métadonnées de provenance des chunks retenus.

        Args:
            chunks: Chunks utilisés pour construire le contexte de réponse.

        Returns:
            list[dict]: Une entrée par chunk avec sa section, sa page et son
                identifiant de document.
        """
        return [
            {"section": c.section, "page": c.page, "document_id": c.document_id}
            for c in chunks
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        vectorial_list: list[Chunk], bm_list: list[Chunk], k: int = 60, limit: int | None = None
    ) -> list[Chunk]:
        """Effectue la fusion par rang réciproque (RRF) de deux listes de résultats, dédoublonnée par id de chunk.

        Args:
            vectorial_list: Chunks ordonnés par proximité vectorielle.
            bm_list: Chunks ordonnés par pertinence BM25.
            k: Constante de décalage RRF qui réduit l'influence des rangs élevés.
            limit: Nombre maximal de chunks conservés; ``CHUNK_LIMIT_RRF`` si absent.

        Returns:
            list[Chunk]: Chunks dédoublonnés par identifiant, triés par score RRF
                décroissant et limités par ``limit``.
        """
        limit = limit if limit is not None else CHUNK_LIMIT_RRF
        scores: dict[int, float] = {}
        chunks_by_id: dict[int, Chunk] = {}

        for rank, chunk in enumerate(vectorial_list):
            chunks_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank + 1)

        for rank, chunk in enumerate(bm_list):
            chunks_by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank + 1)

        ordered_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_id[chunk_id] for chunk_id in ordered_ids[:limit]]
    
    async def ask(
        self,
        question: str,
        history: list,
        model_id: int | None = None,
        error_code: str | None = None,
        config: RagConfig | None = None,
    ) -> dict:
        """Exécute les recherches hybrides, reclasse les sources et génère une réponse.

        Args:
            question: Question textuelle à traiter.
            history: Historique des messages antérieurs de la conversation.
            model_id: Identifiant facultatif du modèle d'appareil à filtrer.
            error_code: Code d'erreur facultatif à utiliser comme filtre de recherche.
            config: Réglages du pipeline (modèle LLM et nombre de chunks à chaque étape);
                valeurs par défaut du serveur si absent.

        Returns:
            dict: Résultat contenant ``answer`` (réponse générée) et ``sources``
                (métadonnées des chunks retenus).
        """
        config = config or RagConfig()
        logger.info(
            "Requête /chat reçue : %s (model_id=%s, error_code=%s, config=%s)",
            question, model_id, error_code, config.model_dump(),
        )
        embedding = await self._embed(question)

        nearest_vectorized_chunks = self._get_nearest_vectorized_chunks(
            embedding, model_id=model_id, error_code=error_code, limit=config.vector_limit
        )
        logger.debug("%d chunk(s) trouvé(s) pour la requête /chat : %s", len(nearest_vectorized_chunks), [c.id for c in nearest_vectorized_chunks])

        nearest_bm25_chunks = self._get_nearest_bm25_chunks(
            question, model_id=model_id, error_code=error_code, limit=config.bm25_limit
        )
        logger.debug("%d chunk(s) BM25 trouvé(s) pour la requête /chat : %s", len(nearest_bm25_chunks), [c.id for c in nearest_bm25_chunks])

        nearest_rrf_chunks = self._reciprocal_rank_fusion(
            nearest_vectorized_chunks, nearest_bm25_chunks, limit=config.rrf_limit
        )
        logger.debug("%d chunk(s) fusionné(s) pour la requête /chat : %s", len(nearest_rrf_chunks), [c.id for c in nearest_rrf_chunks])

        reranked_nearest_chunks = self.reranker.rerank(
            question,
            nearest_rrf_chunks,
            top_k=config.rerank_top_k,
        )
        logger.debug("%d chunk(s) reranké(s) pour la requête /chat : %s", len(reranked_nearest_chunks), [c.id for c in reranked_nearest_chunks])

        rag_content = self._build_rag_message(question, reranked_nearest_chunks)
        full_message = self._build_messages(question, history, rag_content)
        prompt_path = _write_prompt_to_file(full_message)
        logger.debug("Prompt envoyé au LLM écrit dans %s", prompt_path)
        logger.debug("Prompt complet : %s", full_message)
        answer = await asyncio.to_thread(self._chat, full_message, False, config.model)
        response_path = _write_response_to_file(answer)
        logger.debug("Réponse du LLM enregistrée dans %s", response_path)
        logger.info("Réponse /chat générée (%d caractères) : %s", len(answer), answer[:100] + "..." if len(answer) > 100 else answer)

        return {"answer": answer, "sources": self._build_sources(reranked_nearest_chunks)}
