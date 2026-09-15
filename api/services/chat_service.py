import asyncio

from database.chunk_repository import ChunkRepository
from database.database import get_session
from logger import get_logger
from ollama_client import SYSTEM_PROMPT, build_rag_message, chat, stream_chat

logger = get_logger(__name__)


class ChatService:
    def __init__(self, embed_model):
        self.embed_model = embed_model

    async def _embed(self, question: str):
        return await asyncio.to_thread(
            self.embed_model.encode, question, normalize_embeddings=True
        )

    def _get_chunks(self, embedding, limit: int) -> list:
        with get_session() as session:
            return ChunkRepository(session).get_nearest(embedding.tolist(), limit=limit)

    @staticmethod
    def _build_messages(question: str, history: list, rag_content: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": rag_content})
        return messages

    @staticmethod
    def _build_sources(chunks: list) -> list[dict]:
        return [
            {"section": c.section, "page": c.page, "document_id": c.document_id}
            for c in chunks
        ]

    async def ask(self, question: str, history: list, limit: int) -> dict:
        """Retourne une réponse RAG complète : {answer, sources}."""
        logger.info("Requête /chat reçue : %s", question)
        embedding = await self._embed(question)
        chunks = self._get_chunks(embedding, limit)
        logger.debug("%d chunk(s) trouvé(s) pour la requête /chat", len(chunks))

        rag_content = build_rag_message(question, chunks)
        messages = self._build_messages(question, history, rag_content)

        answer = await asyncio.to_thread(chat, messages, False)
        logger.info("Réponse /chat générée (%d caractères).", len(answer))

        return {"answer": answer, "sources": self._build_sources(chunks)}

    async def stream_answer(self, question: str, history: list, limit: int):
        """Génère les événements SSE : sources, tokens successifs, puis l'historique final."""
        logger.info("Requête /chat/stream reçue : %s", question)
        embedding = await self._embed(question)
        chunks = self._get_chunks(embedding, limit)
        logger.debug("%d chunk(s) trouvé(s) pour la requête /chat/stream", len(chunks))

        rag_content = build_rag_message(question, chunks)
        messages = self._build_messages(question, history, rag_content)

        yield {"type": "sources", "sources": self._build_sources(chunks)}

        full_answer = ""
        async for token in stream_chat(messages):
            full_answer += token
            yield {"type": "token", "content": token}

        updated_history = [
            {"role": m.role, "content": m.content} for m in history
        ] + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_answer},
        ]
        logger.info("Réponse /chat/stream terminée (%d caractères).", len(full_answer))
        yield {"type": "done", "history": updated_history}
