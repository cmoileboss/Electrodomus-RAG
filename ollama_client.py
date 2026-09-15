import json
import os

import httpx
import requests

from database.chunk_repository import ChunkRepository
from database.database import get_session
from logger import get_logger
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
logger = get_logger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

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

def chat(messages: list[dict], stream: bool = True) -> str:
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


async def stream_chat(messages: list[dict]):
    """Async generator yielding tokens from Ollama as they arrive."""
    payload = {"model": MODEL, "messages": messages, "stream": True}
    logger.debug("Démarrage du stream Ollama (modèle=%s, %d message(s))", MODEL, len(messages))
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    logger.debug("Stream Ollama terminé.")
                    break


def build_rag_message(user_input: str, chunks: list) -> str:
    if not chunks:
        context = "Aucun extrait pertinent trouvé."
    else:
        context = "\n\n---\n\n".join(
            f"[Source : chunk {c.id}, document {c.document_id or '?'}, {c.section or 'inconnue'}, page {c.page or '?'}]\n{c.content}"
            for c in chunks
        )
    return f"CONTEXTE :\n{context}\n\nQUESTION : {user_input}"


def main():
    print(f"Chat avec {MODEL} (tapez 'exit' pour quitter)\n")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("Vous : ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        logger.info(f"User input: {user_input}")
        embedding = embed_model.encode(user_input, normalize_embeddings=True)
        logger.info(f"Embedding : {embedding.shape[0]} dimensions.")


        with get_session() as session:
            similar_vectorized_chunks = ChunkRepository(session).get_nearest(embedding, limit=5)
            logger.info(f"{len(similar_vectorized_chunks)} chunks trouvés par comparaison vectorielle (similarité cosinus) : {[c.id for c in similar_vectorized_chunks]}")
            similar_bm25_chunks = ChunkRepository(session).search_bm25(user_input, limit=5)
            logger.info(f"{len(similar_bm25_chunks)} chunks trouvés par recherche BM25 : {[c.id for c in similar_bm25_chunks]}")

        similar_chunks = similar_vectorized_chunks & similar_bm25_chunks
        logger.info(f"{len(similar_chunks)} chunks trouvés par combinaison des deux méthodes : {[c.id for c in similar_chunks]}")
        rag_content = build_rag_message(user_input, similar_chunks)
        
        history.append({"role": "user", "content": rag_content})
        print("Assistant : ", end="")
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
