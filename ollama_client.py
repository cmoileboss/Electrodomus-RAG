import requests
import json
from sentence_transformers import SentenceTransformer

from database.chunk_repository import ChunkRepository
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:latest"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

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

    response = requests.post(OLLAMA_URL, json=payload, stream=stream)
    if not response.ok:
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


def build_rag_message(user_input: str, chunks: list) -> str:
    if not chunks:
        context = "Aucun extrait pertinent trouvé."
    else:
        context = "\n\n---\n\n".join(
            f"[Source : {c.section or 'inconnue'}, page {c.page or '?'}]\n{c.content}"
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
            similar_chunks = ChunkRepository(session).get_nearest(embedding, limit=5)
        logger.info(f"{len(similar_chunks)} chunks trouvés : {[c.id for c in similar_chunks]}")
        rag_content = build_rag_message(user_input, similar_chunks)
        
        history.append({"role": "user", "content": rag_content})
        print("Assistant : ", end="")
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
