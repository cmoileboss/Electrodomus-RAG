from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
import re
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

def clean_chunk(text: str) -> str:
    """Nettoyage conservateur d'un chunk."""

    # Espaces insécables
    text = text.replace("\xa0", " ")

    # Tirets Unicode → tiret standard
    text = text.replace("‐", "-")
    text = text.replace("-", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Espaces multiples
    text = re.sub(r"[ \t]+", " ", text)

    # Trop nombreux retours à la ligne
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = 400
DOC_SOURCE = "C:\\Users\\guillaume.pedrona\\Documents\\Projets\\Electrodomus-RAG\\Documentation_Electrodomus\\Manuels\\Manuel_FR-600.pdf"


tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(
        EMBED_MODEL_ID,
        token=HF_TOKEN
    ),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,  # optional, defaults to True, permet de fusionner des éléments voisins
                       # du document lorsqu'ils sont au même niveau hiérarchique,
                       # afin de former des chunks plus cohérents
)

doc = DocumentConverter().convert(source=DOC_SOURCE).document
chunk_iter = chunker.chunk(dl_doc=doc)
chunks = list(chunk_iter)

model = SentenceTransformer("BAAI/bge-m3")


for i, chunk in enumerate(chunks):

    content = clean_chunk(chunk.text)
    embedding_text = clean_chunk(chunker.contextualize(chunk=chunk))
    headings = chunk.meta.headings
    meta = chunk.meta

    document_embedding = model.encode_document(
        [embedding_text],
        normalize_embeddings=True
    )

    print(f"=== {i} tokens : {len(chunk.text.split())} ===")

    print("TEXT:")
    print(content)

    print("\nTexte contextualisé:")
    print(embedding_text)

    print("\nHEADINGS:")
    print(headings)

    print("\nMETA:")
    print(meta)

    print("\nVECTOR:")
    print(document_embedding)

    print()
