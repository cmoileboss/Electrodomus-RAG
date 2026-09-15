from pathlib import Path

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from database.document_repository import DocumentRepository
from database.chunk_repository import ChunkRepository
from database.models import Chunk

import re

from logger import get_logger

logger = get_logger(__name__)


class DocManager:
    def __init__(self, embed_model_id: str, hf_token: str, max_tokens: int):
        logger.info(f"Initialisation du DocManager avec le modèle d'embedding : {embed_model_id}")
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                embed_model_id,
                token=hf_token
            ),
            max_tokens=max_tokens,
        )

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,   # optional, defaults to True, permet de fusionner des éléments voisins
                                # du document lorsqu'ils sont au même niveau hiérarchique,
                                # afin de former des chunks plus cohérents
        )
        self.model = SentenceTransformer(embed_model_id)
        self.converter = DocumentConverter()


    def _clean_chunk(self, text: str) -> str:
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

    def process_document(self, doc_source: str, session: Session):
        filepath = str(Path(doc_source).resolve())
        title = Path(doc_source).stem

        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        document = doc_repo.get_by_filepath(filepath)
        if not document:
            document = doc_repo.create(title=title, filepath=filepath)
            logger.info("Document créé : '%s' (%s)", title, filepath)
        else:
            logger.info("Document déjà existant : '%s'", title)

        logger.debug("Conversion du document : %s", doc_source)
        doc = self.converter.convert(source=doc_source).document
        chunk_iter = self.chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)

        embedding_texts = [
            self._clean_chunk(self.chunker.contextualize(chunk=chunk))
            for chunk in chunks
        ]
        logger.debug("Encodage de %d chunks pour '%s'", len(chunks), title)
        embeddings = self.model.encode(
            embedding_texts,
            normalize_embeddings=True,
            batch_size=32,
        )

        db_chunks = []
        for i, (chunk, embedding_text, embedding) in enumerate(zip(chunks, embedding_texts, embeddings)):
            logger.debug("Chunk brut %d : %s", i, chunk)
            content = self._clean_chunk(chunk.text)
            section = chunk.meta.headings[0] if chunk.meta.headings else None
            page = None
            if chunk.meta.doc_items:
                provs = chunk.meta.doc_items[0].prov
                if provs:
                    page = provs[0].page_no

            db_chunks.append(Chunk(
                document_id=document.id,
                chunk_index=i,
                content=content,
                embedding_text=embedding_text,
                section=section,
                page=page,
                embedding=embedding.tolist(),
            ))
            logger.debug("Ajout en BDD - Document '%s' Chunk %d : section='%s', page=%s", title, i, section, page)

        chunk_repo.bulk_create(db_chunks)
        logger.info("%d chunk(s) insérés pour le document '%s'.", len(db_chunks), title)
