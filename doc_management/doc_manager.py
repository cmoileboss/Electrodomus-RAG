"""Conversion, découpage (chunking), nettoyage et encodage des documents avant indexation en base."""

from pathlib import Path

import pandas as pd

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from requests import session
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer

from sqlalchemy.orm import Session

from database import model_repository
from database.document_repository import DocumentRepository
from database.chunk_repository import ChunkRepository
from database.error_repository import ErrorRepository
from database.model_repository import ModelRepository
from database.models import Chunk

import re

from logger import get_logger

logger = get_logger(__name__)


class DocManager:
    """Convertit un fichier source en chunks nettoyés, contextualisés, encodés puis persistés."""

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

    @staticmethod
    def _process_errors_file(session: Session):
        sheets = pd.read_excel(
            r"C:\Users\guillaume.pedrona\Documents\Projets\Electrodomus-RAG\Documentation_Electrodomus\Referentiel_codes_erreur.xlsx",
            sheet_name=None,
            header=3
        )

        errors_repo = ErrorRepository(session)
        model_repo = ModelRepository(session)

        for sheet_name, df in sheets.items():
            for index, row in df.iterrows():
                print(f"Processing row {index} in sheet {sheet_name}")
                code = row['Code']
                signification = row['Signification']
                client_behaviour = row['Conduite à tenir (client)']
                ass_intervention = row['Intervention SAV (technicien)']
                models = row['Modèles concernés']
                models_list = models.split(',') if pd.notna(models) else []
                for model_name in models_list:
                    model_name = model_name.strip()
                    logger.debug("Processing model '%s' for error code '%s'", model_name, code)
                    model = model_repo.get_by_name(model_name)
                    if not model:
                        logger.warning("Modèle '%s' introuvable, code d'erreur '%s' ignoré", model_name, code)
                        continue
                    errors_repo.create(
                        signification=signification,
                        client_behaviour=client_behaviour,
                        ass_intervention=ass_intervention,
                        code=code,
                        model_id=model.id,
                    )
                session.commit()
                logger.info("Processed error code '%s' with models: %s", code, ", ".join(models_list))

    def process_document(self, doc_source: str, session: Session):
        """Convertit, découpe, encode et persiste les chunks d'un document (créé s'il n'existe pas)."""
        path = Path(doc_source)
        filepath = str(path.resolve())
        title = path.stem

        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)
        model_repo = ModelRepository(session)
        error_repo = ErrorRepository(session)

        document = doc_repo.get_by_filepath(filepath)
        if not document:
            document = doc_repo.create(title=title, filepath=filepath)
            logger.info("Document créé : '%s' (%s)", title, filepath)
        else:
            logger.info("Document déjà existant : '%s'", title)
            return

        if title.lower() == "referentiel_codes_erreur":
            self._process_errors_file(session)

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

            new_chunk = Chunk(
                document_id=document.id,
                chunk_index=i,
                content=content,
                embedding_text=embedding_text,
                section=section,
                page=page,
                embedding=embedding.tolist(),
            )
            # Ajouté tout de suite pour éviter les avertissements d'autoflush sur les relations Model/ErrorCode
            session.add(new_chunk)

            model_names = self._search_models_for_chunk(new_chunk.embedding_text, model_repo)
            matched_models = []
            seen_model_ids = set()
            for model_name in model_names:
                model = model_repo.get_by_name(model_name)
                if model and model.id not in seen_model_ids:
                    new_chunk.models.append(model)
                    matched_models.append(model)
                    seen_model_ids.add(model.id)

            seen_error_keys = set()
            for model in matched_models:
                error_codes = self._search_errors_for_chunk(new_chunk.embedding_text, error_repo, model.id)
                for code in error_codes:
                    key = (code, model.id)
                    if key in seen_error_keys:
                        continue
                    error = error_repo.get_by_id(code, model.id)
                    if error:
                        new_chunk.error_codes.append(error)
                        seen_error_keys.add(key)

            db_chunks.append(new_chunk)

            logger.debug("Ajout en BDD - Document '%s' Chunk %d : section='%s', page=%s", title, i, section, page)

        chunk_repo.bulk_create(db_chunks)
        session.commit()
        logger.info("%d chunk(s) insérés pour le document '%s'.", len(db_chunks), title)

    def _search_models_for_chunk(self, embedding_text: str, model_repo: ModelRepository) -> list[str]:
        codes = model_repo.get_all()
        code_names = [code.name for code in codes]

        pattern = re.compile(
            "|".join(
                f"(?P<code_{i}>"
                + r"\s*-?\s*".join(code_name.replace("-", ""))
                + ")"
                for i, code_name in enumerate(code_names)
            ),
            re.IGNORECASE
        )

        resultats = []

        for match in pattern.finditer(embedding_text):
            for i, code_name in enumerate(code_names):
                if match.group(f"code_{i}") is not None:
                    resultats.append(code_name)
                    break
        return resultats
    
    def _search_errors_for_chunk(self, embedding_text: str, error_repo: ErrorRepository, model_id: int) -> list[str]:
        errors = error_repo.get_error_codes_by_model(model_id)
        error_codes = [error.code for error in errors]

        pattern = re.compile(
            "|".join(
                f"(?P<error_{i}>"
                + r"\s*-?\s*".join(error_code.replace("-", ""))
                + ")"
                for i, error_code in enumerate(error_codes)
            ),
            re.IGNORECASE
        )

        resultats = []

        for match in pattern.finditer(embedding_text):
            for i, error_code in enumerate(error_codes):
                if match.group(f"error_{i}") is not None:
                    resultats.append(error_code)
                    break
        return resultats