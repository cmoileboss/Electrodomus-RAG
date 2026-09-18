"""Accès aux données pour l'entité Chunk : CRUD, recherche vectorielle et recherche BM25."""

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database.models import Chunk, ErrorCode, Model
from logger import get_logger

from dotenv import load_dotenv
import os
    
load_dotenv()

CHUNK_LIMIT_VECTORIAL = int(os.getenv("CHUNK_LIMIT_VECTORIAL", "30"))
CHUNK_LIMIT_BM25 = int(os.getenv("CHUNK_NB_LIMIT", "30"))

logger = get_logger(__name__)


class ChunkRepository:
    """Opérations d'accès aux données pour l'entité Chunk."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding_text: str,
        section: str | None = None,
        page: int | None = None,
        embedding: list[float] | None = None,
    ) -> Chunk:
        chunk = Chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            embedding_text=embedding_text,
            section=section,
            page=page,
            embedding=embedding,
        )
        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)
        logger.debug("Chunk %d créé (document_id=%d)", chunk.id, document_id)
        return chunk

    def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        """Persiste plusieurs chunks en une seule transaction."""
        self.session.add_all(chunks)
        self.session.commit()
        logger.debug("%d chunk(s) créés en masse", len(chunks))
        return chunks

    def get_by_id(self, chunk_id: int) -> Chunk | None:
        """Retourne un chunk par son id, ou None si introuvable."""
        return self.session.get(Chunk, chunk_id)

    def get_by_document(self, document_id: int) -> list[Chunk]:
        """Retourne les chunks d'un document, ordonnés par index."""
        return (
            self.session.query(Chunk)
            .filter_by(document_id=document_id)
            .order_by(Chunk.chunk_index)
            .all()
        )

    def count(self) -> int:
        """Retourne le nombre total de chunks."""
        return self.session.query(Chunk).count()

    def get_nearest(
        self,
        embedding: list[float],
        model_id: int | None = None,
        error_code: str | None = None,
    ) -> list[Chunk]:
        """Recherche les chunks les plus proches par similarité cosinus, filtrés par modèle et/ou code d'erreur."""
        query = self.session.query(Chunk).filter(Chunk.embedding.isnot(None))
        if model_id is not None:
            query = query.join(Chunk.models).filter(Model.id == model_id)
        if error_code is not None:
            query = query.join(Chunk.error_codes).filter(ErrorCode.code == error_code)
        chunks = (
            query
            .order_by(Chunk.embedding.cosine_distance(embedding))
            .limit(CHUNK_LIMIT_VECTORIAL)
            .all()
        )
        chunk_ids = { c.id for c in chunks }
        logger.debug("%d chunk(s) trouvés par similarité cosinus : %s", len(chunks), chunk_ids)
        return chunks

    def search_bm25(
        self,
        query: str,
        model_id: int | None = None,
        error_code: str | None = None,
    ) -> list[Chunk]:
        """Recherche BM25 via pg_textsearch (index chunks_bm25_idx sur content), filtrée par modèle et/ou code d'erreur, triée par pertinence."""
        joins = []
        params = {"query": query, "limit": CHUNK_LIMIT_BM25}
        if model_id is not None:
            joins.append("JOIN chunk_model cm ON cm.chunk_id = chunks.id AND cm.model_id = :model_id")
            params["model_id"] = model_id
        if error_code is not None:
            joins.append("JOIN chunk_error_code cec ON cec.chunk_id = chunks.id AND cec.error_code = :error_code")
            params["error_code"] = error_code
        sql = (
            "SELECT chunks.id, chunks.content <@> to_bm25query(:query, 'chunks_bm25_idx') AS score "
            "FROM chunks " + " ".join(joins) + " ORDER BY score LIMIT :limit"
        )
        rows = self.session.execute(text(sql), params).fetchall()
        if not rows:
            logger.debug("Aucun résultat BM25 pour la requête : %s", query)
            return []
        id_score = {row.id: row.score for row in rows}
        chunks = self.session.query(Chunk).filter(Chunk.id.in_(id_score)).all()
        # pg_textsearch renvoie des scores négatifs : le plus proche de 0 est le meilleur
        chunks.sort(key=lambda c: id_score[c.id])
        logger.debug("%d chunk(s) trouvés par BM25 pour la requête : %s", len(chunks), id_score.items())
        return chunks

    def delete(self, chunk_id: int) -> bool:
        """Supprime un chunk, retourne True si la suppression a eu lieu."""
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            logger.debug("Chunk %d introuvable pour suppression", chunk_id)
            return False
        self.session.delete(chunk)
        self.session.commit()
        logger.debug("Chunk %d supprimé", chunk_id)
        return True

    def delete_by_document(self, document_id: int) -> int:
        """Supprime tous les chunks d'un document, retourne le nombre de lignes supprimées."""
        deleted = (
            self.session.query(Chunk).filter_by(document_id=document_id).delete()
        )
        self.session.commit()
        logger.debug("%d chunk(s) supprimés pour le document %d", deleted, document_id)
        return deleted
