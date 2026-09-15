from pgvector.sqlalchemy import Vector
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database.models import Chunk
from logger import get_logger

logger = get_logger(__name__)


class ChunkRepository:
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
        self.session.add_all(chunks)
        self.session.commit()
        logger.debug("%d chunk(s) créés en masse", len(chunks))
        return chunks

    def get_by_id(self, chunk_id: int) -> Chunk | None:
        return self.session.get(Chunk, chunk_id)

    def get_by_document(self, document_id: int) -> list[Chunk]:
        return (
            self.session.query(Chunk)
            .filter_by(document_id=document_id)
            .order_by(Chunk.chunk_index)
            .all()
        )

    def count(self) -> int:
        return self.session.query(Chunk).count()

    def get_nearest(self, embedding: list[float], limit: int = 5) -> list[Chunk]:
        """Recherche les chunks les plus proches par similarité cosinus."""
        chunks = (
            self.session.query(Chunk)
            .filter(Chunk.embedding.isnot(None))
            .order_by(Chunk.embedding.cosine_distance(embedding))
            .limit(limit)
            .all()
        )
        logger.debug("%d chunk(s) trouvés par similarité cosinus (limit=%d)", len(chunks), limit)
        return chunks

    def search_bm25(self, query: str, limit: int = 5) -> list[tuple[Chunk, float]]:
        """Recherche BM25 via pg_search, retourne (chunk, score)."""
        rows = self.session.execute(
            text("""
                SELECT id, paradedb.score(id) AS score
                FROM chunks
                WHERE chunks @@@ paradedb.parse(:query)
                ORDER BY score DESC
                LIMIT :limit
            """),
            {"query": f"content:{query} OR embedding_text:{query}", "limit": limit},
        ).fetchall()
        if not rows:
            logger.debug("Aucun résultat BM25 pour la requête : %s", query)
            return []
        id_score = {row.id: row.score for row in rows}
        chunks = self.session.query(Chunk).filter(Chunk.id.in_(id_score)).all()
        chunks.sort(key=lambda c: id_score[c.id], reverse=True)
        logger.debug("%d chunk(s) trouvés par BM25 pour la requête : %s", len(chunks), query)
        return [(c, id_score[c.id]) for c in chunks]

    def delete(self, chunk_id: int) -> bool:
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            logger.debug("Chunk %d introuvable pour suppression", chunk_id)
            return False
        self.session.delete(chunk)
        self.session.commit()
        logger.debug("Chunk %d supprimé", chunk_id)
        return True

    def delete_by_document(self, document_id: int) -> int:
        deleted = (
            self.session.query(Chunk).filter_by(document_id=document_id).delete()
        )
        self.session.commit()
        logger.debug("%d chunk(s) supprimés pour le document %d", deleted, document_id)
        return deleted
