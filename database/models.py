"""Modèles SQLAlchemy : Document, Chunk et Model, ainsi que leur association many-to-many."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, ForeignKeyConstraint, Integer, String, Table, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Classe de base déclarative SQLAlchemy commune à tous les modèles."""
    pass


# Table d'association many-to-many entre chunks et models
chunk_model = Table(
    "chunk_model",
    Base.metadata,
    Column("chunk_id", Integer, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
    Column("model_id", Integer, ForeignKey("models.id", ondelete="CASCADE"), primary_key=True),
)

# Table d'association many-to-many entre chunks et error_codes (clé composite code+model_id)
chunk_error_code = Table(
    "chunk_error_code",
    Base.metadata,
    Column("chunk_id", Integer, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
    Column("error_code", String, primary_key=True),
    Column("model_id", Integer, primary_key=True),
    ForeignKeyConstraint(
        ["error_code", "model_id"],
        ["error_codes.code", "error_codes.model_id"],
        ondelete="CASCADE",
    ),
)

class Document(Base):
    """Un document source (PDF, HTML, etc.) ingesté dans la documentation Electrodomus."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False, unique=True)
    filepath = Column(String, nullable=False, unique=True)
    date = Column(DateTime)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Un segment de texte extrait d'un document, avec son embedding et son texte contextualisé."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_text = Column(Text, nullable=False)
    section = Column(String)
    page = Column(Integer)
    embedding = Column(Vector(1024))
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
    models = relationship("Model", secondary=chunk_model, back_populates="chunks")
    error_codes = relationship("ErrorCode", secondary=chunk_error_code, back_populates="chunks")


class Model(Base):
    """Un modèle d'appareil Electrodomus (ex. four, lave-linge) rattaché à des chunks."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    type = Column(String, nullable=False)

    chunks = relationship("Chunk", secondary=chunk_model, back_populates="models")
    error_codes = relationship("ErrorCode", back_populates="model", cascade="all, delete-orphan")

class ErrorCode(Base):
    """Un code d'erreur rattaché à un modèle, avec ses messages associés."""

    __tablename__ = "error_codes"

    code = Column(String, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"), primary_key=True)
    signification = Column(String, nullable=False)
    client_behaviour = Column(String, nullable=False)
    ass_intervention = Column(String, nullable=False)

    model = relationship("Model", back_populates="error_codes")
    chunks = relationship("Chunk", secondary=chunk_error_code, back_populates="error_codes")