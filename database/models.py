"""Modèles SQLAlchemy : Document, Chunk et Model, ainsi que leur association many-to-many."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Table, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Classe de base déclarative SQLAlchemy commune à tous les modèles."""

    pass


# Table d'association many-to-many entre chunks et models
chunk_model = Table(
    "chunk_model",
    Base.metadata,
    Column("chunk_id", Integer, ForeignKey("chunks.id"), primary_key=True),
    Column("model_id", Integer, ForeignKey("models.id"), primary_key=True),
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
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_text = Column(Text, nullable=False)
    section = Column(String)
    page = Column(Integer)
    embedding = Column(Vector(1024))
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
    models = relationship("Model", secondary=chunk_model, back_populates="chunks")


class Model(Base):
    """Un modèle d'appareil Electrodomus (ex. four, lave-linge) rattaché à des chunks."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    type = Column(String, nullable=False)

    chunks = relationship("Chunk", secondary=chunk_model, back_populates="models")
