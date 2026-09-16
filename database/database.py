"""Connexion \u00e0 la base PostgreSQL : configuration du moteur, sessions et initialisation du sch\u00e9ma."""

import os
import sys
from pathlib import Path

# Garantit que la racine du projet est dans sys.path quelle que soit la façon dont le fichier est lancé
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import sessionmaker

from database.models import Base, Model
from logger import get_logger

logger = get_logger(__name__)

load_dotenv()

DRIVER_NAME = os.getenv("DRIVER_NAME", "postgresql+psycopg")
DATABASE_NAME = os.getenv("DATABASE_NAME", "electrodomus")
DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "postgres")
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = int(os.getenv("DATABASE_PORT", "5433"))

database_url = URL.create(
    drivername=DRIVER_NAME,
    username=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT,
    database=DATABASE_NAME,
)

def get_session():
    """Crée un nouveau moteur et retourne une session SQLAlchemy."""
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()

DEFAULT_MODELS = [
    ("FR-600", "four encastrable"), ("FR-650", "four encastrable"),
    ("FR-700", "four encastrable"), ("FR-750", "four encastrable"),
    ("FR-800", "four encastrable"), ("FR-850", "four encastrable"),
    ("LV-100", "lave-vaisselle"), ("LV-150", "lave-vaisselle"),
    ("LV-200", "lave-vaisselle"), ("LV-250", "lave-vaisselle"),
    ("LV-300", "lave-vaisselle"), ("LV-350", "lave-vaisselle"),
    ("WX-300", "lave-linge"), ("WX-350", "lave-linge"),
    ("WX-400", "lave-linge"), ("WX-450", "lave-linge"),
    ("WX-500", "lave-linge"), ("WX-550", "lave-linge"),
]


def init_db():
    """Crée les extensions Postgres, les tables, l'index BM25 et sème les modèles par défaut."""
    engine = create_engine(database_url, echo=True)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("Extension 'vector' créée avec succès.")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch"))
        conn.commit()
        logger.info("Extension 'pg_textsearch' créée avec succès.")
    Base.metadata.create_all(engine)
    logger.info("Tables créées avec succès.")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_bm25_idx
            ON chunks
            USING bm25 (content)
            WITH (
                text_config = 'french'
            )
        """))
        conn.commit()
        logger.info("Index 'chunks_bm25_idx' créé avec succès.")
    _seed_models(engine)


def _seed_models(engine):
    """Insère les modèles d'appareils par défaut absents de la base."""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        existing = {m.name for m in session.query(Model.name).all()}
        new_models = [
            Model(name=name, type=type_)
            for name, type_ in DEFAULT_MODELS
            if name not in existing
        ]
        if new_models:
            session.add_all(new_models)
            session.commit()
            logger.info("%d modèle(s) ajouté(s).", len(new_models))
        else:
            logger.debug("Aucun nouveau modèle à insérer.")


if __name__ == "__main__":
    init_db()
