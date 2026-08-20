import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from database.database import get_session
from doc_management.doc_manager import DocManager
from logger import get_logger

logger = get_logger(__name__)

load_dotenv()

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))
DOC_FOLDER = Path("Documentation_Electrodomus")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html"}


def build_doc_manager() -> DocManager:
    logger.info(f"Initialisation du DocManager avec le modèle d'embedding : {EMBED_MODEL_ID}")
    return DocManager(
        embed_model_id=EMBED_MODEL_ID,
        hf_token=HF_TOKEN,
        max_tokens=MAX_TOKENS,
    )


def process_single(filepath: str):
    doc_manager = build_doc_manager()
    with get_session() as session:
        doc_manager.process_document(doc_source=filepath, session=session)


def process_all():
    doc_manager = build_doc_manager()
    with get_session() as session:
        for fichier in DOC_FOLDER.rglob("*"):
            if fichier.is_file() and fichier.suffix in SUPPORTED_EXTENSIONS:
                logger.info(f"Traitement du fichier : {fichier}")
                doc_manager.process_document(doc_source=str(fichier), session=session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stockage des chunks en base de données.")
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Chemin vers un fichier précis à traiter. Si absent, traite tous les documents.",
    )
    args = parser.parse_args()

    if args.filepath:
        process_single(args.filepath)
    else:
        process_all()