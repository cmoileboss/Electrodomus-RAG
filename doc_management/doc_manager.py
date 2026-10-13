"""Conversion, découpage (chunking), nettoyage et encodage des documents avant indexation en base."""

from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import threading
from dataclasses import dataclass
import pandas as pd
import re, calendar

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.document import DocumentStream
from docling.document_converter import DocumentConverter

from requests import session
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer

from sqlalchemy.orm import Session

from database.document_repository import DocumentRepository
from database.chunk_repository import ChunkRepository
from database.error_repository import ErrorRepository
from database.model_repository import ModelRepository
from database.models import Chunk, Document

import re

from logger import get_logger

logger = get_logger(__name__)

# Sérialise la création des modèles détectés pendant les ingestions concurrentes
_model_creation_lock = threading.Lock()

REFERENTIEL_TITLE = "referentiel_codes_erreur"
# Nombre de lignes (titre, version, ligne vide) qui précèdent la ligne d'en-têtes du tableau du référentiel
REFERENTIEL_HEADER_ROW = 3


@dataclass
class _ChunkEntry:
    """Chunk prêt à être encodé et persisté, quelle que soit sa source (Docling ou ligne de tableau).

    ``models`` et ``error_codes`` imposent les associations quand la source les connaît;
    sinon elles sont détectées dans ``embedding_text``.
    """

    content: str
    embedding_text: str
    section: str | None
    page: int | None
    models: list[str] | None = None
    error_codes: list[str] | None = None


def compute_file_hash(filepath: str | Path) -> str:
    """Calcule l'empreinte SHA-256 du contenu d'un fichier.

    Args:
        filepath: Chemin du fichier à lire.

    Returns:
        str: Empreinte hexadécimale du contenu du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

MOIS = "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
PATTERNS = [
    (rf"\b(\d{{1,2}})(?:er)?\s+({MOIS})\s+(\d{{4}})\b", "day"),
    (rf"\b({MOIS})\s+(\d{{4}})\b", "month"),
    (r"\b(?:édition|edition|version|rév\.?|rev\.?)\s*:?\s*(\d{4})\b", "year"),
    (r"\b((?:19|20)\d{2})\b", "year"),   # dernier recours
]
MOIS_NUM = {
    m: i for i, m in enumerate(
        ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre"], start=1)
}
MOIS_NUM.update({"fevrier": 2, "aout": 8, "decembre": 12})


def extract_date(text: str) -> str | None:
    """Extrait la première date reconnue dans un texte français.

    Args:
        text: Texte source contenant éventuellement une date ou une année.

    Returns:
        str | None: Date au format ``YYYY-MM-DD``, ``YYYY-MM`` ou ``YYYY``,
            selon la précision trouvée; ``None`` si aucune date n'est reconnue.
    """
    for pattern, precision in PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if precision == "day":
            return f"{int(match.group(3)):04d}-{MOIS_NUM[match.group(2).lower()]:02d}-{int(match.group(1)):02d}"
        if precision == "month":
            return f"{int(match.group(2)):04d}-{MOIS_NUM[match.group(1).lower()]:02d}"
        return match.group(1)
    return None

def replace_summary_with_h3(document: str) -> str:
    """Remplace les balises ouvrantes et fermantes ``summary`` par des balises ``h3``.

    Args:
        document: Contenu HTML à transformer.

    Returns:
        str: Contenu HTML avec les éléments ``summary`` convertis en ``h3``.
    """
    return re.sub(
        r"<(/?)summary(?=[\s/>])([^>]*)>",
        r"<\1h3\2>",
        document,
        flags=re.IGNORECASE,
    )


class DocManager:
    """Convertit un fichier source en chunks nettoyés, contextualisés, encodés puis persistés."""

    def __init__(self, embed_model_id: str, hf_token: str, max_tokens: int):
        """Charge le tokenizer, le modèle d'embedding et les outils de conversion.

        Args:
            embed_model_id: Identifiant Hugging Face du modèle de tokenizer et d'embedding.
            hf_token: Jeton d'accès Hugging Face, requis pour les modèles privés.
            max_tokens: Longueur maximale des chunks exprimée en tokens.

        Returns:
            None: Le constructeur prépare les ressources réutilisées pour l'ingestion.
        """
        logger.info("Initialisation du DocManager avec le modèle d'embedding : %s (max_tokens=%d)", embed_model_id, max_tokens)
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                embed_model_id,
                token=hf_token
            ),
            max_tokens=max_tokens,
        )
        logger.debug("Tokenizer HuggingFace chargé pour '%s'", embed_model_id)

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,   # optional, defaults to True, permet de fusionner des éléments voisins
                                # du document lorsqu'ils sont au même niveau hiérarchique,
                                # afin de former des chunks plus cohérents
        )
        self.model = SentenceTransformer(embed_model_id)
        self.converter = DocumentConverter()
        logger.info("DocManager prêt.")


    def _clean_chunk(self, text: str) -> str:
        """Normalise les espaces et tirets d'un chunk sans changer son contenu.

        Args:
            text: Texte extrait à nettoyer.

        Returns:
            str: Texte nettoyé, sans espaces superflus en début et fin.
        """

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
    def _infer_model_type(name: str, model_repo: ModelRepository) -> str | None:
        """Déduit le type d'un modèle à partir des modèles connus partageant son préfixe.

        Args:
            name: Référence du modèle, par exemple ``FR-900``.
            model_repo: Dépôt fournissant les modèles connus.

        Returns:
            str | None: Type des modèles de même préfixe, ou ``None`` si le préfixe est inconnu.
        """
        prefix = name.split("-")[0]
        for model in model_repo.get_all():
            if model.name.split("-")[0] == prefix:
                return model.type
        return None

    @classmethod
    def _get_or_create_model(cls, model_repo: ModelRepository, name: str):
        """Retourne le modèle demandé en le créant s'il est nouveau et que son type est déductible.

        Args:
            name: Référence du modèle recherché ou à créer.
            model_repo: Dépôt de modèles utilisé pour la lecture et la création.

        Returns:
            Model | None: Modèle existant ou créé; ``None`` si le modèle est inconnu et son
                type ne peut pas être déduit.
        """
        with _model_creation_lock:
            model = model_repo.get_by_name(name)
            if model:
                return model
            model_type = cls._infer_model_type(name, model_repo)
            if not model_type:
                return None
            model = model_repo.create(name=name, type=model_type)
        logger.info("Nouveau modèle ajouté en base : '%s' (%s)", name, model_type)
        return model

    def _register_new_models(self, texts: list[str], model_repo: ModelRepository) -> None:
        """Ajoute en base les modèles absents dont la référence apparaît dans les textes.

        Seules les références construites sur un préfixe déjà connu (par exemple ``FR-``) sont
        retenues, afin d'en déduire le type de l'appareil.

        Args:
            texts: Textes des chunks à analyser.
            model_repo: Dépôt de modèles utilisé pour la lecture et la création.

        Returns:
            None: Les nouveaux modèles sont validés en base.
        """
        prefixes = {model.name.split("-")[0] for model in model_repo.get_all()}
        if not prefixes:
            return
        pattern = re.compile(rf"\b(?:{'|'.join(map(re.escape, sorted(prefixes)))})-\d{{3}}\b")
        candidates = {match for text in texts for match in pattern.findall(text)}
        for name in sorted(candidates):
            self._get_or_create_model(model_repo, name)

    @classmethod
    def _process_errors_file(cls, session: Session, filepath: str):
        """Importe le référentiel Excel : ajoute les modèles, codes et associations manquants.

        Les codes déjà présents dont le contenu a changé sont mis à jour.

        Args:
            session: Session SQLAlchemy utilisée pour lire les modèles et enregistrer
                les codes d'erreur importés.
            filepath: Chemin du fichier Excel du référentiel.

        Returns:
            None: Les modèles et associations nouveaux sont enregistrés dans la session.
        """
        logger.info("Lecture du référentiel des codes d'erreur.")
        sheets = pd.read_excel(filepath, sheet_name=None, header=REFERENTIEL_HEADER_ROW)
        logger.debug("%d feuille(s) trouvée(s) dans le référentiel : %s", len(sheets), list(sheets.keys()))

        errors_repo = ErrorRepository(session)
        model_repo = ModelRepository(session)

        created_count = 0
        updated_count = 0
        for sheet_name, df in sheets.items():
            logger.debug("Traitement de la feuille '%s' (%d ligne(s))", sheet_name, len(df))
            for index, row in df.iterrows():
                logger.debug("Traitement de la ligne %d de la feuille '%s'", index, sheet_name)
                code = row['Code']
                signification = row['Signification']
                client_behaviour = row['Conduite à tenir (client)']
                ass_intervention = row['Intervention SAV (technicien)']
                models = row['Modèles concernés']
                models_list = models.split(',') if pd.notna(models) else []
                for model_name in models_list:
                    model_name = model_name.strip()
                    logger.debug("Processing model '%s' for error code '%s'", model_name, code)
                    model = cls._get_or_create_model(model_repo, model_name)
                    if not model:
                        logger.warning("Modèle '%s' introuvable, code d'erreur '%s' ignoré", model_name, code)
                        continue
                    error = errors_repo.get_by_code_and_model(code, model.id)
                    if error:
                        new_values = {
                            "signification": signification,
                            "client_behaviour": client_behaviour,
                            "ass_intervention": ass_intervention,
                        }
                        changed = {k: v for k, v in new_values.items() if getattr(error, k) != v}
                        for field, value in changed.items():
                            setattr(error, field, value)
                        if changed:
                            updated_count += 1
                            logger.info("Code d'erreur '%s' mis à jour pour le modèle '%s'", code, model_name)
                        continue
                    errors_repo.create(
                        signification=signification,
                        client_behaviour=client_behaviour,
                        ass_intervention=ass_intervention,
                        code=code,
                        model_id=model.id,
                    )
                    created_count += 1
                session.commit()
                logger.info("Processed error code '%s' with models: %s", code, ", ".join(models_list))
        logger.info(
            "Référentiel des codes d'erreur traité : %d association(s) code/modèle créée(s), %d mise(s) à jour.",
            created_count, updated_count,
        )

    def _build_docling_entries(self, path: Path) -> list[_ChunkEntry]:
        """Convertit un document avec Docling et le découpe en chunks contextualisés.

        Args:
            path: Chemin du fichier source pris en charge par le convertisseur.

        Returns:
            list[_ChunkEntry]: Chunks nettoyés, avec leur section et leur page.
        """
        logger.debug("Conversion du document : %s", path)
        if path.suffix.lower() in {".html", ".htm"}:
            html_document = replace_summary_with_h3(path.read_text(encoding="utf-8"))
            source = DocumentStream(
                name=path.name,
                stream=BytesIO(html_document.encode("utf-8")),
            )
            doc = self.converter.convert(source=source).document
        else:
            doc = self.converter.convert(source=str(path)).document
        chunks = list(self.chunker.chunk(dl_doc=doc))

        entries = []
        for chunk in chunks:
            page = None
            if chunk.meta.doc_items:
                provs = chunk.meta.doc_items[0].prov
                if provs:
                    page = provs[0].page_no
            entries.append(_ChunkEntry(
                content=self._clean_chunk(chunk.text),
                embedding_text=self._clean_chunk(self.chunker.contextualize(chunk=chunk)),
                section=chunk.meta.headings[0] if chunk.meta.headings else None,
                page=page,
            ))
        return entries

    def _build_referentiel_entries(self, filepath: str) -> tuple[list[_ChunkEntry], str | None]:
        """Découpe le référentiel des codes d'erreur en un chunk par ligne de tableau.

        Chaque chunk reprend les en-têtes du tableau (``En-tête : valeur``). Le titre et la
        version de la feuille précèdent le texte encodé, pour situer la ligne. Les modèles et le
        code d'erreur de la ligne sont donnés explicitement plutôt que détectés dans le texte.

        Args:
            filepath: Chemin du fichier Excel du référentiel.

        Returns:
            tuple[list[_ChunkEntry], str | None]: Les chunks, et la date la plus récente trouvée dans
                les lignes de version des feuilles (``None`` si aucune).
        """
        sheets = pd.read_excel(filepath, sheet_name=None, header=REFERENTIEL_HEADER_ROW)
        intros = pd.read_excel(filepath, sheet_name=None, header=None, nrows=REFERENTIEL_HEADER_ROW)

        entries = []
        dates = []
        for sheet_name, df in sheets.items():
            intro_lines = [self._clean_chunk(str(value)) for value in intros[sheet_name].iloc[:, 0].dropna()]
            sheet_date = extract_date("\n".join(intro_lines))
            if sheet_date:
                dates.append(sheet_date)
            for _, row in df.iterrows():
                if pd.isna(row.get("Code")):
                    continue
                content = "\n".join(
                    f"{header} : {self._clean_chunk(str(value))}"
                    for header, value in row.items()
                    if pd.notna(value)
                )
                models = row.get("Modèles concernés")
                entries.append(_ChunkEntry(
                    content=content,
                    embedding_text="\n".join([*intro_lines, content]),
                    section=sheet_name,
                    page=None,
                    models=[m.strip() for m in str(models).split(",") if m.strip()] if pd.notna(models) else [],
                    error_codes=[str(row["Code"]).strip()],
                ))
        logger.info("Référentiel : %d ligne(s) transformée(s) en chunks.", len(entries))
        return entries, max(dates) if dates else None

    def process_document(self, doc_source: str, session: Session) -> str:
        """Convertit un document, indexe ses chunks et les relie aux modèles et erreurs détectés.

        L'empreinte du fichier est comparée à celle enregistrée en base. Un document inchangé
        est ignoré; un document modifié voit tous ses chunks remplacés et sa date recherchée de
        nouveau, dans une seule transaction (l'ancienne version est conservée en cas d'échec).
        Un fichier portant le nom ``referentiel_codes_erreur`` déclenche également l'import du
        référentiel Excel, et est découpé en un chunk par ligne de tableau plutôt que par Docling.
        Les modèles absents de la base mais cités dans le document y sont ajoutés.

        Args:
            doc_source: Chemin du fichier source pris en charge par le convertisseur.
            session: Session SQLAlchemy utilisée pour créer le document et ses chunks.

        Returns:
            str: ``"created"``, ``"updated"`` ou ``"unchanged"`` selon le traitement effectué.

        Raises:
            FileNotFoundError: Si le fichier source n'existe pas.
        """
        logger.info("Traitement du document demandé : %s", doc_source)
        path = Path(doc_source)
        filepath = str(path.resolve())
        title = path.stem
        content_hash = compute_file_hash(path)

        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)
        model_repo = ModelRepository(session)
        error_repo = ErrorRepository(session)

        document = doc_repo.get_by_filepath(filepath) or doc_repo.get_by_title(title)
        if document and document.content_hash == content_hash:
            logger.info("Document inchangé, ignoré : '%s'", title)
            return "unchanged"
        if document:
            logger.info("Document modifié, régénération des chunks : '%s'", title)

        if title.lower() == REFERENTIEL_TITLE:
            self._process_errors_file(session, filepath)
            entries, referentiel_date = self._build_referentiel_entries(filepath)
        else:
            entries = self._build_docling_entries(path)
            referentiel_date = None

        logger.debug("Encodage de %d chunks pour '%s'", len(entries), title)
        embedding_texts = [entry.embedding_text for entry in entries]
        embeddings = self.model.encode(
            embedding_texts,
            normalize_embeddings=True,
            batch_size=32,
        )

        self._register_new_models(embedding_texts, model_repo)

        # Tout ce qui suit est validé par un unique commit : en cas d'échec, l'ancienne version est conservée
        if document:
            status = "updated"
            session.query(Chunk).filter_by(document_id=document.id).delete(synchronize_session=False)
            document.filepath = filepath
            document.date = None
        else:
            status = "created"
            document = Document(title=title, filepath=filepath)
            session.add(document)
            session.flush()
            logger.info("Document créé : '%s' (%s)", title, filepath)
        document.content_hash = content_hash
        if referentiel_date:
            document.date = referentiel_date

        db_chunks = []
        for i, (entry, embedding) in enumerate(zip(entries, embeddings)):
            logger.debug("Chunk %d : %s", i, entry.content)
            content = entry.content
            if not document.date:
                document.date = extract_date(content)
                if document.date:
                    logger.info("Date du document '%s' : %s (chunk %d)", title, document.date, i)
                else:
                    logger.warning("Impossible d'extraire la date du document '%s' à partir du chunk %d", title, i)
            else:
                logger.debug("Date du document '%s' déjà définie : %s (chunk %d)", title, document.date, i)
            section = entry.section
            page = entry.page

            new_chunk = Chunk(
                document_id=document.id,
                chunk_index=i,
                content=content,
                embedding_text=entry.embedding_text,
                section=section,
                page=page,
                embedding=embedding.tolist(),
            )
            # Ajouté tout de suite pour éviter les avertissements d'autoflush sur les relations Model/ErrorCode
            session.add(new_chunk)

            if entry.models is not None:
                model_names = entry.models
            else:
                model_names = self._search_models_for_chunk(new_chunk.embedding_text, model_repo)

            matched_models = []
            seen_model_ids = set()
            for model_name in model_names:
                model = model_repo.get_by_name(model_name)
                if model and model.id not in seen_model_ids:
                    new_chunk.models.append(model)
                    matched_models.append(model)
                    seen_model_ids.add(model.id)

            if entry.error_codes is not None:
                error_codes = entry.error_codes
            else:
                error_codes = self._search_errors_for_chunk(new_chunk.embedding_text, error_repo)
            seen_codes = set()
            for code in error_codes:
                # chunk_error_code n'a plus de model_id : un même code ne peut être lié qu'une fois par chunk
                if code in seen_codes:
                    continue
                chunk_repo.link_error_code(new_chunk.id, code)
                seen_codes.add(code)

            db_chunks.append(new_chunk)

            logger.debug(
                "Ajout en BDD - Document '%s' Chunk %d : section='%s', page=%s, model(s)=%s, error_code(s)=%s",
                title, i, section, page, [m.name for m in matched_models], sorted(seen_codes),
            )

        chunk_repo.bulk_create(db_chunks)
        session.commit()
        logger.info("%d chunk(s) insérés pour le document '%s' (%s).", len(db_chunks), title, status)
        return status

    def _search_models_for_chunk(self, embedding_text: str, model_repo: ModelRepository) -> list[str]:
        """Repère les références de modèles connues dans le texte d'un chunk.

        Args:
            embedding_text: Texte contextualisé dans lequel rechercher les références.
            model_repo: Dépôt fournissant la liste des modèles connus.

        Returns:
            list[str]: Références détectées, dans leur ordre d'apparition.
        """
        codes = model_repo.get_all()
        code_names = [code.name for code in codes]
        logger.debug("Recherche de modèles parmi %d modèle(s) connu(s)", len(code_names))

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
        logger.debug("%d modèle(s) détecté(s) : %s", len(resultats), resultats)
        return resultats
    
    def _search_errors_for_chunk(self, embedding_text: str, error_repo: ErrorRepository) -> list[str]:
        """Repère les codes d'erreur connus dans le texte d'un chunk.

        Args:
            embedding_text: Texte contextualisé dans lequel rechercher les codes.
            error_repo: Dépôt fournissant les codes d'erreur connus.

        Returns:
            list[str]: Codes détectés, dans leur ordre d'apparition.
        """
        error_codes = error_repo.get_all_unique_codes()
        logger.debug("Recherche de codes d'erreur parmi %d code(s) connu(s)", len(error_codes))

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
        logger.debug("%d code(s) d'erreur détecté(s) : %s", len(resultats), resultats)
        return resultats
