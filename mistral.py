import os
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from mistralai.search.toolkit.ingestion import MistralEmbedder
from mistralai.search.toolkit.ingestion.extractors import MistralOCRExtractor
from mistralai.search.toolkit.ingestion.pipelines import Pipeline
from mistralai.search.toolkit.ingestion.loaders import FilesystemFileLoader
from mistralai.search.toolkit.ingestion.text_splitters import (
    MarkdownTextSplitter,
    MarkdownTextSplitterConfig,
)

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DOC_FOLDER = Path("Documentation_Electrodomus")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".xlsx"}
MAX_RETRIES = 2
DELAY_BETWEEN_UPLOADS = 2  # secondes, pour rester sous la limite de traitement de documents

SYSTEM_PROMPT = """\
Tu es un assistant technique expert pour la marque Electrodomus, spécialisé dans le dépannage,
l'installation et l'entretien des appareils électroménagers (fours, lave-linge, lave-vaisselle).

Règles absolues :
- Réponds UNIQUEMENT à partir des extraits de documentation fournis dans le CONTEXTE.
- Si la réponse ne figure pas dans le contexte, réponds exactement :
  "Je n'ai pas trouvé cette information dans la documentation Electrodomus."
- Ne complète jamais avec tes connaissances générales.
- Cite TOUJOURS la source (section / page) entre parenthèses après chaque information.
- Si plusieurs extraits se contredisent, signale la contradiction.
- Réponds en français, de façon claire, structurée et concise.
"""

client = Mistral(api_key=MISTRAL_API_KEY)

embedder = MistralEmbedder(client=client)

pipeline = Pipeline(
    loader=FilesystemFileLoader(),
    extractor=MistralOCRExtractor(client=client),
    text_splitter=MarkdownTextSplitter(
        MarkdownTextSplitterConfig(chunk_size=4096, chunk_overlap=50)
    ),
    embedder=embedder,
    stores=vector_store,
)