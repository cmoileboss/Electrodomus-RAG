"""Configuration centralisée du logging : sortie console (INFO+) et fichier (DEBUG+)."""

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "electrodomus.log"


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré pour écrire en console et dans le fichier de log.

    Args:
        name: Nom hiérarchique du logger, généralement ``__name__`` du module appelant.

    Returns:
        logging.Logger: Logger au niveau DEBUG, configuré une seule fois avec un
            handler console INFO+ et un handler fichier DEBUG+.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console : INFO et au-dessus
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # Fichier : DEBUG et au-dessus
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
