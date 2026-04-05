"""PDF text extraction using PyMuPDF."""

import logging
import re
from pathlib import Path

import pymupdf

from pdf_to_audio.config.settings import Settings

logger = logging.getLogger(__name__)


def _is_artifact(text: str, hash_block_re: re.Pattern[str], settings: Settings) -> bool:
    """Return True if a block consists entirely of machine-generated artifact text.

    Filters:
    - DRM hash strings: long base64-like tokens with no spaces
    - Copyright watermark lines that repeat on every page
    - Short blocks containing Unicode replacement characters (U+FFFD): garbled running
      headers from PDF symbol fonts with no unicode mapping. Long blocks with an isolated
      U+FFFD are valid body text where a single diacritic had no font encoding; those are
      passed through rather than discarded.
    """
    stripped = text.strip()
    if hash_block_re.match(stripped):
        return True
    if settings.watermark_marker in stripped:
        return True
    if "\ufffd" in stripped and len(stripped) < settings.fffd_length_threshold:
        return True
    return False


def extract_text(pdf_path: Path, settings: Settings) -> str:
    """Extract text from a PDF file as a sequence of paragraph blocks.

    Uses PyMuPDF's blocks extraction mode to group text into logical paragraphs,
    separated by double newlines. The TEXT_DEHYPHENATE flag rejoins words split
    across lines with a hyphen. Machine-generated artifact blocks (DRM hashes,
    watermarks) are removed before the text is returned.

    Args:
        pdf_path: Path to the PDF file.
        settings: Application settings (provides artifact filter configuration).

    Returns:
        Extracted text with paragraph blocks separated by double newlines.
    """
    logger.info("Extracting text from %s", pdf_path)
    hash_block_re = re.compile(rf'^[A-Za-z0-9+/=]{{{settings.hash_block_min_length},}}$')
    all_blocks: list[str] = []
    artifact_count = 0

    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            blocks = page.get_text("blocks", flags=pymupdf.TEXT_DEHYPHENATE)
            for block in blocks:
                if block[6] != 0:
                    continue
                text = block[4].strip()
                if not text:
                    continue
                if _is_artifact(text, hash_block_re, settings):
                    artifact_count += 1
                    continue
                all_blocks.append(text)

    result = "\n\n".join(all_blocks)
    logger.info(
        "Extracted %d blocks (%d characters) from %s, removed %d artifact blocks",
        len(all_blocks),
        len(result),
        pdf_path,
        artifact_count,
    )
    return result
