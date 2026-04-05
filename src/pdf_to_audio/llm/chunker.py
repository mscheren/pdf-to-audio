"""Token-aware text chunking utilities."""

import logging

import tiktoken

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        encoding_name: Tiktoken encoding name.

    Returns:
        Token count.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_into_chunks(text: str, max_tokens: int, encoding_name: str = "o200k_base") -> list[str]:
    """Split text into chunks at paragraph boundaries, keeping each chunk within max_tokens.

    Expects paragraph blocks separated by double newlines, as produced by the PDF extractor.
    Uses greedy accumulation: paragraphs are added to the current chunk until the token
    limit would be exceeded, then a new chunk is started. Single paragraphs that exceed
    max_tokens are placed in their own chunk without splitting.

    Args:
        text: The input text to split.
        max_tokens: Maximum token count per chunk.
        encoding_name: Tiktoken encoding name.

    Returns:
        List of text chunks, each ending at a paragraph boundary.
    """
    paragraphs = text.split("\n\n")
    encoding = tiktoken.get_encoding(encoding_name)

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = len(encoding.encode(paragraph))
        if current_paragraphs and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [paragraph]
            current_tokens = para_tokens
        else:
            current_paragraphs.append(paragraph)
            current_tokens += para_tokens

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    logger.debug("Split text into %d chunks (max %d tokens each)", len(chunks), max_tokens)
    return chunks


def extract_last_paragraph(text: str) -> str | None:
    """Extract the last non-empty paragraph block from a text string.

    Args:
        text: The text to extract the last paragraph from.

    Returns:
        The last non-empty paragraph block, or None if text is empty.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return None
    return paragraphs[-1]
