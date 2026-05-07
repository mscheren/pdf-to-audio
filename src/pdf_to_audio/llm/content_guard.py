"""Content sanitization to work around Azure OpenAI content filter restrictions."""

import logging

from better_profanity import profanity

logger = logging.getLogger(__name__)

profanity.load_censor_words()


def sanitize(text: str) -> str:
    """Replace profanity in text with censored placeholders.

    Intended as a fallback when a chunk triggers Azure OpenAI's content filter.
    Uses better-profanity's default word list to detect and replace flagged terms
    before retrying the API call.

    Args:
        text: Input text that may contain content-filtered terms.

    Returns:
        Text with flagged terms replaced by asterisks.
    """
    sanitized = profanity.censor(text)
    if sanitized != text:
        logger.debug("Content sanitization replaced flagged terms")
    return sanitized
