"""LLM-based text processing for PDF content cleaning."""

import logging

from openai import APIError, BadRequestError, OpenAI

from pdf_to_audio.config.settings import Settings
from pdf_to_audio.llm.chunker import split_into_chunks
from pdf_to_audio.llm.content_guard import sanitize
from pdf_to_audio.llm.exceptions import LLMError
from pdf_to_audio.pipeline.steps import PreprocessingOptions
from pdf_to_audio.templates.template_loader import template_loader

logger = logging.getLogger(__name__)


def process_text(text: str, options: PreprocessingOptions, client: OpenAI, settings: Settings) -> str:
    """Clean and reformat extracted PDF text using an LLM with chunked processing.

    Splits the input text into chunks at paragraph boundaries and processes each
    independently. Chunks are processed without cross-chunk continuation context,
    which was found to cause the model to over-skip content.

    Args:
        text: Raw extracted text from the PDF.
        options: Preprocessing flags.
        client: Configured LLM client.
        settings: Application settings.

    Returns:
        Cleaned text with all chunks joined by double newlines.
    """
    prompt_template = template_loader.load_prompt_template("clean_chunk")
    chunks = split_into_chunks(text, settings.chunk_token_limit)
    logger.info("Processing %d chunk(s) through LLM", len(chunks))

    system_prompt = prompt_template.render(
        skip_footnotes=options.skip_footnotes,
        skip_bibliography=options.skip_bibliography,
        skip_parenthetical_citations=options.skip_parenthetical_citations,
        skip_image_descriptions=options.skip_image_descriptions,
    )

    cleaned_chunks: list[str] = []

    for index, chunk in enumerate(chunks):
        logger.debug("Processing chunk %d/%d", index + 1, len(chunks))

        try:
            response = client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=0,
            )
        except BadRequestError as exc:
            if exc.code != "content_filter":
                raise LLMError(f"LLM request failed for chunk {index + 1}/{len(chunks)}") from exc
            logger.warning(
                "Chunk %d/%d triggered content filter, retrying with sanitized text",
                index + 1,
                len(chunks),
            )
            try:
                response = client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sanitize(chunk)},
                    ],
                    temperature=0,
                )
            except APIError as retry_exc:
                raise LLMError(
                    f"LLM request failed for chunk {index + 1}/{len(chunks)} after content filter retry"
                ) from retry_exc
        except APIError as exc:
            raise LLMError(f"LLM request failed for chunk {index + 1}/{len(chunks)}") from exc

        cleaned = str(response.choices[0].message.content)
        chunk_start = chunk[:50].replace("\n", " ")
        cleaned_start = cleaned[:50].replace("\n", " ")
        usage = response.usage
        if usage:
            logger.info(
                "Chunk %d/%d: %d prompt tokens, %d completion tokens",
                index + 1,
                len(chunks),
                usage.prompt_tokens,
                usage.completion_tokens,
            )
            logger.info(
                "Beginning of chunk: '%s', cleaned to: '%s'",
                chunk_start,
                cleaned_start,
            )

        cleaned_chunks.append(cleaned)

    return "\n\n".join(cleaned_chunks)
