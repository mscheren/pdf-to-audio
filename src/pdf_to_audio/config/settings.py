"""Application settings loaded from defaults.local.json, runtime.local.json, and environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from pdf_to_audio.templates.template_loader import template_loader


@dataclass
class Settings:
    azure_deployment_name: str
    azure_api_version: str
    tts_voice: str
    chunk_token_limit: int
    skip_footnotes: bool
    skip_bibliography: bool
    skip_parenthetical_citations: bool
    skip_image_descriptions: bool
    pdf_base_dir: str
    texts_base_dir: str
    texts_processed_base_dir: str
    audio_base_dir: str
    tts_max_retries: int
    tts_retry_delay_seconds: int
    hash_block_min_length: int
    watermark_marker: str
    fffd_length_threshold: int


def load_settings() -> Settings:
    """Load settings from defaults.local.json and runtime.local.json, with env var overrides for infrastructure settings."""
    load_dotenv()
    defaults = template_loader.load_json_config("defaults.local")
    runtime = template_loader.load_json_config("runtime.local")
    return Settings(
        azure_deployment_name=str(os.environ.get("AZURE_DEPLOYMENT_NAME", defaults["azure_deployment_name"])),
        azure_api_version=str(os.environ.get("AZURE_OPENAI_API_VERSION", defaults["azure_api_version"])),
        tts_voice=str(os.environ.get("TTS_VOICE", defaults["tts_voice"])),
        chunk_token_limit=int(os.environ.get("CHUNK_TOKEN_LIMIT", str(defaults["chunk_token_limit"]))),
        skip_footnotes=bool(runtime["skip_footnotes"]),
        skip_bibliography=bool(runtime["skip_bibliography"]),
        skip_parenthetical_citations=bool(runtime["skip_parenthetical_citations"]),
        skip_image_descriptions=bool(runtime["skip_image_descriptions"]),
        pdf_base_dir=str(os.environ.get("PDF_BASE_DIR", defaults["pdf_base_dir"])),
        texts_base_dir=str(os.environ.get("TEXTS_BASE_DIR", defaults["texts_base_dir"])),
        texts_processed_base_dir=str(
            os.environ.get("TEXTS_PROCESSED_BASE_DIR", defaults["texts_processed_base_dir"])
        ),
        audio_base_dir=str(os.environ.get("AUDIO_BASE_DIR", defaults["audio_base_dir"])),
        tts_max_retries=int(os.environ.get("TTS_MAX_RETRIES", str(defaults["tts_max_retries"]))),
        tts_retry_delay_seconds=int(
            os.environ.get("TTS_RETRY_DELAY_SECONDS", str(defaults["tts_retry_delay_seconds"]))
        ),
        hash_block_min_length=int(
            os.environ.get("HASH_BLOCK_MIN_LENGTH", str(defaults["hash_block_min_length"]))
        ),
        watermark_marker=str(os.environ.get("WATERMARK_MARKER", defaults["watermark_marker"])),
        fffd_length_threshold=int(
            os.environ.get("FFFD_LENGTH_THRESHOLD", str(defaults["fffd_length_threshold"]))
        ),
    )
