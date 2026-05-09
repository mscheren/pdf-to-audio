"""LLM client factory."""

import os

from openai import AzureOpenAI, OpenAI

from pdf_to_audio.config.settings import Settings

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_client(settings: Settings) -> OpenAI:
    """Build an LLM client from settings.

    For azure_openai: credentials are read from the environment (AZURE_OPENAI_ENDPOINT
    and AZURE_OPENAI_API_KEY) by the SDK automatically.
    For google_gemini: GEMINI_API_KEY must be set in the environment.
    """
    if settings.llm_provider == "azure_openai":
        return AzureOpenAI(api_version=settings.api_version)
    if settings.llm_provider == "google_gemini":
        return OpenAI(base_url=_GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}. Expected 'azure_openai' or 'google_gemini'.")
