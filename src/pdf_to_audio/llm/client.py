"""Azure OpenAI client factory."""

from openai import AzureOpenAI

from pdf_to_audio.config.settings import Settings


def build_client(settings: Settings) -> AzureOpenAI:
    """Build an AzureOpenAI client from settings.

    Credentials are read from the environment (AZURE_OPENAI_ENDPOINT and
    AZURE_OPENAI_API_KEY) by the SDK automatically.
    """
    return AzureOpenAI(api_version=settings.azure_api_version)
