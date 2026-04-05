"""Text-to-speech synthesis using Edge TTS."""

import logging
import time
from pathlib import Path

import aiohttp
import edge_tts

from pdf_to_audio.config.settings import Settings

logger = logging.getLogger(__name__)


def synthesize(text: str, output_path: Path, voice: str, settings: Settings) -> None:
    """Synthesize text to an MP3 file using Edge TTS.

    Retries up to settings.tts_max_retries times on transient service errors (e.g. 503).

    Args:
        text: The text to synthesize.
        output_path: Destination path for the output MP3 file.
        voice: Edge TTS voice name (e.g. "de-DE-KatjaNeural").
        settings: Application settings (provides retry configuration).
    """
    logger.info("Synthesizing audio to %s with voice %s", output_path, voice)
    last_error: Exception | None = None
    for attempt in range(1, settings.tts_max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            communicate.save_sync(str(output_path))
            logger.info("Audio saved to %s", output_path)
            return
        except aiohttp.WSServerHandshakeError as exc:
            last_error = exc
            logger.warning(
                "TTS attempt %d/%d failed with %s: %s — retrying in %ds",
                attempt,
                settings.tts_max_retries,
                exc.status,
                exc.message,
                settings.tts_retry_delay_seconds,
            )
            time.sleep(settings.tts_retry_delay_seconds)
    raise RuntimeError(f"TTS synthesis failed after {settings.tts_max_retries} attempts") from last_error
