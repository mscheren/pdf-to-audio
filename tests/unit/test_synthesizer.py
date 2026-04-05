"""Unit tests for the TTS synthesizer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf_to_audio.config.settings import Settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "azure_deployment_name": "exp-gpt-4.1",
        "azure_api_version": "2024-12-01-preview",
        "tts_voice": "de-DE-KatjaNeural",
        "chunk_token_limit": 4000,
        "llm_max_output_tokens": 32768,
        "skip_footnotes": False,
        "skip_bibliography": False,
        "pdf_base_dir": "pdf",
        "texts_base_dir": "texts",
        "texts_processed_base_dir": "texts_processed",
        "audio_base_dir": "audio",
        "tts_max_retries": 3,
        "tts_retry_delay_seconds": 5,
        "hash_block_min_length": 40,
        "watermark_marker": "Urheberrechtlich geschützt",
        "fffd_length_threshold": 100,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestSynthesize:
    def test_calls_communicate_with_correct_voice(self):
        mock_communicate = MagicMock()

        with patch("pdf_to_audio.tts.synthesizer.edge_tts.Communicate", return_value=mock_communicate) as mock_cls:
            from pdf_to_audio.tts.synthesizer import synthesize

            synthesize("hello world", Path("output.mp3"), "de-DE-KatjaNeural", _make_settings())

        mock_cls.assert_called_once_with("hello world", "de-DE-KatjaNeural")

    def test_calls_save_sync_with_output_path(self):
        mock_communicate = MagicMock()

        with patch("pdf_to_audio.tts.synthesizer.edge_tts.Communicate", return_value=mock_communicate):
            from pdf_to_audio.tts.synthesizer import synthesize

            synthesize("text", Path("/tmp/audio.mp3"), "de-DE-KatjaNeural", _make_settings())

        mock_communicate.save_sync.assert_called_once_with("/tmp/audio.mp3")

    def test_passes_text_to_communicate(self):
        mock_communicate = MagicMock()
        test_text = "This is the text to synthesize."

        with patch("pdf_to_audio.tts.synthesizer.edge_tts.Communicate", return_value=mock_communicate) as mock_cls:
            from pdf_to_audio.tts.synthesizer import synthesize

            synthesize(test_text, Path("out.mp3"), "en-US-AriaNeural", _make_settings())

        mock_cls.assert_called_once_with(test_text, "en-US-AriaNeural")
