"""Unit tests for the PDF extractor module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf_to_audio.config.settings import Settings


def _make_block(text: str, block_type: int = 0) -> tuple:
    """Create a mock PyMuPDF block tuple (x0, y0, x1, y1, text, block_no, block_type)."""
    return (0, 0, 100, 100, text, 0, block_type)


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


class TestExtractText:
    def test_joins_blocks_with_double_newline(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("first block"),
            _make_block("second block"),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "first block\n\nsecond block"

    def test_joins_blocks_across_pages_with_double_newline(self):
        mock_page_a = MagicMock()
        mock_page_a.get_text.return_value = [_make_block("page one block")]
        mock_page_b = MagicMock()
        mock_page_b.get_text.return_value = [_make_block("page two block")]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page_a, mock_page_b]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "page one block\n\npage two block"

    def test_skips_image_blocks(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("text block", block_type=0),
            _make_block("image block", block_type=1),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "text block"
        assert "image block" not in result

    def test_skips_empty_blocks(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("content"),
            _make_block("   "),
            _make_block("more content"),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "content\n\nmore content"

    def test_removes_hash_artifact_blocks(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("real content"),
            _make_block("7ATM3/6P9G/EhdMjud4Z6lhuhQs0OZXuTXv6ZWvQxvCZP4HB8qVy0oBlEWCdH2Tk"),
            _make_block("more real content"),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "real content\n\nmore real content"

    def test_removes_short_blocks_with_unicode_replacement_characters(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("real content"),
            _make_block("NAGEL Kulturkontakt \n3\ufffd"),
            _make_block("\ufffd\nEinführung Geschichtswissenschaft"),
            _make_block("more real content"),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "real content\n\nmore real content"

    def test_retains_long_blocks_with_isolated_unicode_replacement_character(self):
        # A 1400+ char body paragraph with a single U+FFFD representing an unmappable
        # diacritic (e.g. "hist\uFFFDria") must not be discarded.
        long_paragraph = (
            "Unser Wort historisch ist selbst der historische Zeuge einer kulturellen Tradition. "
            "Es stammt vom griechischen histor\ufffda, womit jede auf sinnlicher Erfahrung "
            "beruhende Kunde gemeint war. " + "weiterer Text. " * 60
        )
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block(long_paragraph),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert long_paragraph.strip() in result

    def test_removes_watermark_blocks(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            _make_block("real content"),
            _make_block("Urheberrechtlich geschützt. Persönliche Kopie für Matrikelnummer 1263463"),
            _make_block("more real content"),
        ]
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == "real content\n\nmore real content"

    def test_empty_document(self):
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=iter([]))
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("pdf_to_audio.extraction.pdf_extractor.pymupdf.open", return_value=mock_doc):
            from pdf_to_audio.extraction.pdf_extractor import extract_text

            result = extract_text(Path("dummy.pdf"), _make_settings())

        assert result == ""
