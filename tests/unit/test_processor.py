"""Unit tests for the LLM processor module."""

from unittest.mock import MagicMock

from pdf_to_audio.config.settings import Settings
from pdf_to_audio.pipeline.steps import PreprocessingOptions


def _make_settings(**overrides) -> Settings:
    defaults = {
        "azure_deployment_name": "exp-gpt-4.1",
        "azure_api_version": "2024-12-01-preview",
        "tts_voice": "de-DE-KatjaNeural",
        "chunk_token_limit": 24000,
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


def _make_mock_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 200
    return response


class TestProcessText:
    def test_single_chunk_calls_api_once(self):
        from pdf_to_audio.llm.processor import process_text

        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response("cleaned output")

        result = process_text("short input text", PreprocessingOptions(), client, _make_settings())

        client.chat.completions.create.assert_called_once()
        assert result == "cleaned output"

    def test_multiple_chunks_each_call_api_independently(self):
        from pdf_to_audio.llm.chunker import count_tokens
        from pdf_to_audio.llm.processor import process_text

        para_a = "word " * 200
        para_b = "word " * 200
        text = para_a.strip() + "\n\n" + para_b.strip()
        tokens_per_para = count_tokens(para_a)
        settings = _make_settings(chunk_token_limit=tokens_per_para + 10)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_mock_response("chunk one output"),
            _make_mock_response("chunk two output"),
        ]

        result = process_text(text, PreprocessingOptions(), client, settings)

        assert client.chat.completions.create.call_count == 2
        # Each chunk uses the same system prompt — no cross-chunk context injected
        first_prompt = client.chat.completions.create.call_args_list[0].kwargs["messages"][0]["content"]
        second_prompt = client.chat.completions.create.call_args_list[1].kwargs["messages"][0]["content"]
        assert first_prompt == second_prompt

    def test_skip_footnotes_reflected_in_prompt(self):
        from pdf_to_audio.llm.processor import process_text

        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response("output")

        process_text("text", PreprocessingOptions(skip_footnotes=True), client, _make_settings())

        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "footnotes" in system_prompt.lower()

    def test_skip_bibliography_reflected_in_prompt(self):
        from pdf_to_audio.llm.processor import process_text

        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response("output")

        process_text("text", PreprocessingOptions(skip_bibliography=True), client, _make_settings())

        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "bibliography" in system_prompt.lower()

    def test_no_preprocessing_flags_omit_optional_rules(self):
        from pdf_to_audio.llm.processor import process_text

        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response("output")

        process_text("text", PreprocessingOptions(), client, _make_settings())

        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "footnotes" not in system_prompt.lower()
        assert "bibliography" not in system_prompt.lower()

    def test_multiple_chunks_joined_with_double_newline(self):
        from pdf_to_audio.llm.chunker import count_tokens
        from pdf_to_audio.llm.processor import process_text

        para_a = "word " * 200
        para_b = "word " * 200
        text = para_a.strip() + "\n\n" + para_b.strip()
        tokens_per_para = count_tokens(para_a)
        settings = _make_settings(chunk_token_limit=tokens_per_para + 10)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_mock_response("chunk one output"),
            _make_mock_response("chunk two output"),
        ]

        result = process_text(text, PreprocessingOptions(), client, settings)

        assert result == "chunk one output\n\nchunk two output"
