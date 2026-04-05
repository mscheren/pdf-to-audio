"""Unit tests for the text chunker module."""

from pdf_to_audio.llm.chunker import count_tokens, extract_last_paragraph, split_into_chunks


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        count = count_tokens("hello")
        assert count > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        longer = count_tokens("hello world how are you doing today")
        assert longer > short


class TestSplitIntoChunks:
    def test_single_chunk_when_below_limit(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        chunks = split_into_chunks(text, max_tokens=10000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_paragraph_boundary(self):
        para_a = "word " * 100
        para_b = "word " * 100
        text = para_a.strip() + "\n\n" + para_b.strip()
        tokens_a = count_tokens(para_a)
        chunks = split_into_chunks(text, max_tokens=tokens_a + 10)
        assert len(chunks) == 2
        assert para_a.strip() in chunks[0]
        assert para_b.strip() in chunks[1]

    def test_single_oversized_paragraph_becomes_own_chunk(self):
        large_para = "word " * 500
        text = large_para.strip()
        chunks = split_into_chunks(text, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_greedy_accumulation(self):
        text = "para one\n\npara two\n\npara three"
        chunks = split_into_chunks(text, max_tokens=10000)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = split_into_chunks("", max_tokens=1000)
        assert chunks == [""]


class TestExtractLastParagraph:
    def test_returns_last_paragraph(self):
        text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
        assert extract_last_paragraph(text) == "third paragraph"

    def test_ignores_trailing_whitespace(self):
        text = "first\n\nsecond\n\n   \n\n"
        assert extract_last_paragraph(text) == "second"

    def test_single_paragraph(self):
        text = "only paragraph"
        assert extract_last_paragraph(text) == "only paragraph"

    def test_empty_string_returns_none(self):
        assert extract_last_paragraph("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_last_paragraph("   \n\n   ") is None
