# pdf-to-audio

Converts PDF files to MP3 audio via three steps:

1. **Extract** — pulls text from a PDF using PyMuPDF, pages separated by form feeds
2. **Process** — cleans the extracted text using Azure OpenAI GPT-4.1 (chunked to avoid truncation)
3. **Synthesize** — converts the cleaned text to speech using Edge TTS

## Setup

```bash
cp .env.example .env
# Fill in AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env
uv sync
```

## Usage

Input files can be a single file or a folder. When a folder is given, all matching files are processed recursively.

Output mirrors the input directory structure across the configured base directories:

```text
pdf/folder/file.pdf
  -> texts/folder/file.txt
  -> texts_processed/folder/file.txt
  -> audio/folder/file.mp3
```

### Run the full pipeline

```bash
uv run pdf-to-audio run pdf/folder/file.pdf
uv run pdf-to-audio runpdf/folder/
```

**Important:** Only works with relative path.

### Run individual steps

```bash
uv run pdf-to-audio extract pdf/folder/file.pdf
uv run pdf-to-audio process texts/folder/file.txt
uv run pdf-to-audi tts texts_processed/folder/file.txt
```

## Configuration

Preprocessing options are set in `src/pdf_to_audio/templates/config/runtime.local.json`:

| Key | Default | Effect |
| --- | --- | --- |
| `skip_footnotes` | `true` | Remove footnote markers and footnote text |
| `skip_bibliography` | `true` | Remove the bibliography/references section |
| `skip_parenthetical_citations` | `true` | Remove parenthetical citations |

Infrastructure settings are overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_DEPLOYMENT_NAME` | `exp-gpt-4.1` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |
| `TTS_VOICE` | `de-DE-KatjaNeural` | Edge TTS voice |
| `CHUNK_TOKEN_LIMIT` | `4000` | Max input tokens per LLM chunk |
| `PDF_BASE_DIR` | `pdf` | Input PDF base directory |
| `TEXTS_BASE_DIR` | `texts` | Extracted text output directory |
| `TEXTS_PROCESSED_BASE_DIR` | `texts_processed` | Cleaned text output directory |
| `AUDIO_BASE_DIR` | `audio` | Audio output directory |

## Tests

```bash
uv run pytest tests/unit/
```
