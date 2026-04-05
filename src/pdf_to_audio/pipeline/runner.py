"""Pipeline orchestration: file and folder execution modes."""

import logging
from pathlib import Path

from pdf_to_audio.config.settings import Settings
from pdf_to_audio.extraction.pdf_extractor import extract_text
from pdf_to_audio.llm.client import build_client
from pdf_to_audio.llm.processor import process_text
from pdf_to_audio.pipeline.steps import PipelineStep, PreprocessingOptions
from pdf_to_audio.tts.synthesizer import synthesize

logger = logging.getLogger(__name__)


def derive_output_path(input_path: Path, input_base: Path, output_base: Path, suffix: str) -> Path:
    """Compute a mirrored output path preserving subdirectory structure.

    Given ``pdf/folder/file.pdf`` with ``input_base=pdf`` and
    ``output_base=texts``, returns ``texts/folder/file.txt``.

    Args:
        input_path: Absolute or relative path to the input file.
        input_base: Base directory of the input tree.
        output_base: Base directory of the output tree.
        suffix: File extension for the output file (e.g. ".txt").

    Returns:
        Path to the output file.
    """
    relative = input_path.relative_to(input_base)
    return output_base / relative.with_suffix(suffix)


def run_file(
    input_path: Path,
    steps: list[PipelineStep],
    options: PreprocessingOptions,
    settings: Settings,
) -> None:
    """Execute the requested pipeline steps for a single input file.

    The input file type determines which step it feeds into:
    - EXTRACT expects a .pdf in the pdf base directory
    - PROCESS expects a .txt in the texts base directory
    - TTS expects a .txt in the texts_processed base directory

    When multiple steps are requested, the output of each step is fed into the next.
    """
    pdf_base = Path(settings.pdf_base_dir)
    texts_base = Path(settings.texts_base_dir)
    texts_processed_base = Path(settings.texts_processed_base_dir)
    audio_base = Path(settings.audio_base_dir)

    if PipelineStep.EXTRACT in steps:
        text_path = derive_output_path(input_path, pdf_base, texts_base, ".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        raw_text = extract_text(input_path, settings)
        text_path.write_text(raw_text, encoding="utf-8")
        logger.info("Extracted text written to %s", text_path)
        current_text_path = text_path
    else:
        current_text_path = input_path

    if PipelineStep.PROCESS in steps:
        processed_path = derive_output_path(current_text_path, texts_base, texts_processed_base, ".txt")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        raw_text = current_text_path.read_text(encoding="utf-8")
        client = build_client(settings)
        cleaned_text = process_text(raw_text, options, client, settings)
        processed_path.write_text(cleaned_text, encoding="utf-8")
        logger.info("Processed text written to %s", processed_path)
        current_text_path = processed_path
    elif PipelineStep.TTS in steps and PipelineStep.EXTRACT not in steps:
        current_text_path = input_path

    if PipelineStep.TTS in steps:
        if PipelineStep.PROCESS in steps:
            tts_input = current_text_path
            audio_path = derive_output_path(tts_input, texts_processed_base, audio_base, ".mp3")
        else:
            tts_input = input_path
            audio_path = derive_output_path(tts_input, texts_processed_base, audio_base, ".mp3")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tts_text = tts_input.read_text(encoding="utf-8")
        synthesize(tts_text, audio_path, settings.tts_voice, settings)
        logger.info("Audio written to %s", audio_path)


def run_folder(
    folder_path: Path,
    steps: list[PipelineStep],
    options: PreprocessingOptions,
    settings: Settings,
) -> None:
    """Execute the pipeline steps for all relevant files in a folder.

    File discovery is driven by the first step:
    - EXTRACT: discovers .pdf files under folder_path
    - PROCESS (without EXTRACT): discovers .txt files under texts base dir
    - TTS (without EXTRACT or PROCESS): discovers .txt files under texts_processed base dir
    """
    if PipelineStep.EXTRACT in steps:
        files = list(folder_path.rglob("*.pdf"))
    elif PipelineStep.PROCESS in steps:
        files = list(folder_path.rglob("*.txt"))
    else:
        files = list(folder_path.rglob("*.txt"))

    if not files:
        logger.warning("No matching files found in %s", folder_path)
        return

    logger.info("Processing %d file(s) in %s", len(files), folder_path)
    for file_path in files:
        logger.info("Processing %s", file_path)
        run_file(file_path, steps, options, settings)
