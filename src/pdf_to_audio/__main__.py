"""CLI entry point for the pdf-to-audio pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from pdf_to_audio.config.settings import load_settings
from pdf_to_audio.pipeline.runner import run_file, run_folder
from pdf_to_audio.pipeline.steps import PreprocessingOptions, resolve_steps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-to-audio",
        description="Convert PDF files to audio via text extraction, LLM cleaning, and TTS synthesis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ("extract", "process", "tts", "run"):
        sub = subparsers.add_parser(cmd, help=f"Run the {cmd} step.")
        sub.add_argument("input", type=Path, help="Input file or folder.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()

    options = PreprocessingOptions(
        skip_footnotes=settings.skip_footnotes,
        skip_bibliography=settings.skip_bibliography,
        skip_parenthetical_citations=settings.skip_parenthetical_citations,
        skip_image_descriptions=settings.skip_image_descriptions,
    )

    steps = resolve_steps(args.command if args.command != "run" else None)

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if input_path.is_dir():
        run_folder(input_path, steps, options, settings)
    else:
        run_file(input_path, steps, options, settings)


if __name__ == "__main__":
    main()
