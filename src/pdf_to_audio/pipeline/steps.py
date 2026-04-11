"""Pipeline step definitions and preprocessing options."""

from dataclasses import dataclass, field
from enum import Enum


class PipelineStep(Enum):
    EXTRACT = "extract"
    PROCESS = "process"
    TTS = "tts"


@dataclass
class PreprocessingOptions:
    skip_footnotes: bool = field(default=False)
    skip_bibliography: bool = field(default=False)
    skip_parenthetical_citations: bool = field(default=False)
    skip_image_descriptions: bool = field(default=False)


def resolve_steps(step_name: str | None) -> list[PipelineStep]:
    """Return the list of steps for a given subcommand name.

    ``None`` or ``"run"`` returns all steps in order.
    """
    if step_name is None or step_name == "run":
        return [PipelineStep.EXTRACT, PipelineStep.PROCESS, PipelineStep.TTS]
    return [PipelineStep(step_name)]
