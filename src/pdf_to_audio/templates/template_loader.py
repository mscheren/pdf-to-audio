"""Template file loader utilities."""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template


class TemplateLoader:
    """Load templates from the templates directory."""

    def __init__(self, base_path: Path | None = None):
        if base_path is None:
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.base_path)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load_prompt_template(self, name: str) -> Template:
        """Load a Jinja2 prompt template from prompts/{name}.txt."""
        return self.jinja_env.get_template(f"prompts/{name}.txt")

    def load_json_config(self, name: str) -> dict[str, Any]:
        """Load a JSON config file from config/{name}.json."""
        full_path = self.base_path / "config" / f"{name}.json"
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)


template_loader = TemplateLoader()
