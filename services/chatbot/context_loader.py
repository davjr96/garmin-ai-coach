"""Load completed analysis outputs for chatbot context."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnalysisContextLoader:
    """Loads analysis outputs from disk for chatbot context."""

    def __init__(self, output_dir: Path, user_id: str):
        """Initialize context loader.

        Args:
            output_dir: Directory containing analysis outputs
            user_id: User identifier for loading plans
        """
        self.output_dir = Path(output_dir)
        self.user_id = user_id

    def validate_outputs_exist(self) -> bool:
        """Check that required analysis files exist.

        Returns:
            True if all required files exist, False otherwise
        """
        required_files = ["summary.json", "season_plan.md", "weekly_plan.md"]

        missing_files = []
        for filename in required_files:
            file_path = self.output_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False

        return True

    def load_context(self) -> dict[str, Any]:
        """Load all analysis outputs into memory.

        Returns:
            Dictionary containing all loaded context data

        Raises:
            FileNotFoundError: If required files are missing
            json.JSONDecodeError: If JSON files are malformed
        """
        if not self.validate_outputs_exist():
            raise FileNotFoundError(
                f"Required analysis files missing in {self.output_dir}. "
                "Run analysis first with --config"
            )

        context: dict[str, Any] = {}

        # Load summary.json (execution metadata, athlete info, competitions)
        context["summary"] = self._load_json("summary.json")

        # Load expert outputs (optional - may not exist in all analyses)
        for expert_file in [
            "metrics_expert.json",
            "activity_expert.json",
            "physiology_expert.json",
        ]:
            try:
                context[expert_file.replace(".json", "")] = self._load_json(expert_file)
            except FileNotFoundError:
                logger.warning(f"Expert output not found: {expert_file}")
                context[expert_file.replace(".json", "")] = {}

        # Load plan markdown files
        context["season_plan"] = self._load_text("season_plan.md")
        context["weekly_plan"] = self._load_text("weekly_plan.md")

        # Load HTML files (optional - for reference)
        for html_file in ["analysis.html", "planning.html"]:
            try:
                context[html_file.replace(".html", "_html")] = self._load_text(html_file)
            except FileNotFoundError:
                logger.warning(f"HTML output not found: {html_file}")
                context[html_file.replace(".html", "_html")] = ""

        # Extract key metadata for easy access
        summary = context["summary"]
        context["athlete_name"] = summary.get("athlete", "Unknown")
        context["analysis_date"] = summary.get("analysis_date", "Unknown")
        context["competitions"] = summary.get("competitions", [])

        logger.info(
            f"Loaded context for {context['athlete_name']} (analysis: {context['analysis_date']})"
        )

        return context

    def _load_json(self, filename: str) -> dict[str, Any]:
        """Load and parse JSON file.

        Args:
            filename: Name of JSON file in output directory

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is malformed
        """
        file_path = self.output_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {filename}: {e}")
            raise

    def _load_text(self, filename: str) -> str:
        """Load text file.

        Args:
            filename: Name of text file in output directory

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.output_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return file_path.read_text(encoding="utf-8")
