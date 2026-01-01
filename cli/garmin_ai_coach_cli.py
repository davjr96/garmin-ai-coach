#!/usr/bin/env python3

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from core.config import reload_config
from services.ai.ai_settings import ai_settings
from services.ai.langgraph.workflows.planning_workflow import (
    run_complete_analysis_and_planning,
)
from services.ai.utils.plan_storage import FilePlanStorage
from services.chatbot import AnalysisContextLoader, ChatbotService
from services.garmin import ExtractionConfig, TriathlonCoachDataExtractor
from services.outside.client import OutsideApiGraphQlClient

sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ConfigParser:

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        content = self.config_path.read_text(encoding="utf-8")

        if self.config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(content)
        elif self.config_path.suffix == ".json":
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")

    def get_athlete_info(self) -> tuple[str, str]:
        if not (email := self.config.get("athlete", {}).get("email")):
            raise ValueError("Athlete email is required in config file")

        return self.config.get("athlete", {}).get("name", "Athlete"), email

    def get_contexts(self) -> tuple[str, str]:
        return (
            self.config.get("context", {}).get("analysis", "").strip(),
            self.config.get("context", {}).get("planning", "").strip()
        )

    def get_extraction_config(self) -> dict[str, Any]:
        return {
            "activities_days": self.config.get("extraction", {}).get("activities_days", 7),
            "metrics_days": self.config.get("extraction", {}).get("metrics_days", 14),
            "ai_mode": self.config.get("extraction", {}).get("ai_mode", "development"),
            "enable_plotting": self.config.get("extraction", {}).get("enable_plotting", False),
            "hitl_enabled": self.config.get("extraction", {}).get("hitl_enabled", True),
            "skip_synthesis": self.config.get("extraction", {}).get("skip_synthesis", False),
        }

    def get_competitions(self) -> list[dict[str, Any]]:
        competitions = self.config.get("competitions", [])
        return [
            {
                "name": comp.get("name", ""),
                "date": comp.get("date", ""),
                "race_type": comp.get("race_type", ""),
                "priority": comp.get("priority", "B"),
                "target_time": comp.get("target_time", ""),
            }
            for comp in competitions
        ]

    def get_output_directory(self) -> Path:
        return Path(self.config.get("output", {}).get("directory", "./data"))

    def get_password(self) -> str:
        return (
            self.config.get("credentials", {}).get("password", "") or
            getpass.getpass("Enter Garmin Connect password: ")
        )


def fetch_outside_competitions_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    client = OutsideApiGraphQlClient()

    if isinstance(outside_cfg := config.get("outside"), dict) and any(
        isinstance(value, list) for value in outside_cfg.values()
    ):
        return client.get_competitions(outside_cfg)

    aggregate: list[dict[str, Any]] = []

    if isinstance(legacy_bikereg := config.get("bikereg", []), list) and legacy_bikereg:
        aggregate.extend(client.get_competitions(legacy_bikereg))

    if legacy_all := {
        key: entries
        for key in ("runreg", "trireg", "skireg")
        if isinstance(entries := config.get(key, []), list) and entries
    }:
        aggregate.extend(client.get_competitions(legacy_all))

    return aggregate


async def run_analysis_from_config(config_path: Path) -> None:
    config_parser = ConfigParser(config_path)
    athlete_name, email = config_parser.get_athlete_info()
    analysis_context, planning_context = config_parser.get_contexts()
    extraction_settings = config_parser.get_extraction_config()

    competitions = config_parser.get_competitions()
    outside_competitions = fetch_outside_competitions_from_config(config_parser.config)
    if outside_competitions:
        competitions.extend(outside_competitions)

    output_dir = config_parser.get_output_directory()

    logger.info(f"Starting analysis for {athlete_name}")
    logger.info(f"Output directory: {output_dir}")

    password = config_parser.get_password()

    os.environ["AI_MODE"] = extraction_settings.get("ai_mode", "development")
    
    # Reload config and settings to pick up the new AI_MODE
    reload_config()
    ai_settings.reload()
    
    logger.info(f"AI Mode: {os.environ['AI_MODE']}")


    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Extracting Garmin Connect data...")
        extractor = TriathlonCoachDataExtractor(email, password)

        extraction_config = ExtractionConfig(
            activities_range=extraction_settings["activities_days"],
            metrics_range=extraction_settings["metrics_days"],
            include_detailed_activities=True,
            include_metrics=True,
        )

        garmin_data = extractor.extract_data(extraction_config)
        logger.info("Data extraction completed")

        now = datetime.now()
        plotting_enabled = extraction_settings.get("enable_plotting", False)
        hitl_enabled = extraction_settings.get("hitl_enabled", True)
        skip_synthesis = extraction_settings.get("skip_synthesis", False)
        
        logger.info(f"Plotting enabled: {plotting_enabled}")
        logger.info(f"HITL enabled: {hitl_enabled}")
        logger.info(f"Skip synthesis: {skip_synthesis}")
        
        current_date = {"date": now.strftime("%Y-%m-%d"), "day_name": now.strftime("%A")}
        week_dates = [
            {"date": (now + timedelta(days=offset)).strftime("%Y-%m-%d"),
             "day_name": (now + timedelta(days=offset)).strftime("%A")}
            for offset in range(14)
        ]
        
        logger.info("Running AI analysis and planning...")
        
        result = await run_complete_analysis_and_planning(
            user_id="cli_user",
            athlete_name=athlete_name,
            garmin_data=asdict(garmin_data),
            analysis_context=analysis_context,
            planning_context=planning_context,
            competitions=competitions,
            current_date=current_date,
            week_dates=week_dates,
            plotting_enabled=plotting_enabled,
            hitl_enabled=hitl_enabled,
            skip_synthesis=skip_synthesis,
        )

        logger.info("Saving results...")

        files_generated: list[str] = []
        
        for filename, key in [
            ("analysis.html", "analysis_html"),
            ("planning.html", "planning_html"),
        ]:
            if content := result.get(key):
                if isinstance(content, dict):
                    content = content.get("content", "")
                (output_dir / filename).write_text(content, encoding="utf-8")
                files_generated.append(filename)
                logger.info(f"Saved: {output_dir}/{filename}")
        
        for filename, key in [
            ("metrics_expert.json", "metrics_outputs"),
            ("activity_expert.json", "activity_outputs"),
            ("physiology_expert.json", "physiology_outputs"),
        ]:
            if output := result.get(key):
                (output_dir / filename).write_text(
                    json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                files_generated.append(filename)
                logger.info(f"Saved: {output_dir}/{filename}")
        
        for filename, key in [
            ("season_plan.md", "season_plan"),
            ("weekly_plan.md", "weekly_plan"),
        ]:
            if plan_dict := result.get(key):
                output = plan_dict.get("output", plan_dict)
                if isinstance(output, str):
                    (output_dir / filename).write_text(output, encoding="utf-8")
                    files_generated.append(filename)
                    logger.info(f"Saved: {output_dir}/{filename}")
                    
                    # Also save to persistent storage
                    storage = FilePlanStorage()
                    plan_type = "season_plan" if key == "season_plan" else "weekly_plan"
                    # Use the user_id from the result or default to "cli_user"
                    user_id = result.get("user_id", "cli_user")
                    storage.save_plan(user_id, plan_type, output)

        cost_total = float(
            result.get("cost_summary", {}).get("total_cost_usd", 0.0) or
            result.get("execution_metadata", {}).get("total_cost_usd", 0.0) or
            sum(cost.get("total_cost", 0) for cost in result.get("costs", []))
        )
        total_tokens = int(
            result.get("cost_summary", {}).get("total_tokens", 0) or
            result.get("execution_metadata", {}).get("total_tokens", 0)
        )

        (output_dir / "summary.json").write_text(
            json.dumps({
                "athlete": athlete_name,
                "analysis_date": datetime.now().isoformat(),
                "competitions": competitions,
                "total_cost_usd": cost_total,
                "total_tokens": total_tokens,
                "execution_id": result.get("execution_id", ""),
                "trace_id": result.get("execution_metadata", {}).get("trace_id", ""),
                "root_run_id": result.get("execution_metadata", {}).get("root_run_id", ""),
                "files_generated": files_generated,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info("✅ Analysis completed successfully!")
        if outside_competitions:
            logger.info(f"✅  Added {len(outside_competitions)} Outside competitions from config")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"💰 Total cost: ${cost_total:.2f} ({total_tokens} tokens)")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


def list_chat_sessions(config_path: Path) -> None:
    """List all available chat sessions.

    Args:
        config_path: Path to configuration file (YAML or JSON)
    """
    from services.chatbot.conversation_manager import ConversationManager

    config_parser = ConfigParser(config_path)
    output_dir = config_parser.get_output_directory()

    sessions = ConversationManager.list_sessions(output_dir)

    if not sessions:
        logger.info("No chat sessions found.")
        return

    print("\n" + "=" * 80)
    print("Available Chat Sessions")
    print("=" * 80)

    for i, session in enumerate(sessions, 1):
        print(f"\n{i}. Session ID: {session['id']}")
        print(f"   Athlete: {session['athlete_name']}")
        print(f"   Started: {session['started_at']}")
        print(f"   Messages: {session['message_count']}")

    print("\n" + "=" * 80)
    print("\nTo resume a session, use:")
    print("  pixi run coach-chat my_config.yaml --session <session_id>")
    print("\nExample:")
    if sessions:
        print(f"  pixi run coach-chat my_config.yaml --session {sessions[0]['id']}\n")


async def run_chatbot_from_config(
    config_path: Path, resume: bool = False, session_id: str | None = None
) -> None:
    """Run interactive chatbot after analysis is complete.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        resume: If True, resume the most recent conversation
        session_id: Specific session ID to resume
    """
    config_parser = ConfigParser(config_path)
    athlete_name, email = config_parser.get_athlete_info()
    output_dir = config_parser.get_output_directory()
    extraction_settings = config_parser.get_extraction_config()

    # Set AI mode for chatbot
    os.environ["AI_MODE"] = extraction_settings.get("ai_mode", "development")
    reload_config()
    ai_settings.reload()

    logger.info(f"🤖 Starting chatbot for {athlete_name}...")

    # Validate analysis outputs exist
    loader = AnalysisContextLoader(output_dir, user_id="cli_user")

    if not loader.validate_outputs_exist():
        logger.error("❌ No analysis outputs found. Run analysis first with --config")
        logger.info("Required files: summary.json, season_plan.md, weekly_plan.md")
        logger.info(f"Looking in: {output_dir}")
        sys.exit(1)

    # Load context
    logger.info(f"Loading analysis from {output_dir}...")
    try:
        context = loader.load_context()
    except Exception as e:
        logger.error(f"❌ Failed to load analysis context: {e}")
        sys.exit(1)

    # Initialize and run chatbot
    chatbot = ChatbotService(
        context=context,
        user_id="cli_user",
        output_dir=output_dir,
        resume=resume,
        session_id=session_id,
    )

    await chatbot.run_interactive_loop()


def create_config_template(output_path: Path) -> None:
    template_path = Path(__file__).parent / "coach_config_template.yaml"

    if template_path.exists():
        output_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"✅ Config template created: {output_path}")
        logger.info("Edit this file with your settings and run analysis with --config")
    else:
        logger.error("❌ Template file not found")


def main():
    parser = argparse.ArgumentParser(
        description="Garmin AI Coach CLI - AI Triathlon Coach",
        epilog="Example: python garmin_ai_coach_cli.py --config my_config.yaml",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to configuration file (YAML or JSON)")
    group.add_argument("--init-config", type=Path, help="Create a configuration template file")
    group.add_argument(
        "--chat", type=Path, help="Start chatbot with existing analysis (YAML or JSON config)"
    )
    group.add_argument(
        "--list-sessions",
        type=Path,
        help="List all available chat sessions (YAML or JSON config)",
    )

    parser.add_argument("--output-dir", type=Path, help="Override output directory from config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent conversation (use with --chat)",
    )
    parser.add_argument(
        "--session", type=str, help="Resume specific session by ID (use with --chat)"
    )

    args = parser.parse_args()

    if args.init_config:
        create_config_template(args.init_config)
        return

    if args.list_sessions:
        list_chat_sessions(args.list_sessions)
        return

    if args.chat:
        try:
            asyncio.run(
                run_chatbot_from_config(
                    args.chat, resume=args.resume, session_id=args.session
                )
            )
        except KeyboardInterrupt:
            logger.info("\n❌ Chat session ended by user")
        except Exception as e:
            logger.error(f"❌ Chatbot failed: {e}")
            sys.exit(1)
        return

    if args.config:
        try:
            asyncio.run(run_analysis_from_config(args.config))
        except KeyboardInterrupt:
            logger.info("❌ Analysis cancelled by user")
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
