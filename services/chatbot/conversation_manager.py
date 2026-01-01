"""Manage chatbot conversation history and context."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and builds context-aware prompts."""

    def __init__(
        self,
        user_id: str,
        output_dir: Path,
        context: dict[str, Any],
        resume: bool = False,
        session_id: str | None = None,
    ):
        """Initialize conversation manager.

        Args:
            user_id: User identifier
            output_dir: Directory for storing conversation history
            context: Analysis context loaded by AnalysisContextLoader
            resume: If True, attempt to resume the most recent conversation
            session_id: Specific session ID to resume (overrides resume=True)
        """
        self.user_id = user_id
        self.output_dir = Path(output_dir)
        self.context = context
        self.conversation_history: list[dict[str, str]] = []

        if session_id:
            # Resume specific session by ID
            self._resume_session_by_id(session_id)
        elif resume:
            # Resume most recent session
            self._resume_last_session()
        else:
            # Start new session
            self.session_id = self._generate_session_id()
            self.storage_path = self._get_storage_path()

    def _generate_session_id(self) -> str:
        """Generate unique session ID with timestamp."""
        return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_storage_path(self) -> Path:
        """Get path for storing conversation history."""
        chat_dir = self.output_dir / "chat_history"
        chat_dir.mkdir(parents=True, exist_ok=True)
        return chat_dir / f"{self.session_id}.json"

    def _resume_last_session(self) -> None:
        """Resume the most recent conversation session."""
        chat_dir = self.output_dir / "chat_history"
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Find all chat history files
        chat_files = sorted(chat_dir.glob("chat_*.json"), reverse=True)

        if not chat_files:
            logger.info("No previous conversations found, starting new session")
            self.session_id = self._generate_session_id()
            self.storage_path = self._get_storage_path()
            return

        # Load the most recent session
        latest_file = chat_files[0]
        try:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
            self.session_id = data.get("session_id", self._generate_session_id())
            self.conversation_history = data.get("messages", [])
            self.storage_path = latest_file

            logger.info(
                f"Resumed conversation: {self.session_id} "
                f"({len(self.conversation_history)} messages)"
            )
        except Exception as e:
            logger.error(f"Failed to resume session from {latest_file}: {e}")
            self.session_id = self._generate_session_id()
            self.storage_path = self._get_storage_path()

    def _resume_session_by_id(self, session_id: str) -> None:
        """Resume a specific conversation session by ID.

        Args:
            session_id: Session ID to resume (e.g., 'chat_20260101_080000')
        """
        chat_dir = self.output_dir / "chat_history"
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Clean up session_id if it doesn't have .json extension
        if not session_id.endswith(".json"):
            session_file = chat_dir / f"{session_id}.json"
        else:
            session_file = chat_dir / session_id

        if not session_file.exists():
            logger.error(f"Session file not found: {session_file}")
            logger.info("Starting new session instead")
            self.session_id = self._generate_session_id()
            self.storage_path = self._get_storage_path()
            return

        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            self.session_id = data.get("session_id", self._generate_session_id())
            self.conversation_history = data.get("messages", [])
            self.storage_path = session_file

            logger.info(
                f"Resumed conversation: {self.session_id} "
                f"({len(self.conversation_history)} messages)"
            )
        except Exception as e:
            logger.error(f"Failed to resume session from {session_file}: {e}")
            self.session_id = self._generate_session_id()
            self.storage_path = self._get_storage_path()

    @staticmethod
    def list_sessions(output_dir: Path) -> list[dict[str, Any]]:
        """List all available chat sessions.

        Args:
            output_dir: Directory containing chat history

        Returns:
            List of session info dicts with id, started_at, message_count, athlete_name
        """
        chat_dir = output_dir / "chat_history"
        if not chat_dir.exists():
            return []

        sessions = []
        chat_files = sorted(chat_dir.glob("chat_*.json"), reverse=True)

        for chat_file in chat_files:
            try:
                data = json.loads(chat_file.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "id": data.get("session_id", chat_file.stem),
                        "started_at": data.get("started_at", "Unknown"),
                        "message_count": len(data.get("messages", [])),
                        "athlete_name": data.get("athlete_name", "Unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse {chat_file}: {e}")

        return sessions

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Format messages for LLM with system prompt.

        Returns:
            List of messages including system prompt and conversation history
        """
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (without timestamps for LLM)
        for msg in self.conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    def _build_system_prompt(self) -> str:
        """Build context-aware system prompt.

        Returns:
            System prompt with athlete context and tool information
        """
        summary = self.context.get("summary", {})
        athlete_name = self.context.get("athlete_name", "Unknown")
        analysis_date = self.context.get("analysis_date", "Unknown")
        competitions = self.context.get("competitions", [])

        comp_text = ""
        if competitions:
            comp_list = "\n".join(
                [
                    f"  - {c.get('name', 'Unknown')} ({c.get('date', 'TBD')}) - "
                    f"Priority {c.get('priority', '?')} {c.get('race_type', '')}"
                    for c in competitions
                ]
            )
            comp_text = f"\n\nUpcoming Competitions:\n{comp_list}"

        return f"""You are an AI training coach assistant for the Garmin AI Coach system.

You have access to the athlete's complete training analysis and plans from their most recent analysis.

## Athlete Information
- Name: {athlete_name}
- Analysis Date: {analysis_date}{comp_text}

## Available Tools

**IMPORTANT: You MUST use these tools to answer questions. DO NOT ask the user for information that exists in the analysis.**

1. **query_analysis**: Search expert analysis outputs (metrics, activity, physiology)
   - **USE THIS FIRST** for any questions about fitness, HRV, heart rate, training load, recovery, sleep, stress, or physiological data
   - Can filter by domain (metrics/activity/physiology/all)
   - The analysis contains detailed information - search it before asking the user

2. **modify_plan**: Propose modifications to training plans
   - Use this when the athlete requests changes to their weekly or season plan
   - Always explain the reasoning for modifications
   - Creates versioned plans (never overwrites originals)

3. **lookup_data**: Retrieve specific metrics, activities, or competition details
   - Use this for competition info, execution metadata, or athlete summary

## Critical Guidelines

1. **ALWAYS search the analysis first**: When the user asks about their data (HRV, HR, sleep, training load, etc.), use `query_analysis` to find the answer. The expert outputs contain comprehensive physiological analysis.

2. **Be efficient with tool use**:
   - Use 1-3 targeted queries rather than many small queries
   - Each tool result contains substantial information - read it carefully
   - After getting relevant results, synthesize and answer - don't keep searching
   - If you find the answer in the first tool call, stop and respond

3. **Answer from data, not speculation**: After using tools, provide specific answers based on what you found. Quote actual values and trends from the analysis.

4. **Only ask clarifying questions** when:
   - The user's request is ambiguous (e.g., "which week?" when multiple options exist)
   - The data truly doesn't exist in the analysis
   - You need to understand their preferences for plan modifications

5. **Never ask for data you can search**: Don't ask "What's your HRV?" or "How did you sleep?" - use the tools to find this information.

6. **Be concise and actionable**: Reference specific numbers and timeframes from the analysis.

## Example Interactions

❌ BAD:
User: "What's my current HRV status?"
Assistant: "I'd need to know your recent HRV values. Can you tell me what they've been?"

✅ GOOD:
User: "What's my current HRV status?"
Assistant: [Uses query_analysis tool with query="HRV" domain="physiology"]
Assistant: "Based on your analysis, your HRV has been running 39 ms recently (last night), which is below your baseline band of 48-61 ms. Your weekly average is 42 ms, consistently below your baseline floor. This indicates your system is in a suppressed state, likely due to the active illness starting Dec 29."
"""

    def save(self) -> None:
        """Persist conversation to disk."""
        data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "athlete_name": self.context.get("athlete_name", "Unknown"),
            "analysis_date": self.context.get("analysis_date", "Unknown"),
            "started_at": (
                self.conversation_history[0]["timestamp"] if self.conversation_history else None
            ),
            "messages": self.conversation_history,
        }

        try:
            self.storage_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.debug(f"Saved conversation to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def get_session_summary(self) -> str:
        """Get a summary of the current session.

        Returns:
            Summary string with session info and message count
        """
        msg_count = len(self.conversation_history)
        user_msgs = sum(1 for m in self.conversation_history if m["role"] == "user")
        assistant_msgs = sum(1 for m in self.conversation_history if m["role"] == "assistant")

        return (
            f"Session: {self.session_id}\n"
            f"Messages: {msg_count} total ({user_msgs} user, {assistant_msgs} assistant)\n"
            f"Storage: {self.storage_path}"
        )
