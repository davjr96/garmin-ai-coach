"""Main chatbot service for post-analysis querying."""

import logging
from pathlib import Path
from typing import Any

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.nodes.tool_calling_helper import (
    extract_text_content, handle_tool_calling_in_node)
from services.ai.model_config import ModelSelector
from services.chatbot.conversation_manager import ConversationManager
from services.chatbot.tools import (create_analysis_query_tool,
                                    create_data_lookup_tool,
                                    create_plan_modifier_tool)

logger = logging.getLogger(__name__)


class ChatbotService:
    """Main service orchestrating the chatbot loop."""

    def __init__(
        self,
        context: dict[str, Any],
        user_id: str,
        output_dir: Path,
        resume: bool = False,
        session_id: str | None = None,
    ):
        """Initialize chatbot service.

        Args:
            context: Analysis context loaded by AnalysisContextLoader
            user_id: User identifier
            output_dir: Directory for storing conversation history and modified plans
            resume: If True, resume the most recent conversation
            session_id: Specific session ID to resume (overrides resume)
        """
        self.context = context
        self.user_id = user_id
        self.output_dir = Path(output_dir)

        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            user_id, output_dir, context, resume=resume, session_id=session_id
        )

        # Initialize LLM using existing ModelSelector pattern
        self.llm = ModelSelector.get_llm(AgentRole.CHATBOT)

        # Configure tools
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        logger.info(
            f"ChatbotService initialized for {context.get('athlete_name')} "
            f"(session: {self.conversation_manager.session_id})"
        )

    def _setup_tools(self) -> list:
        """Configure chatbot tools.

        Returns:
            List of LangChain tools
        """
        return [
            create_analysis_query_tool(self.context),
            create_plan_modifier_tool(self.context, self.output_dir, self.user_id),
            create_data_lookup_tool(self.context),
        ]

    async def run_interactive_loop(self) -> None:
        """Run the main interactive REPL loop.

        This is the main entry point for chatbot interaction.
        Continues until user types 'exit', 'quit', or 'q'.
        """
        print(self._get_welcome_message())

        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n> ").strip()
                except EOFError:
                    # Handle Ctrl+D
                    print("\n")
                    break

                # Handle exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    await self._handle_exit()
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Add user message to history
                self.conversation_manager.add_message("user", user_input)

                # Process message with LLM + tools
                try:
                    response = await self._process_message(user_input)

                    # Display response
                    print(f"\nAssistant: {response}\n")

                    # Add assistant message to history
                    self.conversation_manager.add_message("assistant", response)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    print(f"\n❌ Error: {e}\n")
                    print("Please try again or rephrase your question.\n")

                # Persist conversation after each turn
                self.conversation_manager.save()

        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\n")
            await self._handle_exit()

    async def _process_message(self, user_input: str) -> str:
        """Process a message with LLM and tool calling.

        Args:
            user_input: User's message

        Returns:
            Assistant's response
        """
        # Get full message list with system prompt
        messages = self.conversation_manager.get_messages_for_llm()

        # Use existing tool calling pattern from tool_calling_helper.py
        response = await handle_tool_calling_in_node(
            llm_with_tools=self.llm_with_tools,
            messages=messages,
            tools=self.tools,
            max_iterations=8,
            base_llm=self.llm,
        )

        # Extract text content from response
        return extract_text_content(response)

    def _get_welcome_message(self) -> str:
        """Generate welcome message with athlete info.

        Returns:
            Welcome message string
        """
        athlete = self.context.get("athlete_name", "Unknown")
        date = self.context.get("analysis_date", "Unknown")
        num_competitions = len(self.context.get("competitions", []))

        comp_text = ""
        if num_competitions > 0:
            comp_text = f"\nUpcoming competitions: {num_competitions}"

        # Check if resuming
        msg_count = len(self.conversation_manager.conversation_history)
        resume_text = ""
        if msg_count > 0:
            resume_text = f"\n\n✓ Resumed conversation with {msg_count} previous messages"

        return f"""
{'='*60}
Garmin AI Coach - Interactive Chatbot
{'='*60}

Athlete: {athlete}
Analysis Date: {date}{comp_text}{resume_text}

I can help you:
  • Explain your analysis findings
  • Answer questions about your training data
  • Modify your training plan
  • Look up specific metrics or competitions

Available commands:
  /help    - Show this message
  /info    - Show session information
  /new     - Start a new conversation
  /exit    - Exit the chatbot

Type your question or 'exit' to quit.
{'='*60}
"""

    def _handle_command(self, command: str) -> None:
        """Handle special commands.

        Args:
            command: Command string starting with /
        """
        cmd = command.lower().strip()

        if cmd == "/help":
            print(self._get_welcome_message())

        elif cmd == "/info":
            print("\n" + self.conversation_manager.get_session_summary())

        elif cmd == "/new":
            # Start a new conversation
            old_session = self.conversation_manager.session_id
            self.conversation_manager.conversation_history = []
            self.conversation_manager.session_id = self.conversation_manager._generate_session_id()
            self.conversation_manager.storage_path = self.conversation_manager._get_storage_path()
            print(f"\n✓ Started new conversation (previous: {old_session})\n")

        elif cmd == "/exit":
            print("\nUse 'exit' or Ctrl+C to quit.\n")

        else:
            print(f"\nUnknown command: {command}")
            print("Available commands: /help, /info, /new, /exit\n")

    async def _handle_exit(self) -> None:
        """Handle graceful exit."""
        # Save final conversation state
        self.conversation_manager.save()

        # Display session summary
        print("\n" + "=" * 60)
        print("Chat Session Summary")
        print("=" * 60)
        print(self.conversation_manager.get_session_summary())
        print("=" * 60)
        print("\nThank you for using Garmin AI Coach Chatbot!")
        print("Your conversation has been saved.\n")
