"""Tool for querying expert analysis outputs."""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_analysis_query_tool(context: dict[str, Any]):
    """Create analysis query tool with context.

    Args:
        context: Analysis context loaded by AnalysisContextLoader

    Returns:
        LangChain tool for querying analysis outputs
    """

    @tool
    def query_analysis(query: str, domain: str = "all") -> str:
        """Query the expert analysis outputs for specific information.

        Use this tool to search through the analysis outputs when answering questions
        about the athlete's training data, fitness metrics, or physiological insights.

        Args:
            query: The question or topic to search for (e.g., "VO2max trends", "training load")
            domain: Which expert to query - "metrics", "activity", "physiology", or "all"

        Returns:
            Relevant excerpts from expert analysis outputs
        """
        results = []

        # Map domain to expert files
        expert_mapping = {
            "metrics": "metrics_expert",
            "activity": "activity_expert",
            "physiology": "physiology_expert",
        }

        # Determine which experts to search
        if domain == "all":
            experts_to_search = expert_mapping
        elif domain in expert_mapping:
            experts_to_search = {domain: expert_mapping[domain]}
        else:
            return f"Invalid domain '{domain}'. Use: metrics, activity, physiology, or all"

        query_lower = query.lower()

        # Search each expert's outputs
        for domain_name, expert_key in experts_to_search.items():
            expert_data = context.get(expert_key, {})

            if not expert_data:
                logger.debug(f"No data found for {expert_key}")
                continue

            # Search in the expert output structure
            # Expert outputs are Pydantic models dumped as JSON with structure like:
            # {"for_synthesis": {...}, "for_season_planner": {...}, "for_weekly_planner": {...}}
            if isinstance(expert_data, dict):
                for output_type, output_content in expert_data.items():
                    if isinstance(output_content, dict):
                        # Search through all fields in the output
                        for field_name, field_value in output_content.items():
                            if _matches_query(field_value, query_lower):
                                results.append(
                                    _format_result(
                                        domain_name, output_type, field_name, field_value
                                    )
                                )

        if not results:
            # Try searching in plans as fallback
            plan_results = _search_plans(context, query_lower)
            if plan_results:
                return plan_results

            return (
                f"No relevant information found for '{query}'. "
                "Try rephrasing your question or using the lookup_data tool for specific metrics."
            )

        # Limit results to top 3 most relevant (reduced from 5 to avoid overwhelming)
        limited_results = results[:3]
        result_text = "\n\n---\n\n".join(limited_results)

        if len(results) > 3:
            result_text += (
                f"\n\n... and {len(results) - 3} more results available. "
                "You have enough information to answer - synthesize what you found above."
            )

        return result_text

    return query_analysis


def _matches_query(value: Any, query_lower: str) -> bool:
    """Check if value contains the query terms.

    Matches if all individual terms (2+ chars) appear in the value,
    or if the full query appears as a substring. This allows queries like
    "HRV baseline" to match content containing both words separately.

    Args:
        value: Value to search in
        query_lower: Lowercase query string

    Returns:
        True if value matches query
    """
    if isinstance(value, str):
        value_lower = value.lower()
    elif isinstance(value, (list, dict)):
        value_lower = json.dumps(value, ensure_ascii=False).lower()
    else:
        value_lower = str(value).lower()

    # First try exact substring match
    if query_lower in value_lower:
        return True

    # Fall back to matching all individual terms (words 2+ chars)
    terms = [t for t in query_lower.split() if len(t) >= 2]
    if terms and all(term in value_lower for term in terms):
        return True

    return False


def _format_result(domain: str, output_type: str, field_name: str, field_value: Any) -> str:
    """Format a search result for display.

    Args:
        domain: Expert domain (metrics/activity/physiology)
        output_type: Output type (for_synthesis/for_season_planner/etc)
        field_name: Field name
        field_value: Field value

    Returns:
        Formatted result string
    """
    # Truncate long values
    if isinstance(field_value, str):
        display_value = field_value[:1000] + "..." if len(field_value) > 1000 else field_value
    elif isinstance(field_value, (list, dict)):
        value_str = json.dumps(field_value, indent=2, ensure_ascii=False)
        display_value = value_str[:1000] + "..." if len(value_str) > 1000 else value_str
    else:
        display_value = str(field_value)

    return f"**[{domain.upper()} - {output_type} - {field_name}]**\n\n{display_value}"


def _search_plans(context: dict[str, Any], query_lower: str) -> str:
    """Search in season and weekly plans as fallback.

    Args:
        context: Analysis context
        query_lower: Lowercase query string

    Returns:
        Search results from plans or empty string
    """
    results = []

    for plan_key in ["season_plan", "weekly_plan"]:
        plan_content = context.get(plan_key, "")
        if isinstance(plan_content, str) and query_lower in plan_content.lower():
            # Extract relevant section (paragraph containing query)
            lines = plan_content.split("\n")
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Get context (2 lines before and after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    excerpt = "\n".join(lines[start:end])
                    results.append(f"**[{plan_key.upper()}]**\n\n{excerpt}\n...")
                    break

    if results:
        return "\n\n---\n\n".join(results[:3])

    return ""
