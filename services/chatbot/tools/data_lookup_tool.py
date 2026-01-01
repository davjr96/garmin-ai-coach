"""Tool for looking up specific data from analysis."""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_data_lookup_tool(context: dict[str, Any]):
    """Create data lookup tool with context.

    Args:
        context: Analysis context loaded by AnalysisContextLoader

    Returns:
        LangChain tool for looking up specific data
    """

    @tool
    def lookup_data(data_type: str, filter_param: str = "") -> str:
        """Look up specific training data, metrics, or competition details.

        Use this tool to retrieve specific data points like competitions, summary statistics,
        or execution metadata.

        Args:
            data_type: Type of data to look up - "competitions", "summary", "metadata", or "athlete"
            filter_param: Optional filter parameter (e.g., race name, metric name)

        Returns:
            Formatted data matching the request
        """
        summary = context.get("summary", {})

        if data_type == "competitions":
            competitions = context.get("competitions", [])
            if not competitions:
                return "No competitions found in the analysis."

            if filter_param:
                # Filter by race name or type
                filter_lower = filter_param.lower()
                filtered = [
                    c
                    for c in competitions
                    if filter_lower in c.get("name", "").lower()
                    or filter_lower in c.get("race_type", "").lower()
                ]
                competitions = filtered if filtered else competitions

            comp_list = []
            for comp in competitions:
                comp_str = (
                    f"- **{comp.get('name', 'Unknown')}**\n"
                    f"  - Date: {comp.get('date', 'TBD')}\n"
                    f"  - Type: {comp.get('race_type', 'Unknown')}\n"
                    f"  - Priority: {comp.get('priority', '?')}"
                )
                if "target_time" in comp:
                    comp_str += f"\n  - Target: {comp['target_time']}"
                comp_list.append(comp_str)

            return "**Upcoming Competitions:**\n\n" + "\n\n".join(comp_list)

        elif data_type == "summary":
            # Return key summary information
            return f"""**Analysis Summary:**

- **Athlete:** {summary.get('athlete', 'Unknown')}
- **Analysis Date:** {summary.get('analysis_date', 'Unknown')}
- **Total Cost:** ${summary.get('total_cost_usd', 0):.2f}
- **Total Tokens:** {summary.get('total_tokens', 0):,}
- **Execution ID:** {summary.get('execution_id', 'Unknown')}
- **Files Generated:** {len(summary.get('files_generated', []))} files

**Generated Files:**
{chr(10).join('- ' + f for f in summary.get('files_generated', []))}
"""

        elif data_type == "metadata":
            # Return execution metadata
            metadata = {
                "execution_id": summary.get("execution_id"),
                "trace_id": summary.get("trace_id"),
                "root_run_id": summary.get("root_run_id"),
                "analysis_date": summary.get("analysis_date"),
                "total_cost_usd": summary.get("total_cost_usd"),
                "total_tokens": summary.get("total_tokens"),
            }
            return "**Execution Metadata:**\n\n```json\n" + json.dumps(metadata, indent=2) + "\n```"

        elif data_type == "athlete":
            # Return athlete information
            return f"""**Athlete Information:**

- **Name:** {context.get('athlete_name', 'Unknown')}
- **Analysis Date:** {context.get('analysis_date', 'Unknown')}
- **Competitions Registered:** {len(context.get('competitions', []))}
"""

        else:
            return (
                f"Unknown data_type '{data_type}'. "
                "Available types: competitions, summary, metadata, athlete"
            )

    return lookup_data
