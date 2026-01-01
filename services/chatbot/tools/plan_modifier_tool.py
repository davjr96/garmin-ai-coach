"""Tool for modifying training plans."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from services.ai.utils.plan_storage import FilePlanStorage

logger = logging.getLogger(__name__)


def create_plan_modifier_tool(context: dict[str, Any], output_dir: Path, user_id: str):
    """Create plan modifier tool with context.

    Args:
        context: Analysis context loaded by AnalysisContextLoader
        output_dir: Directory for storing modified plans
        user_id: User identifier

    Returns:
        LangChain tool for modifying training plans
    """

    @tool
    def modify_plan(modification_request: str, plan_type: str = "weekly") -> str:
        """Modify a training plan based on athlete requests.

        Use this tool when the athlete wants to adjust their training plan.
        This creates a new versioned plan file without overwriting the original.

        Args:
            modification_request: Description of desired changes (e.g., "move long run to Sunday",
                                 "reduce mileage by 10%", "add rest day on Tuesday")
            plan_type: Type of plan to modify - "weekly" or "season"

        Returns:
            Confirmation message with path to modified plan and preview
        """
        if plan_type not in ["weekly", "season"]:
            return f"Invalid plan_type '{plan_type}'. Use 'weekly' or 'season'."

        # Load current plan
        plan_key = f"{plan_type}_plan"
        current_plan = context.get(plan_key, "")

        if not current_plan:
            return f"No {plan_type} plan found in analysis outputs. Cannot modify."

        # Generate modified plan
        # For MVP, we append the modification request with clear formatting
        # Future enhancement: use planner nodes to regenerate plan intelligently
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version_suffix = datetime.now().strftime("v%Y%m%d_%H%M%S")

        modification_section = f"""

---

## MODIFICATION ({timestamp})

### Requested Changes
{modification_request}

### Implementation Notes
This is a manual modification request. The plan above remains unchanged as the original baseline.
Please review the requested changes and apply them manually, or regenerate the plan with updated constraints.

### Modification History
- {timestamp}: {modification_request}
"""

        modified_plan = current_plan + modification_section

        # Save versioned copy using FilePlanStorage
        try:
            storage = FilePlanStorage()
            versioned_plan_type = f"{plan_type}_plan_{version_suffix}"
            storage.save_plan(user_id, versioned_plan_type, modified_plan)

            # Also save to output directory for immediate access
            version_file = Path(output_dir) / f"{plan_type}_plan_{version_suffix}.md"
            version_file.write_text(modified_plan, encoding="utf-8")

            logger.info(f"Saved modified {plan_type} plan: {version_file}")

            # Generate preview
            preview_lines = modified_plan.split("\n")[-15:]  # Last 15 lines
            preview = "\n".join(preview_lines)

            return f"""✅ Modified {plan_type} plan saved successfully!

**File locations:**
- Output directory: {version_file}
- Storage: data/storage/{user_id}/{versioned_plan_type}.md

**Preview of modification:**
```
{preview}
```

**Important:** This modification has been logged but not automatically applied to the plan.
The original plan structure remains intact. To see the full updated plan, review the file above.

**Note:** For sophisticated plan regeneration that respects training principles, consider
re-running the full analysis with updated planning context that includes these constraints.
"""
        except Exception as e:
            logger.error(f"Failed to save modified plan: {e}")
            return f"❌ Error saving modified plan: {e}"

    return modify_plan
