import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import AgentOutput
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.message_helper import normalize_langchain_messages
from services.ai.langgraph.utils.output_helper import extract_agent_content, extract_expert_output
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import (
    configure_node_tools,
    create_cost_entry,
    execute_node_with_error_handling,
    log_node_completion,
)
from .prompt_components import get_hitl_instructions, get_workflow_context
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)

WEEKLY_PLANNER_SYSTEM_PROMPT = """## Goal
Create detailed, practical training plans that balance stress and recovery.
## Principles
- Adaptation: Progressive overload with adequate recovery.
- Specificity: Training must match the demands of the event.
- Individualization: Adapt to the athlete's current state and history."""

WEEKLY_PLANNER_USER_PROMPT = """## Task
Create a detailed 28-day (4-week) training plan.

## Constraints
- **Honor the Phase**: Prioritize the Season Plan's phase intent.
- **Respect Readiness**: Adjust intensity based on Physiology/Metrics signals (e.g., pull back if recovery is low).
- **Integrate Signals**: Use Activity Expert advice for session structure.
- **Brevity**: Use standard notation (e.g., "4x(5' Z4, 2' r)") to keep the plan compact.

## Inputs
### Season Plan
```markdown
{season_plan}
```
### Athlete Context
- Name: {athlete_name}
- Date: ```json {current_date} ```
- Upcoming Weeks: ```json {week_dates} ```
- Competitions: ```json {competitions} ```
- **User Context**: ``` {planning_context} ```

### Expert Analysis
- Metrics: ``` {metrics_analysis} ```
- Activity: ``` {activity_analysis} ```
- Physiology: ``` {physiology_analysis} ```

## Output Requirements
1. **Zones Table**: Define intensity zones first.
2. **Structure**: Group by Week (1-4).
3. **Daily Format**:
   - **DAY & DATE**: e.g., "Mon, Nov 24"
   - **FOCUS**: 1-2 words (e.g., "Recovery", "VO2max")
   - **WORKOUT**: Concise structure string.
   - **PURPOSE**: One short sentence.
   - **ADAPTATION**: "If tired: ..."

**Important:**
- Use recent activity data to continue the current training flow and don't start a new phase.
- Use the Season Plan as a guide, but don't force it.
- Place sessions smartly to avoid back to back high intensity sessions or strength sessions etc.
- **Intensity distribution**: For recreational/amateur athletes, limit to **1 high-intensity session per week** (sustained Z4+ efforts: intervals, tempo, threshold). Elite athletes may handle 2-3. The remaining volume should be Z1-2 (80-90% of total training). Violating this balance is the most common mistake — protect aerobic volume. **Exception**: hill sprints (≤10s maximal efforts) are a neuromuscular/power stimulus, not a metabolic load — they do not count toward the high-intensity session cap and can be added to easy days.
- **Double-threshold sessions** (advanced, race-specific phase only): back-to-back threshold efforts on consecutive days (e.g. threshold intervals day 1, tempo run day 2). Each day counts as one high-intensity session — only use for athletes who can sustain 2+ high-intensity sessions per week (i.e. not recreational/amateur). Use sparingly and only when physiology signals strong readiness.
- **Strength sessions**: Always prescribe specific exercises, sets, reps, and weights. Use the "Strength Baseline" and progressive overload targets from the Activity Expert. Format as: `Exercise: Nxreps @ Xkg`. If no baseline exists, use bodyweight or note "start light."
  - ME gym sessions require **7-10 days recovery** — do not schedule more frequently.
  - When building from scratch, progress load every 1-2 sessions: start bodyweight 4×10, add ~10% BW every few sessions, increase reps before load, reduce rest intervals as fitness improves. Do not push to failure.
- **Trail-specific session formats** (when applicable to the athlete's sport):
  - Hill sprints (power, steep): `8x(10" max, 20%+ grade, 3' walk r)` — short, maximal, full recovery; not a cardio session. During Base Period: 1-2x/week. Outside Base Period (maintenance): every 12-14 days. Progression: steeper grade or add 10% BW via weight vest.
  - Hill intervals (muscular endurance, moderate grade): `5x(2' hard uphill, 6-8% grade, easy jog down r)` — build from 10 min to 20 min total uphill work over weeks; excellent VO2max stimulus with lower injury risk than flat running
  - Downhill intervals (anterior chain): `6x(3' controlled descent, >20% grade, 2' r)` — quad/eccentric focus; max every 2-3 weeks; requires easy days before and after
  - Muscular endurance uphill: `3x(10' Z3 uphill, 3' r)` or `4x(5' Z4, 2' r)` — sustained effort, not sprints
  - Long run with embedded tempo: 3-4 hour run; after 20-30 min warm-up, insert 20 min continuous tempo OR `3x(8' tempo, 5' r)`; prefer this over 5-6+ hour slow runs (diminishing returns beyond 3-4 hours)
  - Combo workout (fatigue resistance): `5x(90" hill, easy 5') → 20' tempo` or similar hill+tempo sequences; max every 2 weeks in final 3 months pre-race; schedule last combo session ≥3 weeks before race
  - Epic mountain day (simulation): 8-10 hours entirely Z1; schedule 1-2 times in the build (4-8 weeks pre-race); use to test gear, shoes, and nutrition strategy; full easy day before and after
- **Ski-to-run transition** (when athlete is balancing ski season with running): During active ski season, preserve 1-2 short runs per week to maintain running-specific neuromuscular patterns; add strides (4-6x20s fast, relaxed) to at least one run per week. When transitioning out of ski season, replace ski days with run days gradually over 2-4 weeks rather than abruptly. Delay high-volume downhill running or hard intensity until 2-4 weeks of consistent run volume is re-established — cardiovascular fitness transfers well from skiing but tendons and connective tissue require dedicated adaptation time.
"""

WEEKLY_PLANNER_FINAL_CHECKLIST = """
## Final Checklist
- Follow 28-day horizon and week grouping.
- Do not contradict expert constraints.
- Keep output compact and structured.
"""


async def weekly_planner_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting weekly planner node")

    hitl_enabled = state.get("hitl_enabled", True)
    logger.info("Weekly planner node: HITL %s", "enabled" if hitl_enabled else "disabled")

    agent_start_time = datetime.now()

    tools = configure_node_tools(
        agent_name="weekly_planner",
        plot_storage=None,
        plotting_enabled=False,
    )

    system_prompt = (
        get_workflow_context("weekly_planner")
        + WEEKLY_PLANNER_SYSTEM_PROMPT
        + (get_hitl_instructions("weekly_planner") if hitl_enabled else "")
        + WEEKLY_PLANNER_FINAL_CHECKLIST
    )

    qa_messages = normalize_langchain_messages(state.get("weekly_planner_messages", []))
    user_message = {
        "role": "user",
        "content": WEEKLY_PLANNER_USER_PROMPT.format(
            season_plan=extract_agent_content(state.get("season_plan")),
            athlete_name=state["athlete_name"],
            current_date=json.dumps(state["current_date"], indent=2),
            week_dates=json.dumps(state["week_dates"], indent=2),
            competitions=json.dumps(state["competitions"], indent=2),
            planning_context=state["planning_context"],
            metrics_analysis=extract_expert_output(state.get("metrics_outputs"), "for_weekly_planner"),
            activity_analysis=extract_expert_output(state.get("activity_outputs"), "for_weekly_planner"),
            physiology_analysis=extract_expert_output(state.get("physiology_outputs"), "for_weekly_planner"),
        ),
    }
    base_messages = [{"role": "system", "content": system_prompt}, user_message]

    base_llm = ModelSelector.get_llm(AgentRole.WORKOUT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(AgentOutput)

    async def call_weekly_planning():
        messages_with_qa = base_messages + qa_messages
        if tools:
            return await handle_tool_calling_in_node(
                llm_with_tools=llm_with_structure,
                messages=messages_with_qa,
                tools=tools,
                max_iterations=15,
            )
        return await llm_with_structure.ainvoke(messages_with_qa)

    async def node_execution():
        agent_output = await retry_with_backoff(
            call_weekly_planning, AI_ANALYSIS_CONFIG, "Weekly Planning"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        log_node_completion("Weekly planning", execution_time)

        return {
            "weekly_plan": agent_output.model_dump(),
            "costs": [create_cost_entry("weekly_planner", execution_time)],
        }

    return await execute_node_with_error_handling(
        node_name="Weekly planner",
        node_function=node_execution,
        error_message_prefix="Weekly planning failed",
    )
