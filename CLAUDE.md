# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Garmin AI Coach is a multi-agent AI system that transforms Garmin Connect training data into personalized performance analysis and training plans. Built with LangGraph for AI orchestration, it uses a sophisticated 2-stage architecture with parallel data processing and specialized agents.

## Essential Commands

### Development Setup
```bash
# Install dependencies (requires Pixi: https://pixi.sh)
pixi install

# Create a config template
pixi run coach-init my_training_config.yaml

# Run analysis and planning
pixi run coach-cli --config my_training_config.yaml
```

### Code Quality
```bash
# Lint with Ruff
pixi run lint-ruff

# Auto-fix linting issues
pixi run ruff-fix

# Format code (Black + isort)
pixi run format

# Type checking with MyPy
pixi run type-check
```

### Testing
```bash
# Run test suite
pixi run test

# Run with coverage
pixi run test-cov
```

### Analysis
```bash
# Detect dead code with Vulture
pixi run dead-code
```

## AI Configuration & Provider System

### AI Modes
The system uses four AI modes that determine model assignments:
- `development` — Fast iteration (gemini-3-flash-preview for summarizers, claude-sonnet-4.6 for experts/planners)
- `standard` — Production quality (same as development)
- `cost_effective` — Budget-conscious (same model split as standard)
- `pro` — Maximum performance (gemini-3-flash-preview for summarizers, claude-opus for experts/planners)

### Model Assignment Strategy
Model selection is **role-based** via `services/ai/ai_settings.py:19-55`. The `AISettings` class maps each `AgentRole` to a specific model ID for each `AIMode`.

**Key files:**
- `services/ai/ai_settings.py` — `AISettings.model_assignments` dict defines role→model mapping for each mode
- `services/ai/model_config.py` — `ModelSelector.CONFIGURATIONS` dict (line 22-69) maps model IDs to provider configurations
- `core/config.py` — Loads `AI_MODE` from environment and validates API keys

**Provider Selection Logic:**
1. CLI reads config's `extraction.ai_mode` and exports `AI_MODE` env var (cli/garmin_ai_coach_cli.py:126)
2. `AISettings.get_model_for_role(role)` retrieves the model ID for the current mode
3. `ModelSelector.get_llm(role)` (model_config.py:72-135) creates the LLM client:
   - Reads model config from `CONFIGURATIONS`
   - Auto-selects API key based on `base_url` (anthropic/openai/google)
   - Applies model-specific configs (thinking modes, reasoning params)

**Supported Models:**
- Anthropic: claude-sonnet-4.6 (64K tokens, 1M context beta), claude-sonnet-4.6-thinking, claude-opus, claude-3-haiku
- Google AI Studio (direct): gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash

## Architecture

### LangGraph Workflow System
The system uses **LangGraph** (services/ai/langgraph/) for state-based AI orchestration with:
- Typed state management (`TrainingAnalysisState` in state/training_analysis_state.py)
- Parallel execution of independent nodes
- Automatic state reducers for lists and dicts
- Built-in HITL (Human-in-the-Loop) via `GraphInterrupt`
- LangSmith observability integration

### Workflow Structure

**Analysis Workflow** (services/ai/langgraph/workflows/analysis_workflow.py):
```
START → [metrics_summarizer, physiology_summarizer, activity_summarizer] (parallel)
          ↓                      ↓                          ↓
    metrics_expert      physiology_expert        activity_expert
          ↓                      ↓                          ↓
                    master_orchestrator
                            ↓
                       synthesis
                            ↓
                       formatter
                            ↓
                   plot_resolution → END
```

**Planning Workflow** (services/ai/langgraph/workflows/planning_workflow.py):
```
START → season_planner → master_orchestrator → data_integration → weekly_planner
                                    ↓                                    ↓
                              plan_formatter ← ← ← ← ← ← ← ← ← ← ← ← ← ←
                                    ↓
                                   END
```

**Integrated Workflow** (create_integrated_analysis_and_planning_workflow):
- Combines both workflows with single execution context
- Master orchestrator uses dynamic routing with `Command(goto=...)`
- Supports parallel branches for analysis and planning

### Agent Nodes

**2-Stage Architecture:**
1. **Summarizers** (Stage 1) — Data reduction and structuring
   - metrics_summarizer_node.py — Training load, VO₂ max trends
   - physiology_summarizer_node.py — HRV, stress, recovery metrics
   - activity_summarizer_node.py — Raw activity data processing

2. **Experts** (Stage 2) — Analysis and insights
   - metrics_expert_node.py — Performance metrics analysis
   - physiology_expert_node.py — Physiological adaptation insights
   - activity_expert_node.py — Training pattern interpretation

**Synthesis & Formatting:**
- synthesis_node.py — Combines expert insights into coherent analysis
- formatter_node.py — Produces analysis HTML reports
- plan_formatter_node.py — Produces planning HTML reports
- plot_resolution_node.py — Resolves `[PLOT:id]` references in HTML

**Planning Nodes:**
- season_planner_node.py — Long-term periodization (macrocycle design)
- data_integration_node.py — Integrates analysis outputs with planning context
- weekly_planner_node.py — 28-day detailed training plans

**Orchestration:**
- orchestrator_node.py — Master orchestrator for dynamic workflow routing
- Uses `Command(goto=...)` to route between analysis/planning stages

### State Management

**TrainingAnalysisState** (services/ai/langgraph/state/training_analysis_state.py):
- Extends `MessagesState` for LangGraph
- Uses annotated reducers for automatic aggregation:
  - Lists: `lambda x, y: x + y` (append)
  - Dicts: `lambda x, y: {**x, **y}` (merge)
  - Booleans: `lambda x, y: x or y` (OR logic)
- Per-agent HITL message storage (metrics_expert_messages, weekly_planner_messages, etc.)

**Key state fields:**
- Input: user_id, athlete_name, garmin_data, analysis_context, planning_context
- Summarizer outputs: metrics_summary, physiology_summary, activity_summary
- Expert outputs: metrics_outputs, activity_outputs, physiology_outputs (typed Pydantic models)
- Final outputs: synthesis_result, season_plan, weekly_plan, analysis_html, planning_html
- Metadata: plots, costs, errors, tool_usage, available_plots

### Human-in-the-Loop (HITL)

**Mechanism:**
- Enabled by default via `hitl_enabled: true` in config
- Implemented using LangGraph's `GraphInterrupt` with `ask_human_tool`
- Agents can pause execution to ask clarifying questions

**Coverage:**
- All expert nodes (metrics, physiology, activity)
- Planning nodes (season planner, weekly planner)
- Terminal-based prompts during execution

**Implementation Pattern:**
```python
from langgraph.errors import GraphInterrupt

def some_expert_node(state):
    if state["hitl_enabled"] and need_clarification:
        raise GraphInterrupt({"question": "...", "context": "..."})
    # ... continue processing
```

### Plotting System

**Location:** services/ai/tools/plotting/
- `PlotStorage` — Manages plot metadata and HTML storage
- `ProductionSecureExecutor` — Sandboxed Python code execution for plot generation
- `PlotReferenceResolver` — Replaces `[PLOT:plot_id]` with embedded HTML

**Usage Pattern:**
1. Agents use plotting tool to generate visualizations
2. Plots stored with unique IDs in state
3. `plot_resolution_node` embeds plots in final HTML

## Configuration System

### Config File Structure (YAML/JSON)
```yaml
athlete:
  name: "Athlete Name"
  email: "email@example.com"  # Required
  timezone: "Europe/Paris"    # Optional — defaults to UTC

context:
  analysis: "Past data interpretation context (injuries, priorities, etc.)"
  planning: "Future training context (goals, constraints, zones, preferences)"

extraction:
  activities_days: 21      # Activity data lookback (7-56)
  metrics_days: 56         # Metrics data lookback (14-90)
  ai_mode: "standard"      # development | standard | cost_effective | pro
  hitl_enabled: true       # Enable conversational agents (default: true)
  skip_synthesis: false    # Skip synthesis stage to save tokens (default: false)

competitions:
  - name: "Race Name"
    date: "2025-10-12"     # ISO format YYYY-MM-DD
    race_type: "Half Marathon"
    priority: "A"          # A (key race), B (important), C (training)
    target_time: "01:40:00"  # Optional HH:MM:SS

output:
  directory: "./data"

credentials:
  password: ""  # Leave empty for secure prompt at runtime
```

### Environment Variables (.env)
```bash
# LLM Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Observability (optional but recommended)
LANGSMITH_API_KEY=lsv2_...

# AI Mode (overridden by config's extraction.ai_mode)
AI_MODE=development

# Logging
LOG_LEVEL=INFO
```

## Data Flow & Dependencies

### Garmin Data Extraction
**Location:** services/garmin/
- `client.py` — Garmin Connect API client
- `data_extractor.py` — Orchestrates data fetching and processing
- `models.py` — Pydantic models for Garmin data (activities, metrics, sleep)
- `competition_models.py` — Competition data models

### Outside AthleteReg Integration
**Location:** services/outside/
- `client.py` — AthleteReg API client (BikeReg, RunReg, TriReg, SkiReg)
- `models.py` — Competition models
- Optional auto-import of registered competitions

### Report Generation
**Location:** services/report/
- HTML template processing
- Embedded plot resolution
- Cost tracking integration

## Common Development Patterns

### Adding a New Agent Node

1. Create node file in `services/ai/langgraph/nodes/`
2. Define agent role in `services/ai/ai_settings.py` (add to `AgentRole` enum)
3. Update model assignments in `AISettings.model_assignments`
4. Implement node function following this pattern:
```python
from services.ai.ai_settings import AgentRole
from services.ai.model_config import ModelSelector
from services.ai.langgraph.nodes.node_base import configure_node_tools

def my_agent_node(state: TrainingAnalysisState) -> dict:
    agent_name = "MyAgent"
    llm = ModelSelector.get_llm(AgentRole.MY_ROLE)

    # Configure tools if needed
    tools = configure_node_tools(
        agent_name=agent_name,
        plotting_enabled=state.get("plotting_enabled", False),
    )

    # Agent logic here

    return {"my_result": result}
```

5. Add node to workflow in `services/ai/langgraph/workflows/`
6. Update state schema if adding new fields

### Modifying Model Assignments

**To change which model is used for a specific role:**
1. Edit `services/ai/ai_settings.py` → `AISettings.model_assignments`
2. Update the model ID for the desired `AIMode` and `AgentRole`
3. Ensure model ID exists in `services/ai/model_config.py` → `ModelSelector.CONFIGURATIONS`

**To add a new model:**
1. Add model configuration to `ModelSelector.CONFIGURATIONS` dict
2. Update `AISettings.model_assignments` to reference new model ID

### Testing Workflow Changes

**For local testing:**
1. Set `AI_MODE=development` in .env
2. Use small data ranges (activities_days: 7, metrics_days: 14)
3. Enable debug logging: `LOG_LEVEL=DEBUG`
4. Check LangSmith dashboard for execution traces (requires LANGSMITH_API_KEY)

**For cost-effective testing:**
1. Use `ai_mode: "cost_effective"` in config
2. Set `skip_synthesis: true` to bypass synthesis stage
3. Monitor costs via summary.json output

## Code Style & Conventions

### Tools Configuration
- **Ruff** line length: 120 chars (pyproject.toml:38)
- **Black** line length: 100 chars (pyproject.toml:32)
- Prefer Ruff for most checks; Black for formatting
- Use `pixi run format` before committing

### Type Hints
- Use Python 3.13+ type syntax: `dict[str, Any]`, `list[str]`
- Pydantic v2 for data validation
- MyPy for static type checking

### Logging
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Info level for workflow progress
- Debug level for detailed diagnostics
- Warning/Error for issues

### File Organization
```
services/ai/
├── ai_settings.py          # AgentRole enum, AISettings class
├── model_config.py         # ModelSelector, model configurations
├── langgraph/
│   ├── config/             # LangSmith setup
│   ├── nodes/              # Agent node implementations
│   ├── workflows/          # Workflow definitions
│   ├── state/              # State schemas
│   ├── schemas/            # Pydantic output models
│   └── utils/              # Cost tracking, helpers
└── tools/                  # Plotting and other tools
```

## Output Artifacts

**Generated in `output.directory` (default: ./data):**
- `analysis.html` — Interactive performance analysis report
- `planning.html` — Detailed 28-day training plan
- `metrics_result.md` — Metrics expert output (intermediate)
- `activity_result.md` — Activity expert output (intermediate)
- `physiology_result.md` — Physiology expert output (intermediate)
- `season_plan.md` — Season planner output (intermediate)
- `summary.json` — Execution metadata and cost tracking

**summary.json structure:**
```json
{
  "athlete": "Name",
  "analysis_date": "2025-12-31",
  "total_cost_usd": 1.23,
  "total_tokens": 50000,
  "execution_id": "user_20251231_123456_complete",
  "trace_id": "langsmith-trace-id",
  "root_run_id": "langsmith-run-id",
  "files_generated": ["analysis.html", "planning.html"],
  "competitions": [...]
}
```

## Troubleshooting

### Provider API Key Issues
**Problem:** "Invalid API key format" or "No API key available"
**Solution:**
1. Check .env file has correct provider key
2. Verify `AI_MODE` mapping in ai_settings.py points to correct provider
3. If using only OpenAI: set `ai_mode: "standard"` in config OR update ai_settings.py
4. If using only Anthropic: set `ai_mode: "development"` or `"cost_effective"`

### Workflow Execution Failures
**Problem:** Node fails or workflow hangs
**Solution:**
1. Enable debug logging: `LOG_LEVEL=DEBUG`
2. Check LangSmith dashboard (if enabled) for detailed traces
3. Review state at failure point (logged to console)
4. Ensure all required state fields are populated

### HITL Not Working
**Problem:** Agent doesn't pause for questions
**Solution:**
1. Verify `hitl_enabled: true` in config
2. Check node implements GraphInterrupt correctly
3. Ensure terminal is interactive (not running in background)

### High Costs
**Problem:** Workflow too expensive
**Solution:**
1. Use `ai_mode: "cost_effective"` for budget mode
2. Reduce data ranges (activities_days, metrics_days)
3. Enable `skip_synthesis: true` to bypass synthesis stage
4. Check LangSmith for token usage breakdown by node

## Important Notes

- **Never commit .env files** — Use .env.example as template
- **Config validation** — athlete.email is required; dates must be ISO format
- **Parallel execution** — Summarizer nodes run concurrently for speed
- **State reducers** — Understand how lists/dicts automatically merge in state
- **Dynamic routing** — Master orchestrator uses Command(goto=...) not static edges
- **Plot references** — Use `[PLOT:plot_id]` syntax in markdown; resolved at end
- **HITL mode** — Agents can ask questions; design prompts to minimize interruptions
- **Provider mapping** — AI mode determines which provider is used; ensure API keys match
