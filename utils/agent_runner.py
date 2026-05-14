from smolagents import ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep

from utils.agent_tools import (
    SearchDocumentTool,
    ExtractEntitiesTool,
    TaskListTool,
    GenerateEmailTool,
    SummarizeTool,
)


LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_ID = "qwen2.5-3b-instruct"


def run_agent(query: str) -> tuple[str, list[str]]:
    """
    Runs the agent on a free-text query against loaded documents.
    Returns (final_answer, steps) where steps is a list of human-readable reasoning strings.
    """

    model = OpenAIServerModel(
        model_id=MODEL_ID,
        api_base=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
    )

    agent = ToolCallingAgent(
        tools=[
            SearchDocumentTool(),
            ExtractEntitiesTool(),
            TaskListTool(),
            GenerateEmailTool(),
            SummarizeTool(),
        ],
        model=model,
        max_steps=6,
        verbosity_level=0,
    )

    result = agent.run(query, reset=True)

    steps = _collect_steps(agent)

    return str(result), steps


def _collect_steps(agent: ToolCallingAgent) -> list[str]:
    steps = []

    for step in agent.memory.steps:
        if not isinstance(step, ActionStep):
            continue

        if step.tool_calls:
            for call in step.tool_calls:
                args_preview = str(call.arguments)
                if len(args_preview) > 120:
                    args_preview = args_preview[:120] + "..."
                steps.append(f"**Tool chamada:** `{call.name}({args_preview})`")

        if step.observations:
            obs = step.observations
            preview = obs[:400] + "..." if len(obs) > 400 else obs
            steps.append(f"**Resultado:** {preview}")

        if step.error:
            steps.append(f"**Erro:** {step.error}")

    return steps
