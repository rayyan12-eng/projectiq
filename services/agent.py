"""
ProjectIQ Agent
---------------
Given a natural-language question about a construction project, the
agent (Claude, via tool-use) decides which tools to call - the
TensorFlow-backed risk model, comparable past projects, material cost
estimates, supplier reliability stats - and synthesizes a grounded
answer, e.g. "Should we bid on this project? What's our realistic
cost/schedule risk?"

Requires ANTHROPIC_API_KEY to be set in the environment to run live.
"""
import json
import os

import httpx
from anthropic import Anthropic

from services import tools

RISK_SERVICE_URL = os.environ.get("RISK_SERVICE_URL", "http://localhost:8002")

SYSTEM_PROMPT = """You are ProjectIQ, a construction project risk advisor for a contracting company.

You have tools to: get an ML-based cost overrun / schedule delay risk prediction for a project,
look up comparable past projects, estimate material costs, and pull supplier reliability stats.
Use as many tools as needed - typically a risk prediction, then comparables and/or supplier stats
to sanity-check it, and a material cost estimate if the user is scoping a bid - before answering.

Always ground your recommendation in the actual tool outputs (cite the numbers). Be direct about
uncertainty - the risk model gives an estimate, not a guarantee. Keep the final answer concise
and structured: risk verdict, key numbers, and 2-3 sentences of reasoning."""

TOOL_DEFINITIONS = [
    {
        "name": "predict_risk",
        "description": "Get a TensorFlow ML-based prediction of cost overrun % and schedule delay (days) for a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_type": {"type": "string", "enum": ["Residential", "Commercial", "Industrial", "Infrastructure"]},
                "size_sqm": {"type": "number"},
                "planned_budget_aed": {"type": "number"},
                "planned_duration_days": {"type": "number"},
                "num_suppliers": {"type": "integer"},
                "supplier_reliability_score": {"type": "number"},
                "num_subcontractors": {"type": "integer", "default": 0},
                "design_change_orders": {"type": "integer", "default": 0},
                "weather_risk_days": {"type": "integer", "default": 0},
                "site_congestion_score": {"type": "number", "default": 0.5},
                "equipment_utilization_pct": {"type": "number", "default": 70.0},
                "permit_delay_days": {"type": "integer", "default": 0},
                "labor_turnover_pct": {"type": "number", "default": 15.0},
            },
            "required": [
                "project_type", "size_sqm", "planned_budget_aed", "planned_duration_days",
                "num_suppliers", "supplier_reliability_score",
            ],
        },
    },
    {
        "name": "get_comparable_projects",
        "description": "Look up similar past projects by type and size to sanity-check a risk prediction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_type": {"type": "string"},
                "size_sqm": {"type": "number"},
            },
            "required": ["project_type", "size_sqm"],
        },
    },
    {
        "name": "estimate_material_costs",
        "description": "Get a rough material cost breakdown (concrete, steel, finishing, MEP) for a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "size_sqm": {"type": "number"},
                "project_type": {"type": "string"},
                "material_price_index": {"type": "number", "default": 1.0},
            },
            "required": ["size_sqm", "project_type"],
        },
    },
    {
        "name": "get_supplier_reliability_stats",
        "description": "Get aggregate historical stats on cost/delay outcomes filtered by supplier reliability.",
        "input_schema": {
            "type": "object",
            "properties": {"min_reliability": {"type": "number", "default": 0.0}},
        },
    },
]


def _call_risk_service(args: dict) -> dict:
    resp = httpx.post(f"{RISK_SERVICE_URL}/predict", json=args, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _execute_tool(name: str, args: dict) -> dict:
    if name == "predict_risk":
        return _call_risk_service(args)
    if name == "get_comparable_projects":
        return {"comparable_projects": tools.get_comparable_projects(**args)}
    if name == "estimate_material_costs":
        return tools.estimate_material_costs(**args)
    if name == "get_supplier_reliability_stats":
        return tools.get_supplier_reliability_stats(**args)
    raise ValueError(f"Unknown tool: {name}")


def run_agent(user_query: str, max_turns: int = 6, model: str = "claude-sonnet-4-6") -> str:
    """Runs the full agent loop: plan -> call tools -> observe -> repeat -> answer."""
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    messages = [{"role": "user", "content": user_query}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = _execute_tool(block.name, block.input)
                content = json.dumps(result)
            except Exception as exc:
                content = json.dumps({"error": str(exc)})

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        messages.append({"role": "user", "content": tool_results})

    return "Reached max tool-use turns without a final answer."


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or (
        "We're bidding on a 3000 sqm Commercial project with a planned budget of AED 5,400,000 "
        "and 180 planned duration days. We'd use 8 suppliers with historical reliability around 0.7, "
        "and expect 3 design change orders. What's our realistic cost and schedule risk?"
    )
    print(run_agent(query))
