"""
Non-ML tools available to the ProjectIQ agent. Plain, testable Python
functions - swap the dataset-backed ones for a real project/supplier
database in production.
"""
from __future__ import annotations

import pandas as pd

_DATA_PATH = None


def _load_dataset():
    import os

    global _DATA_PATH
    if _DATA_PATH is None:
        _DATA_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "construction_projects.csv"
        )
    return pd.read_csv(_DATA_PATH)


def get_comparable_projects(project_type: str, size_sqm: float, tolerance_pct: float = 25.0, limit: int = 5) -> list[dict]:
    """Find similar past projects by type and size, for sanity-checking a risk prediction."""
    df = _load_dataset()
    low = size_sqm * (1 - tolerance_pct / 100)
    high = size_sqm * (1 + tolerance_pct / 100)
    subset = df[(df["project_type"] == project_type) & (df["size_sqm"].between(low, high))]

    if subset.empty:
        return []

    sample = subset.sample(min(limit, len(subset)), random_state=1)
    return sample[
        [
            "project_type", "size_sqm", "planned_budget_aed", "planned_duration_days",
            "cost_overrun_pct", "delay_days",
        ]
    ].to_dict(orient="records")


def estimate_material_costs(size_sqm: float, project_type: str, material_price_index: float = 1.0) -> dict:
    """
    Rough material cost breakdown estimate based on project type and size.
    material_price_index: multiplier for current market prices vs. baseline (e.g. 1.1 = 10% above baseline).
    """
    baselines = {
        "Residential": {"concrete": 220, "steel": 180, "finishing": 260, "mep": 140},
        "Commercial": {"concrete": 260, "steel": 220, "finishing": 300, "mep": 200},
        "Industrial": {"concrete": 300, "steel": 260, "finishing": 120, "mep": 180},
        "Infrastructure": {"concrete": 380, "steel": 320, "finishing": 60, "mep": 100},
    }
    if project_type not in baselines:
        raise ValueError(f"Unknown project_type '{project_type}'")

    breakdown = {
        material: round(cost_per_sqm * size_sqm * material_price_index, -2)
        for material, cost_per_sqm in baselines[project_type].items()
    }
    breakdown["total_material_cost_aed"] = round(sum(breakdown.values()), -2)
    return breakdown


def get_supplier_reliability_stats(min_reliability: float = 0.0) -> dict:
    """Aggregate historical supplier reliability stats across past projects."""
    df = _load_dataset()
    subset = df[df["supplier_reliability_score"] >= min_reliability]
    if subset.empty:
        return {"error": "No projects match the given reliability threshold"}

    return {
        "sample_size": int(len(subset)),
        "median_supplier_reliability_score": round(float(subset["supplier_reliability_score"].median()), 2),
        "median_cost_overrun_pct": round(float(subset["cost_overrun_pct"].median()), 2),
        "median_delay_days": round(float(subset["delay_days"].median()), 1),
    }
