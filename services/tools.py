"""
Non-ML tools available to the ProjectIQ agent. Plain, testable Python
functions - swap the dataset-backed ones for a real project/supplier
database in production.
"""
from __future__ import annotations

import os

import joblib
import pandas as pd

_DATA_PATH = None
_EMBEDDINGS_PATH = None
_embeddings_cache = None


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


def _load_embeddings():
    """Lazily load the TF-IDF vectorizer + project vectors built by ml/build_embeddings.py."""
    global _EMBEDDINGS_PATH, _embeddings_cache
    if _embeddings_cache is None:
        if _EMBEDDINGS_PATH is None:
            _EMBEDDINGS_PATH = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ml", "model", "project_embeddings.joblib",
            )
        _embeddings_cache = joblib.load(_EMBEDDINGS_PATH)
    return _embeddings_cache


def find_similar_projects_semantic(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Semantic-style similarity search over past projects using TF-IDF vector
    embeddings and cosine similarity - finds projects whose free-text
    description overlaps with the query, not just exact type/size matches
    (see get_comparable_projects for that). Useful for fuzzier queries like
    "a congested urban commercial site with unreliable suppliers".

    Note: TF-IDF is a classical (term-overlap) vector embedding, not a deep
    neural embedding - it won't catch paraphrases with no shared vocabulary.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    bundle = _load_embeddings()
    vectorizer = bundle["vectorizer"]
    tfidf_matrix = bundle["tfidf_matrix"]
    metadata = bundle["metadata"]
    descriptions = bundle["descriptions"]

    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        if sims[i] <= 0:
            continue
        row = dict(metadata[i])
        row["similarity_score"] = round(float(sims[i]), 3)
        row["description"] = descriptions[i]
        results.append(row)
    return results
