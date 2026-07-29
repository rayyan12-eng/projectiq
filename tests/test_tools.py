import pytest

from services import tools


def test_get_comparable_projects_shape():
    results = tools.get_comparable_projects(project_type="Residential", size_sqm=3000, limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for row in results:
        assert row["project_type"] == "Residential"


def test_get_comparable_projects_no_match():
    results = tools.get_comparable_projects(project_type="Residential", size_sqm=0.001, tolerance_pct=1)
    assert results == []


def test_estimate_material_costs_valid():
    result = tools.estimate_material_costs(size_sqm=1000, project_type="Commercial")
    assert result["total_material_cost_aed"] > 0
    assert "concrete" in result and "steel" in result


def test_estimate_material_costs_invalid_type():
    with pytest.raises(ValueError):
        tools.estimate_material_costs(size_sqm=1000, project_type="NotAType")


def test_get_supplier_reliability_stats():
    stats = tools.get_supplier_reliability_stats(min_reliability=0.5)
    assert stats["sample_size"] > 0
    assert 0 <= stats["median_supplier_reliability_score"] <= 1


def test_get_supplier_reliability_stats_no_match():
    stats = tools.get_supplier_reliability_stats(min_reliability=1.5)
    assert "error" in stats
