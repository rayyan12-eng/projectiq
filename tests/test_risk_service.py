from fastapi.testclient import TestClient

from services.risk_service import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload():
    payload = {
        "project_type": "Commercial",
        "size_sqm": 3000,
        "planned_budget_aed": 5_400_000,
        "planned_duration_days": 180,
        "num_suppliers": 8,
        "supplier_reliability_score": 0.7,
        "num_subcontractors": 4,
        "design_change_orders": 3,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_final_cost_aed"] > 0
    assert body["predicted_final_duration_days"] >= payload["planned_duration_days"]
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_unknown_project_type_rejected():
    payload = {
        "project_type": "Spaceport",
        "size_sqm": 1000,
        "planned_budget_aed": 1_000_000,
        "planned_duration_days": 90,
        "num_suppliers": 3,
        "supplier_reliability_score": 0.8,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400


def test_predict_invalid_reliability_rejected():
    payload = {
        "project_type": "Commercial",
        "size_sqm": 1000,
        "planned_budget_aed": 1_000_000,
        "planned_duration_days": 90,
        "num_suppliers": 3,
        "supplier_reliability_score": 1.5,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
