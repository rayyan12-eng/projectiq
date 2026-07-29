"""
Generates a synthetic but realistic construction project dataset, standing
in for real project management, BIM, supplier, and equipment-sensor data
(e.g. from a system like the one built during a contracting-company
internship). Swap this out for real historical project data in production.
"""
import numpy as np
import pandas as pd

PROJECT_TYPES = ["Residential", "Commercial", "Industrial", "Infrastructure"]

# (project type, base cost/sqft AED, base duration days per 1000 sqm, complexity multiplier)
TYPE_PROFILES = {
    "Residential": (1400, 45, 1.0),
    "Commercial": (1800, 55, 1.15),
    "Industrial": (1200, 60, 1.25),
    "Infrastructure": (2200, 90, 1.4),
}


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        ptype = rng.choice(PROJECT_TYPES, p=[0.4, 0.3, 0.2, 0.1])
        base_cost_psqm, base_duration_per_1000sqm, complexity = TYPE_PROFILES[ptype]

        size_sqm = max(200, rng.normal(3000, 1500))
        planned_budget_aed = base_cost_psqm * size_sqm * (1 + rng.normal(0, 0.05))
        planned_duration_days = max(30, (size_sqm / 1000) * base_duration_per_1000sqm * complexity)

        num_suppliers = int(rng.integers(2, 15))
        supplier_reliability_score = np.clip(rng.normal(0.75, 0.15), 0.1, 1.0)  # 0-1, historical on-time delivery rate
        num_subcontractors = int(rng.integers(1, 10))
        design_change_orders = int(rng.poisson(2 + complexity * 2))
        weather_risk_days = int(rng.poisson(5))  # expected adverse-weather days in the schedule window
        site_congestion_score = np.clip(rng.normal(0.5, 0.2), 0, 1)  # 0=remote/easy, 1=dense urban site
        equipment_utilization_pct = np.clip(rng.normal(70, 15), 20, 100)  # avg equipment-sensor utilization
        permit_delay_days = int(max(0, rng.normal(7, 5)))
        labor_turnover_pct = np.clip(rng.normal(15, 8), 0, 60)

        # ---- true generating process for cost overrun % and delay days ----
        risk_score = (
            0.35 * design_change_orders
            + 0.30 * (1 - supplier_reliability_score) * 10
            + 0.20 * site_congestion_score * 10
            + 0.15 * (labor_turnover_pct / 10)
            + 0.10 * (weather_risk_days / 2)
            + 0.05 * (num_subcontractors)
            - 0.05 * (equipment_utilization_pct / 10)
        )
        noise = rng.normal(0, 2.0)
        cost_overrun_pct = max(-5, risk_score * 1.6 + noise)
        delay_days = max(0, risk_score * 3.2 + permit_delay_days * 0.8 + rng.normal(0, 6))

        actual_cost_aed = planned_budget_aed * (1 + cost_overrun_pct / 100)
        actual_duration_days = planned_duration_days + delay_days

        rows.append(
            {
                "project_type": ptype,
                "size_sqm": round(size_sqm),
                "planned_budget_aed": round(planned_budget_aed),
                "planned_duration_days": round(planned_duration_days),
                "num_suppliers": num_suppliers,
                "supplier_reliability_score": round(supplier_reliability_score, 2),
                "num_subcontractors": num_subcontractors,
                "design_change_orders": design_change_orders,
                "weather_risk_days": weather_risk_days,
                "site_congestion_score": round(site_congestion_score, 2),
                "equipment_utilization_pct": round(equipment_utilization_pct, 1),
                "permit_delay_days": permit_delay_days,
                "labor_turnover_pct": round(labor_turnover_pct, 1),
                "cost_overrun_pct": round(cost_overrun_pct, 2),
                "delay_days": round(delay_days, 1),
                "actual_cost_aed": round(actual_cost_aed),
                "actual_duration_days": round(actual_duration_days),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("ml/construction_projects.csv", index=False)
    print(f"Wrote {len(df)} rows to ml/construction_projects.csv")
    print(df.head())
