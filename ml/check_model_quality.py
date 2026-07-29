"""
CI quality gate for the trained risk model. Fails if held-out MAE for
either target exceeds acceptable thresholds.

Run:
    python ml/check_model_quality.py
"""
import json
import os
import sys

MAX_ACCEPTABLE_COST_OVERRUN_MAE = 5.0   # percentage points
MAX_ACCEPTABLE_DELAY_MAE = 10.0         # days

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")


def main():
    if not os.path.exists(METRICS_PATH):
        print(f"No metrics file found at {METRICS_PATH} - did training run?")
        sys.exit(1)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    cost_mae = metrics.get("test_mae_cost_overrun_pct")
    delay_mae = metrics.get("test_mae_delay_days")
    print(f"Model quality check: cost overrun MAE={cost_mae:.2f} pct points, delay MAE={delay_mae:.2f} days")

    failed = False
    if cost_mae is None or cost_mae > MAX_ACCEPTABLE_COST_OVERRUN_MAE:
        print(f"FAIL: cost overrun MAE {cost_mae} exceeds threshold {MAX_ACCEPTABLE_COST_OVERRUN_MAE}")
        failed = True
    if delay_mae is None or delay_mae > MAX_ACCEPTABLE_DELAY_MAE:
        print(f"FAIL: delay MAE {delay_mae} exceeds threshold {MAX_ACCEPTABLE_DELAY_MAE}")
        failed = True

    if failed:
        sys.exit(1)

    print("PASS: model meets quality thresholds")


if __name__ == "__main__":
    main()
