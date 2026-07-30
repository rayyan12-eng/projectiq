# ProjectIQ — Agentic AI Construction Project Risk Assistant

https://projectiq-risk.fly.dev/

An agentic AI assistant for construction/contracting companies. Given a project's
planning-stage details (size, budget, suppliers, subcontractors, site conditions),
it predicts likely cost overrun and schedule delay risk, and can pull comparable
past projects, material cost estimates, and supplier reliability stats to support
a bid/go decision.

## Why this exists

Built as an extension of hands-on experience centralizing project, BIM, supplier,
and equipment-sensor data into unified reporting (Apache Spark + Power BI) during
a Data Analyst internship at a building contracting company. This project adds a
formal ML + agentic layer on top of that same kind of data: instead of static
dashboards, an LLM (Claude) decides which tools to call - a risk model, comparables,
cost estimates, supplier stats - based on the actual question a project manager asks.

## Architecture

```
User query ("Should we bid on this 3000 sqm commercial project?")
        │
        ▼
  services/agent.py  ── Claude tool-use loop (the agentic layer)
        │
        ├── predict_risk ──────────► services/risk_service.py (FastAPI)
        │                                    │
        │                                    ▼
        │                            ml/train_model.py (TensorFlow multi-output model:
        │                                                cost overrun % + delay days)
        │
        ├── get_comparable_projects ───► services/tools.py
        ├── estimate_material_costs ───► services/tools.py
        └── get_supplier_reliability_stats ─► services/tools.py
        │
        ▼
  Final synthesized answer (risk verdict + numbers + reasoning)
```

**Why this counts as "agentic," not just an LLM call:** the model plans multi-step
tool sequences on its own (e.g. risk prediction -> comparables -> supplier stats),
observes each tool's output, and can call further tools before answering - a real
tool-use loop (see `run_agent()` in `services/agent.py`), not a fixed pipeline.

## Stack

| Requirement | Where it lives |
|---|---|
| TensorFlow | `ml/train_model.py` - a Keras multi-output regression model predicting cost overrun % and delay days |
| LLM / Agentic AI | `services/agent.py` - Claude's tool-use API, no framework (LangChain etc.) needed |
| CI/CD pipeline | `.github/workflows/ci-cd.yml` - test -> train -> **model quality gate** -> Docker build -> push -> deploy |
| Python | FastAPI (`services/risk_service.py`), plain Python tools (`services/tools.py`) |

## Running it locally

```bash
pip install -r requirements.txt

# 1. Generate the (synthetic, stand-in) dataset and train the model
python ml/generate_data.py
python ml/train_model.py

# 2. Run the tests
python -m pytest tests/ -v

# 3. Start the risk microservice
uvicorn services.risk_service:app --reload --port 8002

# 4. In another terminal, run the agent (needs ANTHROPIC_API_KEY set)
export ANTHROPIC_API_KEY=sk-...
python services/agent.py "We're bidding on a 3000 sqm Commercial project, budget AED 5,400,000, 180 planned days, 8 suppliers at 0.7 reliability. What's our risk?"
```

## CI/CD pipeline (`.github/workflows/ci-cd.yml`)

1. **test** - installs deps, generates data, trains the model, runs the full pytest suite.
2. **model-quality-gate** - retrains and checks held-out MAE for both cost overrun % and
   delay days against thresholds (`ml/check_model_quality.py`) - a bad retrain never
   proceeds to build/deploy.
3. **build-and-push** - (main branch only) builds the Docker image and pushes it to
   GitHub Container Registry, tagged with the commit SHA.
4. **deploy** - deploys the image to Fly.io (`fly.toml`), requires a `FLY_API_TOKEN`
   repo secret from your own Fly.io account.

## Notes on the data

`ml/generate_data.py` generates a synthetic-but-realistic construction project dataset
(cost/duration by project type, supplier reliability effects, design change orders,
site congestion, weather risk, permit delays) so the whole pipeline is runnable without
an external data dependency. Swap it for real historical project data (the kind pulled
from project management, BIM, supplier, and equipment-sensor systems) by replacing the
CSV `ml/train_model.py` reads from - the training and serving code doesn't change.

## Known limitations (worth stating honestly in an interview)

- The dataset is synthetic; real-world cost/delay drivers (labor disputes, regulatory
  changes, currency fluctuations on imported materials) aren't modeled.
- `get_comparable_projects` / `get_supplier_reliability_stats` read from a static CSV
  standing in for a real project/supplier database.
- `estimate_material_costs` uses simplified per-sqm baselines, not live market pricing
  or a real bill-of-quantities breakdown.
- The risk model's cost/delay outputs are point estimates without a formal confidence
  interval - in production this would warrant a proper uncertainty quantification method.


  <img width="1392" height="127" alt="image" src="https://github.com/user-attachments/assets/2780f476-5e92-4da7-9aeb-770c85cf585f" />
  <img width="1572" height="276" alt="image" src="https://github.com/user-attachments/assets/9d3df1bb-a9ac-4939-b48c-f7254940e624" />
  <img width="1576" height="262" alt="image" src="https://github.com/user-attachments/assets/e45ba6a9-621f-4733-9ff9-251cecc90494" />
  


  

