"""
Builds TF-IDF vector embeddings over free-text descriptions of each
project in the dataset, for semantic-style similarity search.

Honest scope note: this is a classical (term-frequency) vector
embedding technique, not a deep neural embedding (e.g. Sentence-
Transformers or an embeddings API) - those require downloading
pretrained model weights from a host this environment can't reach.
TF-IDF still gives genuine vector representations and cosine-similarity
retrieval; it just matches on term overlap rather than deep semantic
meaning. Swap in a neural embedding model here if that's available in
your deployment environment - the retrieval code downstream doesn't
change.

Run:
    python ml/build_embeddings.py
Outputs:
    ml/model/project_embeddings.joblib
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "construction_projects.csv")
MODEL_DIR = os.path.join(HERE, "model")


def project_to_text(row) -> str:
    """Renders a project's structured features as a free-text description."""
    reliability_word = (
        "highly reliable" if row.supplier_reliability_score >= 0.85
        else "reliable" if row.supplier_reliability_score >= 0.65
        else "unreliable" if row.supplier_reliability_score >= 0.4
        else "very unreliable"
    )
    congestion_word = (
        "dense urban" if row.site_congestion_score >= 0.7
        else "moderately congested" if row.site_congestion_score >= 0.4
        else "open, low-congestion"
    )
    return (
        f"{row.project_type} project, {row.size_sqm:.0f} square meters, "
        f"planned budget {row.planned_budget_aed:.0f} AED over {row.planned_duration_days:.0f} days. "
        f"{row.num_suppliers} suppliers with {reliability_word} delivery history, "
        f"{row.num_subcontractors} subcontractors, {row.design_change_orders} design change orders. "
        f"Site conditions: {congestion_word} site, {row.weather_risk_days} expected weather risk days, "
        f"{row.permit_delay_days} days of permit delay, {row.labor_turnover_pct:.0f}% labor turnover, "
        f"{row.equipment_utilization_pct:.0f}% equipment utilization."
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    descriptions = [project_to_text(row) for row in df.itertuples()]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    metadata = df[
        ["project_type", "size_sqm", "planned_budget_aed", "planned_duration_days",
         "cost_overrun_pct", "delay_days"]
    ].to_dict(orient="records")

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "descriptions": descriptions,
            "metadata": metadata,
        },
        os.path.join(MODEL_DIR, "project_embeddings.joblib"),
    )
    print(f"Built TF-IDF embeddings for {len(descriptions)} projects, vocab size {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
