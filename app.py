from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Candidate–Job Matching Research Prototype",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "experiments" / "all_model_scores.csv"
FEATURES = ROOT / "experiments" / "structured_features.csv"
RESULTS = ROOT / "experiments" / "ranking_results_mean.csv"

st.title("Candidate–Job Matching Research Prototype")
st.caption(
    "Research demonstration using 100 synthetic CVs and five jobs. "
    "These rankings must not be used to make hiring decisions."
)

missing = [p for p in (SCORES, FEATURES, RESULTS) if not p.exists()]
if missing:
    st.error(
        "Required results files are missing. Keep app.py inside the project "
        "folder alongside the experiments directory."
    )
    st.code("\n".join(str(p.relative_to(ROOT)) for p in missing))
    st.stop()

scores = pd.read_csv(SCORES)
features = pd.read_csv(FEATURES)
summary = pd.read_csv(RESULTS)

assert len(scores) == 500, "Expected 500 candidate–job scores."

data = scores.merge(
    features[
        ["candidate_id", "job_id", "matched_required_skills",
         "missing_required_skills", "matched_preferred_skills"]
    ],
    on=["candidate_id", "job_id"],
    how="left",
    validate="one_to_one"
)

st.subheader("Overall evaluation")
st.dataframe(summary.round(4), use_container_width=True, hide_index=True)

jobs = (
    data[["job_id", "job_title"]]
    .drop_duplicates()
    .sort_values("job_id")
)
job_options = dict(zip(jobs["job_title"], jobs["job_id"]))

selected_title = st.selectbox("Choose a job", list(job_options))
selected_job = job_options[selected_title]

methods = {
    "Hybrid 60:40": "hybrid_score",
    "BM25": "bm25_score",
    "Sentence-BERT": "semantic_score",
    "Skill only": "skill_score"
}
selected_method = st.selectbox("Rank candidates by", list(methods))

ranked = (
    data[data["job_id"] == selected_job]
    .sort_values(
        [methods[selected_method], "candidate_id"],
        ascending=[False, True],
        kind="stable"
    )
    .reset_index(drop=True)
)
ranked.insert(0, "rank", range(1, len(ranked) + 1))

st.subheader(f"Rankings: {selected_title}")
st.dataframe(
    ranked[
        ["rank", "candidate_id", "final_relevance_score",
         "semantic_score", "skill_score", "bm25_score", "hybrid_score"]
    ].round(4),
    use_container_width=True,
    hide_index=True
)

selected_candidate = st.selectbox(
    "Inspect a candidate",
    ranked["candidate_id"].tolist()
)
record = ranked.loc[
    ranked["candidate_id"] == selected_candidate
].iloc[0]

def parse_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []

left, right = st.columns(2)
with left:
    st.write("**Matched required skills**")
    st.write(parse_list(record["matched_required_skills"]) or "None identified")
    st.write("**Matched preferred skills**")
    st.write(parse_list(record["matched_preferred_skills"]) or "None identified")
with right:
    st.write("**Missing required skills**")
    st.write(parse_list(record["missing_required_skills"]) or "None identified")
    st.write("**Score components**")
    st.write({
        "semantic": round(float(record["semantic_score"]), 4),
        "skill": round(float(record["skill_score"]), 4),
        "hybrid": round(float(record["hybrid_score"]), 4)
    })

with st.expander("Synthetic CV text"):
    st.write(record["cv_text"])

st.info(
    "Matched skills and component scores show part of the ranking basis. "
    "They do not fully explain Sentence-BERT's internal representation. "
    "The relevance label is a reference judgement, not the model prediction."
)
