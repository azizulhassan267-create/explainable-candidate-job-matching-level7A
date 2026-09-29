from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Candidate–Job Matching Research Prototype",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

# The CSV files are beside app.py in the GitHub repository.
SCORES_PATH = ROOT / "all_model_scores.csv"
FEATURES_PATH = ROOT / "structured_features.csv"
SUMMARY_PATH = ROOT / "ranking_results_mean.csv"


def parse_skill_list(value):
    """Read a JSON list stored in a CSV cell."""
    if pd.isna(value):
        return []

    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (TypeError, ValueError):
        return []


@st.cache_data
def load_data():
    missing = [
        path.name
        for path in (SCORES_PATH, FEATURES_PATH, SUMMARY_PATH)
        if not path.is_file()
    ]

    if missing:
        return None, None, missing

    scores = pd.read_csv(SCORES_PATH)
    features = pd.read_csv(FEATURES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    skill_columns = [
        "candidate_id",
        "job_id",
        "matched_required_skills",
        "missing_required_skills",
        "matched_preferred_skills",
    ]

    data = scores.merge(
        features[skill_columns],
        on=["candidate_id", "job_id"],
        how="left",
        validate="one_to_one",
    )

    return data, summary, []


st.title("Candidate–Job Matching Research Prototype")
st.caption(
    "Demonstration using 100 synthetic CVs and five jobs. "
    "The rankings must not be used to make hiring decisions."
)

try:
    data, summary, missing_files = load_data()
except Exception as exc:
    st.error("The results files could not be read.")
    st.exception(exc)
    st.stop()

if missing_files:
    st.error("These required files are missing from the GitHub repository:")
    st.code("\n".join(missing_files))
    st.stop()

if len(data) != 500:
    st.error(
        f"Expected 500 candidate–job pairs, but found {len(data)}. "
        "Check that the correct CSV files were uploaded."
    )
    st.stop()

st.subheader("Overall evaluation")
st.dataframe(
    summary.round(4),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "These are mean results across five synthetic jobs. "
    "The hybrid leads on NDCG@5; skill matching leads on NDCG@10."
)

jobs = (
    data[["job_id", "job_title"]]
    .drop_duplicates()
    .sort_values("job_id")
)
job_options = dict(zip(jobs["job_title"], jobs["job_id"]))

selected_title = st.selectbox(
    "Choose a job",
    options=list(job_options.keys()),
)
selected_job_id = job_options[selected_title]

methods = {
    "Hybrid 60:40": "hybrid_score",
    "BM25": "bm25_score",
    "Sentence-BERT": "semantic_score",
    "Skill only": "skill_score",
}

selected_method = st.selectbox(
    "Rank candidates by",
    options=list(methods.keys()),
)
score_column = methods[selected_method]

ranked = (
    data.loc[data["job_id"] == selected_job_id]
    .sort_values(
        by=[score_column, "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )
    .reset_index(drop=True)
    .copy()
)
ranked.insert(0, "rank", range(1, len(ranked) + 1))

st.subheader(f"Candidate rankings: {selected_title}")

display_columns = [
    "rank",
    "candidate_id",
    "final_relevance_score",
    "semantic_score",
    "skill_score",
    "bm25_score",
    "hybrid_score",
]

st.dataframe(
    ranked[display_columns].round(4),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "The relevance score is the reference label from the annotation files. "
    "It is not a prediction made by the selected ranking method."
)

selected_candidate = st.selectbox(
    "Inspect a candidate",
    options=ranked["candidate_id"].tolist(),
)

record = ranked.loc[
    ranked["candidate_id"] == selected_candidate
].iloc[0]

st.subheader(f"Evidence for {selected_candidate}")

left, right = st.columns(2)

with left:
    st.write("**Matched required skills**")
    st.write(
        parse_skill_list(record["matched_required_skills"])
        or "None identified"
    )

    st.write("**Matched preferred skills**")
    st.write(
        parse_skill_list(record["matched_preferred_skills"])
        or "None identified"
    )

with right:
    st.write("**Missing required skills**")
    st.write(
        parse_skill_list(record["missing_required_skills"])
        or "None identified"
    )

    st.write("**Score components**")
    st.write(
        {
            "semantic score": round(float(record["semantic_score"]), 4),
            "skill score": round(float(record["skill_score"]), 4),
            "hybrid score": round(float(record["hybrid_score"]), 4),
        }
    )

with st.expander("View synthetic CV text"):
    st.write(record["cv_text"])

st.info(
    "The app displays saved results from the Colab experiment. "
    "It does not score newly uploaded CVs. Matched skills and component "
    "scores provide partial evidence, but do not fully explain the "
    "Sentence-BERT model."
)
