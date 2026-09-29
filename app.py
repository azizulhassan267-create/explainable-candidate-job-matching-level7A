from io import BytesIO
import re

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Explainable Candidate–Job Matching",
    layout="wide",
)

st.title("Explainable Candidate–Job Matching")
st.caption(
    "Research prototype: upload a job description and one or more CVs. "
    "Results are for demonstration, not hiring decisions."
)


# Skill names and alternative terms recognized by this prototype.
SKILLS = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "java script"],
    "typescript": ["typescript"],
    "sql": ["sql"],
    "excel": ["excel", "microsoft excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "data analysis": ["data analysis", "data analytics"],
    "data visualization": ["data visualization", "data visualisation"],
    "statistics": ["statistics", "statistical analysis"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes"],
    "git": ["git", "github"],
    "linux": ["linux"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs"],
    "html": ["html"],
    "css": ["css"],
    "rest api": ["rest api", "restful api", "rest apis"],
    "cybersecurity": ["cybersecurity", "cyber security"],
    "network security": ["network security"],
    "penetration testing": ["penetration testing", "pentesting"],
    "risk assessment": ["risk assessment"],
    "project management": ["project management"],
    "agile": ["agile"],
    "scrum": ["scrum"],
    "stakeholder management": ["stakeholder management"],
    "communication": ["communication skills", "written communication"],
    "digital marketing": ["digital marketing"],
    "seo": [
        "seo",
        "search engine optimization",
        "search engine optimisation",
    ],
    "google analytics": ["google analytics"],
    "social media": ["social media"],
    "content marketing": ["content marketing"],
}


def read_file(uploaded_file):
    """Extract text from a TXT, PDF, or DOCX file."""
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return content.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if name.endswith(".docx"):
        document = Document(BytesIO(content))
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    raise ValueError("Unsupported file type. Use TXT, PDF, or DOCX.")


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text, phrase):
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def extract_skills(text):
    found = set()

    for skill, aliases in SKILLS.items():
        if any(
            contains_phrase(text, alias)
            for alias in aliases
        ):
            found.add(skill)

    return found


@st.cache_resource(show_spinner="Loading Sentence-BERT model...")
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------- INPUTS --------------------

st.subheader("1. Job description")

jd_file = st.file_uploader(
    "Upload a job description",
    type=["txt", "pdf", "docx"],
    key="job_description",
)

jd_typed = st.text_area(
    "Or paste the job description here",
    height=170,
    placeholder="Paste the job title, responsibilities, and required skills...",
)

st.subheader("2. Candidate CVs")

cv_files = st.file_uploader(
    "Upload one CV, or several CVs to compare",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    key="candidate_cvs",
)

st.subheader("3. Scoring weights")

semantic_weight_percent = st.slider(
    "Sentence-BERT semantic weight (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="The remaining percentage is assigned to skill matching.",
)

skill_weight_percent = 100 - semantic_weight_percent

st.write(
    f"**Current ratio:** {semantic_weight_percent}% Sentence-BERT "
    f"+ {skill_weight_percent}% skill matching"
)


# -------------------- MATCHING --------------------

if st.button("Match candidates", type="primary"):
    try:
        if jd_typed.strip():
            job_text = clean_text(jd_typed)
        elif jd_file is not None:
            job_text = clean_text(read_file(jd_file))
        else:
            job_text = ""

        if not job_text:
            st.error("Please paste or upload a job description.")
            st.stop()

        if not cv_files:
            st.error("Please upload at least one CV.")
            st.stop()

        candidates = []

        for cv_file in cv_files:
            cv_text = clean_text(read_file(cv_file))

            if not cv_text:
                st.warning(
                    f"No readable text was found in {cv_file.name}. "
                    "A scanned PDF may require OCR."
                )
                continue

            candidates.append(
                {
                    "filename": cv_file.name,
                    "text": cv_text,
                }
            )

        if not candidates:
            st.error(
                "No readable text was found in the uploaded CVs."
            )
            st.stop()

        model = load_model()

        job_embedding = model.encode(
            job_text,
            normalize_embeddings=True,
        )

        cv_embeddings = model.encode(
            [candidate["text"] for candidate in candidates],
            normalize_embeddings=True,
        )

        job_skills = extract_skills(job_text)
        semantic_weight = semantic_weight_percent / 100
        skill_weight = skill_weight_percent / 100

        results = []

        for candidate, cv_embedding in zip(
            candidates,
            cv_embeddings,
        ):
            cv_skills = extract_skills(candidate["text"])

            matched_skills = sorted(
                job_skills & cv_skills
            )
            missing_skills = sorted(
                job_skills - cv_skills
            )

            # Dot product of normalized embeddings is cosine similarity.
            cosine_similarity = float(
                job_embedding @ cv_embedding
            )

            # Bound the displayed score to the range 0–1.
            semantic_score = max(
                0.0,
                min(1.0, cosine_similarity),
            )

            if job_skills:
                skill_score = (
                    len(matched_skills) / len(job_skills)
                )

                combined_score = (
                    semantic_weight * semantic_score
                    + skill_weight * skill_score
                )
            else:
                skill_score = None
                combined_score = semantic_score

            results.append(
                {
                    "cv": candidate["filename"],
                    "combined_score": combined_score,
                    "semantic_score": semantic_score,
                    "skill_score": skill_score,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                }
            )

        results.sort(
            key=lambda result: (
                -result["combined_score"],
                result["cv"].lower(),
            )
        )

        # -------------------- RESULTS --------------------

        st.subheader("4. Results")

        st.write(
            "**Job skills identified:** "
            + (
                ", ".join(sorted(job_skills))
                if job_skills
                else "None from the app's skill list"
            )
        )

        if not job_skills:
            st.info(
                "No listed skills were identified in the job description. "
                "These results use semantic similarity only."
            )

        if len(results) == 1:
            result = results[0]

            st.metric(
                "CV match score",
                f"{result['combined_score'] * 100:.1f}%",
            )
            st.write(f"**CV:** {result['cv']}")

        else:
            table_rows = []

            for rank, result in enumerate(
                results,
                start=1,
            ):
                table_rows.append(
                    {
                        "Rank": rank,
                        "CV": result["cv"],
                        "Match score (%)": round(
                            result["combined_score"] * 100,
                            1,
                        ),
                        "Semantic score (%)": round(
                            result["semantic_score"] * 100,
                            1,
                        ),
                        "Skill score (%)": (
                            round(
                                result["skill_score"] * 100,
                                1,
                            )
                            if result["skill_score"] is not None
                            else None
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

        for result in results:
            with st.expander(
                f"{result['cv']} — "
                f"{result['combined_score'] * 100:.1f}% match",
                expanded=len(results) == 1,
            ):
                left, right = st.columns(2)

                with left:
                    st.write("**Matched job skills**")
                    st.write(
                        ", ".join(result["matched_skills"])
                        or "None identified"
                    )

                with right:
                    st.write(
                        "**Job skills not identified in this CV**"
                    )
                    st.write(
                        ", ".join(result["missing_skills"])
                        or "None identified"
                    )

                st.write(
                    "**Sentence-BERT semantic score:** "
                    f"{result['semantic_score'] * 100:.1f}%"
                )

                if result["skill_score"] is not None:
                    st.write(
                        "**Skill overlap score:** "
                        f"{result['skill_score'] * 100:.1f}%"
                    )

        export_rows = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            export_rows.append(
                {
                    "rank": rank,
                    "cv": result["cv"],
                    "semantic_weight_percent": (
                        semantic_weight_percent
                    ),
                    "skill_weight_percent": (
                        skill_weight_percent
                    ),
                    "match_score_percent": round(
                        result["combined_score"] * 100,
                        2,
                    ),
                    "semantic_score_percent": round(
                        result["semantic_score"] * 100,
                        2,
                    ),
                    "skill_score_percent": (
                        round(
                            result["skill_score"] * 100,
                            2,
                        )
                        if result["skill_score"] is not None
                        else None
                    ),
                    "matched_skills": "; ".join(
                        result["matched_skills"]
                    ),
                    "missing_skills": "; ".join(
                        result["missing_skills"]
                    ),
                }
            )

        csv_data = pd.DataFrame(
            export_rows
        ).to_csv(index=False)

        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name="candidate_matching_results.csv",
            mime="text/csv",
        )

        st.info(
            f"Selected weights: {semantic_weight_percent}% "
            f"Sentence-BERT and {skill_weight_percent}% skill matching. "
            "If no listed job skills are found, the score uses semantic "
            "similarity alone. Scores are research indicators, not "
            "probabilities of candidate suitability. A skill being "
            "'not identified' does not prove that a candidate lacks it."
        )

    except Exception as exc:
        st.error(
            f"Could not process the uploaded files: {exc}"
        )
