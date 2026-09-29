from io import BytesIO
from pathlib import Path
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
    "Research prototype. Upload a job description and one or more CVs "
    "to inspect their matches. Do not use these scores as hiring decisions."
)

# Edit this list to match the skills used in your research dataset.
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
    "seo": ["seo", "search engine optimization", "search engine optimisation"],
    "google analytics": ["google analytics"],
    "social media": ["social media"],
    "content marketing": ["content marketing"],
}


def read_file(uploaded_file):
    """Extract text from a TXT, PDF, or DOCX upload."""
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return content.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        document = Document(BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError("Please upload a TXT, PDF, or DOCX file.")


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text, phrase):
    """Match a skill as a complete phrase, ignoring case."""
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text):
    found = set()
    for skill, aliases in SKILLS.items():
        if any(contains_phrase(text, alias) for alias in aliases):
            found.add(skill)
    return found


@st.cache_resource(show_spinner="Loading Sentence-BERT model...")
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


st.subheader("1. Job description")

jd_file = st.file_uploader(
    "Upload the job description",
    type=["txt", "pdf", "docx"],
    key="job_description",
)
jd_typed = st.text_area(
    "Or paste the job description here",
    height=180,
    placeholder="Paste the job title, responsibilities, and required skills...",
)

st.subheader("2. Candidate CVs")

cv_files = st.file_uploader(
    "Upload one CV, or several CVs to compare",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    key="candidate_cvs",
)

if st.button("Match candidates", type="primary"):
    try:
        job_text = clean_text(
            jd_typed if jd_typed.strip() else read_file(jd_file)
        ) if (jd_typed.strip() or jd_file is not None) else ""

        if not job_text:
            st.error("Enter or upload a job description.")
            st.stop()

        if not cv_files:
            st.error("Upload at least one CV.")
            st.stop()

        candidates = []
        for cv_file in cv_files:
            cv_text = clean_text(read_file(cv_file))
            if not cv_text:
                st.warning(
                    f"No readable text was found in {cv_file.name}. "
                    "Scanned PDFs may need OCR."
                )
                continue

            candidates.append(
                {"filename": cv_file.name, "text": cv_text}
            )

        if not candidates:
            st.error("None of the uploaded CVs contained readable text.")
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
        results = []

        for candidate, cv_embedding in zip(candidates, cv_embeddings):
            cv_skills = extract_skills(candidate["text"])
            matched = sorted(job_skills & cv_skills)
            missing = sorted(job_skills - cv_skills)

            # Normalized embeddings make their dot product cosine similarity.
            cosine = float(job_embedding @ cv_embedding)

            # Keep the displayed semantic component between 0 and 1.
            semantic_score = max(0.0, min(1.0, cosine))

            # If the job description contains no recognized skills,
            # do not invent a skill match.
            skill_score = (
                len(matched) / len(job_skills)
                if job_skills
                else None
            )

            if skill_score is None:
                combined_score = semantic_score
            else:
                combined_score = (
                    0.60 * semantic_score + 0.40 * skill_score
                )

            results.append(
                {
                    "CV": candidate["filename"],
                    "Match score": round(100 * combined_score, 1),
                    "Semantic score": round(100 * semantic_score, 1),
                    "Skill score": (
                        round(100 * skill_score, 1)
                        if skill_score is not None
                        else None
                    ),
                    "Matched skills": matched,
                    "Missing skills": missing,
                }
            )

        results.sort(
            key=lambda item: (-item["Match score"], item["CV"].lower())
        )

        st.subheader("3. Results")
        st.write(
            f"**Job skills identified:** "
            f"{', '.join(sorted(job_skills)) if job_skills else 'None'}"
        )

        if not job_skills:
            st.info(
                "No skills from the app's skill list were identified in "
                "the job description. The match score therefore uses "
                "semantic similarity only."
            )

        if len(results) == 1:
            result = results[0]
            st.metric("CV match score", f"{result['Match score']:.1f}%")
            st.write(f"**CV:** {result['CV']}")
        else:
            display_rows = []
            for rank, result in enumerate(results, start=1):
                display_rows.append(
                    {
                        "Rank": rank,
                        "CV": result["CV"],
                        "Match score (%)": result["Match score"],
                        "Semantic score (%)": result["Semantic score"],
                        "Skill score (%)": result["Skill score"],
                    }
                )
            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
            )

        for result in results:
            with st.expander(
                f"{result['CV']} — {result['Match score']:.1f}% match",
                expanded=len(results) == 1,
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Matched job skills**")
                    st.write(
                        ", ".join(result["Matched skills"])
                        or "None identified"
                    )

                with col2:
                    st.write("**Job skills not identified in this CV**")
                    st.write(
                        ", ".join(result["Missing skills"])
                        or "None identified"
                    )

                st.write(
                    f"**Semantic similarity:** "
                    f"{result['Semantic score']:.1f}%"
                )
                if result["Skill score"] is not None:
                    st.write(
                        f"**Skill overlap:** "
                        f"{result['Skill score']:.1f}%"
                    )

        export_rows = []
        for rank, result in enumerate(results, start=1):
            export_rows.append(
                {
                    "rank": rank,
                    "cv": result["CV"],
                    "match_score_percent": result["Match score"],
                    "semantic_score_percent": result["Semantic score"],
                    "skill_score_percent": result["Skill score"],
                    "matched_skills": "; ".join(result["Matched skills"]),
                    "missing_skills": "; ".join(result["Missing skills"]),
                }
            )

        st.download_button(
            "Download results as CSV",
            data=pd.DataFrame(export_rows).to_csv(index=False),
            file_name="candidate_matching_results.csv",
            mime="text/csv",
        )

        st.info(
            "The match score combines semantic similarity (60%) and "
            "recognized skill overlap (40%). It is a research score, "
            "not a probability that someone is qualified. A missing "
            "skill means the app did not find its listed terms in the CV; "
            "it does not prove the candidate lacks that skill."
        )

    except Exception as exc:
        st.error(f"Could not process the uploads: {exc}")
