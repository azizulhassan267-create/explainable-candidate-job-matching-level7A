# Explainable Candidate–Job Matching System

## Project overview

This research prototype compares four methods for ranking synthetic CVs against job descriptions:

1. BM25 lexical matching
2. Sentence-BERT semantic similarity (`all-MiniLM-L6-v2`)
3. Structured skill matching
4. A hybrid of 60% semantic similarity and 40% skill matching

It is a controlled research experiment, not a system for making real hiring decisions.

## Dataset

- 100 synthetic CVs
- 5 job descriptions
- 500 candidate–job pairs
- Relevance labels from 0 (not relevant) to 3 (highly relevant)

The job areas are software engineering, data analysis, cybersecurity,
IT project management and digital marketing.

Two scored annotation spreadsheets and a disagreement-resolution spreadsheet
were used to create the final labels. The identities, expertise and actual
assessment process of the reviewers must be documented separately and accurately.

## Scoring

The structured skill score combines:

- 80% coverage of required skills
- 20% coverage of preferred skills

The hybrid score is:

`0.60 × normalised Sentence-BERT score + 0.40 × skill score`

BM25 is evaluated as a separate baseline; its raw score is not added to
the hybrid.

## Evaluation

Each method ranks the same 100 CVs separately for each job. The reported
measures are NDCG@5, NDCG@10 and Precision@5. For Precision@5, relevance
labels of 2 or 3 count as relevant.

| Method | Mean NDCG@5 | Mean NDCG@10 | Mean Precision@5 |
|---|---:|---:|---:|
| BM25 | 0.9318 | 0.9393 | 1.00 |
| Sentence-BERT | 0.8896 | 0.8803 | 0.96 |
| Skill only | 0.9716 | 0.9811 | 1.00 |
| Hybrid 60:40 | 0.9770 | 0.9774 | 1.00 |

The hybrid has the highest mean NDCG@5. Skill matching alone has the
highest mean NDCG@10. These results apply only to this synthetic dataset.

## How to reproduce

1. Open the project `.ipynb` notebook in Google Colab.
2. Select **Runtime → Run all**.
3. At the first upload prompt, upload both completed reviewer annotation
   spreadsheets together.
4. At the second upload prompt, upload the completed disagreement-resolution
   spreadsheet.
5. Run the added Sentence-BERT/BM25 scoring cell and the ranking-evaluation
   cell in order.
6. Check that the output reports **500 pairs scored** and displays results
   for all five jobs and four methods.

A fresh Colab runtime may need to download Python packages, the spaCy
English model and the Sentence-BERT model.

## Key output files

- `data/candidate_dataset.csv`: generated synthetic CV dataset
- `data/job_descriptions/job_descriptions.csv`: generated job descriptions
- `data/annotations/final_ground_truth.csv`: resolved relevance labels
- `data/skills/skill_ontology.json`: structured skill dictionary
- `experiments/structured_features.csv`: skill features
- `experiments/all_model_scores.csv`: all four scores for 500 pairs
- `experiments/ranking_results_by_job.csv`: per-job evaluation
- `experiments/ranking_results_mean.csv`: mean evaluation across five jobs

## Limitations

The CVs and jobs are synthetic and may contain simpler, clearer skill
signals than real recruitment documents. The fixed skill dictionary does
not cover every expression or occupation. The relevance labels depend
on the annotation rubric and actual reviewer process. No evaluation
with real recruiters, subgroup fairness audit or real-world hiring
validation is reported. The 60:40 weighting was a design choice and
was not established as optimal.

The rankings require human inspection and must not be used as the sole
basis for selecting or rejecting applicants.
