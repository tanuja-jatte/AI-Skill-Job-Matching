"""
AI / MACHINE LEARNING COMPONENT - TF-IDF & COSINE SIMILARITY
Module: src/recommendation.py

Viva Explanation:
-----------------
What is TF-IDF?
TF-IDF stands for Term Frequency - Inverse Document Frequency.
It is a numerical statistic used in Information Retrieval and Natural Language Processing
to reflect how important a word/skill is to a document in a collection or corpus.

1. Term Frequency (TF):
   Measures how frequently a skill appears in a specific job description.
   TF(t, d) = (Number of times skill t appears in job d) / (Total skills in job d)

2. Inverse Document Frequency (IDF):
   Measures the importance of the skill across the entire job market.
   IDF(t, D) = log_e(Total number of jobs N / (1 + Number of jobs requiring skill t))
   Rare skills (e.g., PyTorch, CUDA) get higher IDF weight, while universal skills
   get lower weight.

TF-IDF Score = TF * IDF

What is Cosine Similarity?
Cosine Similarity measures the cosine of the angle between two non-zero vectors in a
multi-dimensional space:
    Similarity = (Vector A · Vector B) / (||Vector A|| * ||Vector B||)

Why Cosine Similarity instead of Euclidean Distance?
1. Length-Invariant: A student listing 4 skills can be fairly compared against a job
   listing 8 skills. Cosine similarity evaluates direction (skill overlap orientation),
   not the absolute number of skills (vector magnitude).
2. Range: Yields a neat score between 0.0 (0% - no skill match) and 1.0 (100% - perfect match).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_cleaning import standardize_skills
from src.data_structures import sort_recommended_jobs


def skill_tokenizer(text: str) -> List[str]:
    """
    Custom tokenizer for TF-IDF that splits on commas.
    This preserves multi-word skills intact:
    e.g., "machine learning", "power bi", "spring boot", "c++"
    """
    if not isinstance(text, str):
        return []
    return [s.strip().lower() for s in text.split(",") if s.strip()]


def recommend_jobs(
    student_skills: str,
    jobs_df: pd.DataFrame,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    AI / ML Recommendation Engine:
    Recommends top matching jobs for a given student skill set using
    TF-IDF Vectorization and Cosine Similarity.

    Parameters:
        student_skills (str): Comma-separated student skills
                              e.g., "Python, SQL, Pandas, Excel"
        jobs_df (pd.DataFrame): DataFrame containing job records
        top_n (int): Number of top recommendations to return (default: 5)

    Returns:
        list of dict: Top recommended job records sorted by matching score descending
    """
    # Defensive checks
    if jobs_df is None or jobs_df.empty:
        return []

    # Clean and standardize the student's entered skills
    cleaned_student_skills = standardize_skills(student_skills)
    if not cleaned_student_skills:
        return []

    student_skill_list = [s.strip() for s in cleaned_student_skills.split(",") if s.strip()]
    student_skill_set = set(student_skill_list)

    # Prepare job skills list
    job_skills_corpus = [
        standardize_skills(str(skills)) for skills in jobs_df["required_skills"]
    ]

    # Combine all job skills + student skills to build unified vocabulary
    full_corpus = job_skills_corpus + [cleaned_student_skills]

    # Initialize TF-IDF Vectorizer with our comma-based skill tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=skill_tokenizer,
        token_pattern=None,
        lowercase=True
    )

    try:
        # Fit vectorizer and transform all documents into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(full_corpus)

        # Job vectors are all rows except the last one
        job_vectors = tfidf_matrix[:-1]

        # Student vector is the very last row
        student_vector = tfidf_matrix[-1:]

        # Compute Cosine Similarity between student vector and all job vectors
        # Result shape: (1, num_jobs) -> flatten to 1D array
        similarity_scores = cosine_similarity(student_vector, job_vectors).flatten()

    except Exception as e:
        print(f"Error during TF-IDF Vectorization: {e}")
        return []

    # Build recommendations list
    recommendations: List[Dict[str, Any]] = []

    for idx, (_, row) in enumerate(jobs_df.iterrows()):
        raw_score = float(similarity_scores[idx])
        
        # Convert to percentage 0.0 - 100.0%
        matching_score = round(raw_score * 100.0, 1)

        # Analyze skill overlap: matched skills vs missing skills
        job_skill_list = [s.strip() for s in str(row.get("required_skills", "")).split(",") if s.strip()]
        job_skill_set = set([s.lower() for s in job_skill_list])

        matched = sorted(list(student_skill_set.intersection(job_skill_set)))
        missing = sorted(list(job_skill_set.difference(student_skill_set)))

        rec_item = {
            "job_id": row.get("job_id", f"JOB_{idx+1}"),
            "job_title": row.get("job_title", "Unknown Role"),
            "company": row.get("company", "Unknown Company"),
            "required_skills": row.get("required_skills", ""),
            "experience": row.get("experience", "Not Disclosed"),
            "location": row.get("location", "Remote"),
            "salary_range": row.get("salary_range", "Not Disclosed"),
            "job_description": row.get("job_description", ""),
            "matching_score": matching_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "match_ratio": f"{len(matched)} / {len(job_skill_set)} skills"
        }
        recommendations.append(rec_item)

    # DATA STRUCTURE: Use Sorting (Timsort) to rank highest matching scores first
    sorted_recommendations = sort_recommended_jobs(
        recommendations,
        key="matching_score",
        reverse=True
    )

    # Return top N recommendations
    return sorted_recommendations[:top_n]


def explain_match(student_skills: str, job_required_skills: str) -> Dict[str, Any]:
    """
    Generates a step-by-step mathematical explanation of the match for viva defense.
    """
    s_clean = standardize_skills(student_skills)
    j_clean = standardize_skills(job_required_skills)

    s_list = [s.strip() for s in s_clean.split(",") if s.strip()]
    j_list = [s.strip() for s in j_clean.split(",") if s.strip()]

    s_set = set(s_list)
    j_set = set(j_list)

    overlap = s_set.intersection(j_set)
    missing = j_set.difference(s_set)

    vectorizer = TfidfVectorizer(tokenizer=skill_tokenizer, token_pattern=None, lowercase=True)
    matrix = vectorizer.fit_transform([s_clean, j_clean])
    cos_sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return {
        "student_skills": s_list,
        "job_skills": j_list,
        "overlapping_skills": sorted(list(overlap)),
        "missing_skills": sorted(list(missing)),
        "cosine_similarity": round(float(cos_sim), 4),
        "percentage_match": round(float(cos_sim) * 100, 1),
        "feature_names": vectorizer.get_feature_names_out().tolist()
    }
