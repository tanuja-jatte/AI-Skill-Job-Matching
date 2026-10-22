"""
DATA ENGINEERING COMPONENT - STEP 3: DATA TRANSFORMATION & FEATURE EXTRACTION
Module: src/data_transformation.py

Viva Explanation:
-----------------
What is Data Transformation?
Data transformation converts data from its cleaned storage format into a structured
format tailored for analytical processing and machine learning feature extraction.

What is Feature Extraction?
In text and skill-matching tasks, raw strings of skills cannot be fed directly to
mathematical algorithms. Feature extraction parses the text to identify individual
skill tokens and build a vocabulary vector space.
"""

import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Set


def extract_unique_skills(jobs_df: pd.DataFrame) -> List[str]:
    """
    Extracts an alphabetically sorted list of all unique skills present in the dataset.

    Returns:
        list of str: sorted unique skills
    """
    unique_skills: Set[str] = set()

    for skills_str in jobs_df["required_skills"].dropna():
        for skill in skills_str.split(","):
            s = skill.strip()
            if s:
                unique_skills.add(s)

    return sorted(list(unique_skills))


def compute_skill_frequencies(jobs_df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes frequency distribution of each skill across all jobs.
    Uses Python Counter (Hash Map).

    Returns:
        dict: {skill: occurrence_count} sorted descending by frequency
    """
    counter = Counter()

    for skills_str in jobs_df["required_skills"].dropna():
        tokens = [s.strip() for s in skills_str.split(",") if s.strip()]
        counter.update(tokens)

    # Return as ordered dictionary
    return dict(counter.most_common())


def compute_dataset_statistics(jobs_df: pd.DataFrame, students_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Calculates essential data engineering metrics and descriptive statistics.

    Returns:
        dict of summary statistics
    """
    total_jobs = len(jobs_df)
    unique_skills = extract_unique_skills(jobs_df)
    skill_counts = compute_skill_frequencies(jobs_df)

    # Calculate average number of skills per job
    skills_per_job = jobs_df["required_skills"].apply(
        lambda x: len([s for s in x.split(",") if s.strip()]) if isinstance(x, str) else 0
    )
    avg_skills_per_job = round(float(skills_per_job.mean()), 2) if total_jobs > 0 else 0

    # Location distribution
    location_dist = jobs_df["location"].value_counts().to_dict() if "location" in jobs_df else {}

    # Experience distribution
    exp_dist = jobs_df["experience"].value_counts().to_dict() if "experience" in jobs_df else {}

    # Unique companies
    unique_companies = int(jobs_df["company"].nunique()) if "company" in jobs_df else 0

    stats = {
        "total_jobs": total_jobs,
        "total_unique_skills": len(unique_skills),
        "unique_companies": unique_companies,
        "avg_skills_per_job": avg_skills_per_job,
        "location_distribution": location_dist,
        "experience_distribution": exp_dist,
        "top_10_skills": dict(list(skill_counts.items())[:10]),
        "total_students": len(students_df) if students_df is not None else 0
    }

    return stats


def prepare_skill_corpus(jobs_df: pd.DataFrame) -> List[str]:
    """
    Transforms the skills column into a clean corpus of space-separated skill tokens
    ready for TF-IDF Vectorization.
    """
    corpus = []
    for skills_str in jobs_df["required_skills"]:
        if isinstance(skills_str, str):
            # Replace comma with space so n-grams or individual words can be processed
            corpus.append(skills_str.replace(",", " "))
        else:
            corpus.append("")
    return corpus
