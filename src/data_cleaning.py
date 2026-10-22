"""
DATA ENGINEERING COMPONENT - STEP 2: DATA CLEANING
Module: src/data_cleaning.py

Viva Explanation:
-----------------
What is Data Cleaning?
Data cleaning is the process of detecting and correcting (or removing) corrupt,
inaccurate, duplicate, or irrelevant records from raw data.

Why is Data Cleaning crucial for Machine Learning?
"Garbage In, Garbage Out": If raw skills contain inconsistent cases (e.g. 'Python' vs 'python'),
extra spaces ('  SQL  '), or duplicate job listings, the TF-IDF vectorizer will treat them as
completely different tokens. This damages cosine similarity accuracy. Cleaning guarantees
normalized, consistent inputs for the AI model.
"""

import pandas as pd
from typing import Tuple, Dict, Any, List


def standardize_skills(skill_text: Any) -> str:
    """
    Standardizes a comma-separated skill string.

    Transformation Rules:
    1. Converts everything to lowercase ("Python", "PYTHON" -> "python")
    2. Strips leading and trailing whitespaces ("  SQL  " -> "sql")
    3. Removes empty tokens and deduplicates skills per record while preserving order
    4. Re-joins into a consistent comma-separated format: "python, sql, pandas"

    Example:
        Input:  " Python , PYTHON, pandas ,  sql  "
        Output: "python, pandas, sql"
    """
    if not isinstance(skill_text, str) or not skill_text.strip():
        return ""

    tokens = [t.strip().lower() for t in skill_text.split(",") if t.strip()]
    
    # Remove duplicates within the single record while preserving order
    seen = set()
    deduped = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)

    return ", ".join(deduped)


def clean_jobs_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs complete data cleaning on the jobs dataframe.

    Steps:
    1. Detect and handle missing values
    2. Remove duplicate records
    3. Strip unnecessary whitespace from all text columns
    4. Standardize skill names into lowercase consistent tokens

    Returns:
        tuple: (cleaned_df, cleaning_report)
    """
    if raw_df.empty:
        return raw_df, {"error": "Input dataframe is empty"}

    df = raw_df.copy()
    initial_rows = len(df)

    # Step 1: Detect missing values
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())

    # Fill critical missing fields with sensible defaults or drop
    df["required_skills"] = df["required_skills"].fillna("")
    df["company"] = df["company"].fillna("Unknown Company")
    df["location"] = df["location"].fillna("Remote")
    df["experience"] = df["experience"].fillna("0-2 years")
    df["salary_range"] = df["salary_range"].fillna("Not Disclosed")
    df["job_description"] = df["job_description"].fillna("No description provided.")

    # Step 2: Strip trailing/leading spaces across all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Step 3: Remove duplicate records
    # Check duplicates based on (job_title, company) or identical job_id
    duplicates_count = int(df.duplicated(subset=["job_title", "company"]).sum())
    df = df.drop_duplicates(subset=["job_title", "company"], keep="first")
    df = df.reset_index(drop=True)

    # Step 4: Standardize skills
    df["required_skills"] = df["required_skills"].apply(standardize_skills)

    cleaned_rows = len(df)

    report = {
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_removed": duplicates_count,
        "missing_cells_imputed": total_nulls,
        "null_counts_by_col": null_counts,
        "standardization": "Applied lowercase, trimmed whitespace, deduplicated skill tokens."
    }

    return df, report


def clean_students_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data cleaning on the student profile dataframe.

    Returns:
        tuple: (cleaned_df, cleaning_report)
    """
    if raw_df.empty:
        return raw_df, {"error": "Input dataframe is empty"}

    df = raw_df.copy()
    initial_rows = len(df)

    # Fill nulls
    df["skills"] = df["skills"].fillna("")
    df["education"] = df["education"].fillna("B.Tech")
    df["experience"] = df["experience"].fillna("Fresher")

    # Strip spaces
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Deduplicate by student_id or name
    duplicates_count = int(df.duplicated(subset=["student_id"]).sum())
    df = df.drop_duplicates(subset=["student_id"], keep="first")
    df = df.reset_index(drop=True)

    # Standardize skills
    df["skills"] = df["skills"].apply(standardize_skills)

    report = {
        "initial_rows": initial_rows,
        "cleaned_rows": len(df),
        "duplicates_removed": duplicates_count,
        "missing_cells_imputed": int(raw_df.isnull().sum().sum())
    }

    return df, report
