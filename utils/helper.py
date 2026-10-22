"""
HELPER UTILITIES
Module: utils/helper.py

Provides common utility functions for data formatting, UI rendering,
and orchestrating the end-to-end Data Engineering pipeline.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd

from src.data_ingestion import load_raw_jobs, load_raw_students
from src.data_cleaning import clean_jobs_data, clean_students_data
from src.data_transformation import compute_dataset_statistics
from src.database import init_db, insert_jobs, insert_students, get_all_jobs, DEFAULT_DB_PATH


def run_pipeline(
    jobs_csv: str = "data/jobs.csv",
    students_csv: str = "data/students.csv",
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Executes the complete Data Engineering Pipeline:
    1. Ingestion: Reads raw CSVs
    2. Cleaning: Handles nulls, strips whitespace, removes duplicates, standardizes skills
    3. Transformation: Computes statistics and vocabulary
    4. Database Storage: Stores cleaned data into SQLite database
    
    Returns:
        dict: Detailed execution report for each pipeline stage
    """
    pipeline_report = {
        "status": "FAILED",
        "stages": {}
    }

    # Step 1: Ingestion
    raw_jobs_df, jobs_ingest_rep = load_raw_jobs(jobs_csv)
    raw_students_df, students_ingest_rep = load_raw_students(students_csv)

    pipeline_report["stages"]["1_ingestion"] = {
        "jobs": jobs_ingest_rep,
        "students": students_ingest_rep
    }

    if raw_jobs_df.empty:
        pipeline_report["message"] = "Jobs dataset ingestion failed."
        return pipeline_report

    # Step 2: Cleaning
    clean_jobs_df, jobs_clean_rep = clean_jobs_data(raw_jobs_df)
    clean_students_df, students_clean_rep = clean_students_data(raw_students_df)

    pipeline_report["stages"]["2_cleaning"] = {
        "jobs": jobs_clean_rep,
        "students": students_clean_rep
    }

    # Step 3: Transformation
    dataset_stats = compute_dataset_statistics(clean_jobs_df, clean_students_df)
    pipeline_report["stages"]["3_transformation"] = dataset_stats

    # Step 4: Database Storage
    init_db(db_path)
    jobs_saved = insert_jobs(clean_jobs_df, db_path)
    students_saved = insert_students(clean_students_df, db_path)

    pipeline_report["stages"]["4_database"] = {
        "db_path": db_path,
        "jobs_stored": jobs_saved,
        "students_stored": students_saved
    }

    pipeline_report["status"] = "SUCCESS"
    pipeline_report["message"] = f"Pipeline executed successfully. Stored {jobs_saved} jobs & {students_saved} students in SQLite."
    return pipeline_report


def format_score_badge(score: float) -> str:
    """
    Returns an HTML badge with dynamic color based on score magnitude.
    """
    if score >= 80:
        color = "#10B981"  # Emerald Green
        level = "High Match"
    elif score >= 50:
        color = "#F59E0B"  # Amber
        level = "Moderate Match"
    else:
        color = "#EF4444"  # Red / Coral
        level = "Low Match"

    return f'<span style="background-color: {color}22; color: {color}; border: 1px solid {color}; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.9rem;">{score:.1f}% ({level})</span>'
