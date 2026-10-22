"""
DATA ENGINEERING COMPONENT - STEP 1: DATA INGESTION
Module: src/data_ingestion.py

Viva Explanation:
-----------------
What is Data Ingestion?
Data ingestion is the process of transporting data from external sources (such as CSV files,
APIs, sensors, or databases) into a storage repository or data processing pipeline.

Why isolate Data Ingestion?
In professional data engineering, isolating ingestion ensures:
1. Decoupled Architecture: If the source format changes (e.g., from CSV to JSON or SQL),
   only this module changes.
2. Error Boundaries: File missing, network timeout, or encoding issues are caught
   immediately before touching downstream cleaning or AI logic.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_jobs(csv_path: str = "data/jobs.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests raw job records from a CSV file.

    Parameters:
        csv_path (str): Relative or absolute path to jobs.csv

    Returns:
        tuple: (raw_dataframe, ingestion_report_dict)
    """
    report = {
        "source": csv_path,
        "status": "FAILED",
        "rows_loaded": 0,
        "columns": [],
        "missing_cells": 0,
        "message": ""
    }

    if not os.path.exists(csv_path):
        report["message"] = f"File not found at path: {csv_path}"
        return pd.DataFrame(), report

    try:
        # Read raw CSV using Pandas
        df = pd.read_csv(csv_path)

        # Basic schema check
        expected_cols = [
            "job_id", "job_title", "company", "required_skills",
            "experience", "location", "salary_range", "job_description"
        ]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            report["message"] = f"Missing expected columns in CSV: {missing_cols}"
            return df, report

        report["status"] = "SUCCESS"
        report["rows_loaded"] = len(df)
        report["columns"] = list(df.columns)
        report["missing_cells"] = int(df.isnull().sum().sum())
        report["message"] = f"Successfully ingested {len(df)} job records."
        return df, report

    except Exception as e:
        report["message"] = f"Ingestion error: {str(e)}"
        return pd.DataFrame(), report


def load_raw_students(csv_path: str = "data/students.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests raw student profiles from a CSV file.

    Parameters:
        csv_path (str): Relative or absolute path to students.csv

    Returns:
        tuple: (raw_dataframe, ingestion_report_dict)
    """
    report = {
        "source": csv_path,
        "status": "FAILED",
        "rows_loaded": 0,
        "columns": [],
        "missing_cells": 0,
        "message": ""
    }

    if not os.path.exists(csv_path):
        report["message"] = f"File not found at path: {csv_path}"
        return pd.DataFrame(), report

    try:
        df = pd.read_csv(csv_path)
        expected_cols = ["student_id", "name", "skills", "education", "experience"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            report["message"] = f"Missing expected columns in CSV: {missing_cols}"
            return df, report

        report["status"] = "SUCCESS"
        report["rows_loaded"] = len(df)
        report["columns"] = list(df.columns)
        report["missing_cells"] = int(df.isnull().sum().sum())
        report["message"] = f"Successfully ingested {len(df)} student profiles."
        return df, report

    except Exception as e:
        report["message"] = f"Ingestion error: {str(e)}"
        return pd.DataFrame(), report
