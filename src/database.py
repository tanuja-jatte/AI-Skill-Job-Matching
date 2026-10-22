"""
DATA ENGINEERING COMPONENT - STEP 4: DATABASE STORAGE
Module: src/database.py

Viva Explanation:
-----------------
What is SQLite?
SQLite is a C-language library that implements a small, fast, self-contained, high-reliability,
full-featured SQL database engine. Unlike MySQL or PostgreSQL, it is serverless, zero-configuration,
and stores the entire database in a single disk file.

Why use SQLite in this project?
1. Embedded: Runs inside the application process without external server installation.
2. ACID Compliant: Ensures Atomicity, Consistency, Isolation, and Durability.
3. Fast & Lightweight: Perfectly suited for local student projects and desktop applications.
4. Relational Integrity: Links students to their generated job recommendations via foreign keys.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = os.path.join("database", "job_matching.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Ensures parent directories exist.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Allow dictionary-like row access
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Initializes database tables: jobs, students, recommendations.

    Returns:
        bool: True if initialized successfully
    """
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()

        # 1. Jobs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_title TEXT NOT NULL,
                company TEXT NOT NULL,
                required_skills TEXT NOT NULL,
                experience TEXT,
                location TEXT,
                salary_range TEXT,
                job_description TEXT
            );
        """)

        # 2. Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                skills TEXT NOT NULL,
                education TEXT,
                experience TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 3. Recommendations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                job_id TEXT,
                matching_score REAL NOT NULL,
                recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id)
            );
        """)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False


def insert_jobs(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Inserts or updates cleaned job records into the jobs table.
    """
    if df.empty:
        return 0

    conn = get_connection(db_path)
    cursor = conn.cursor()

    inserted_count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO jobs 
            (job_id, job_title, company, required_skills, experience, location, salary_range, job_description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row.get("job_id", "")),
            str(row.get("job_title", "")),
            str(row.get("company", "")),
            str(row.get("required_skills", "")),
            str(row.get("experience", "0-2 years")),
            str(row.get("location", "Remote")),
            str(row.get("salary_range", "Not Disclosed")),
            str(row.get("job_description", ""))
        ))
        inserted_count += 1

    conn.commit()
    conn.close()
    return inserted_count


def insert_students(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Inserts student records into the students table.
    """
    if df.empty:
        return 0

    conn = get_connection(db_path)
    cursor = conn.cursor()

    inserted_count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO students 
            (student_id, name, skills, education, experience)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(row.get("student_id", "")),
            str(row.get("name", "")),
            str(row.get("skills", "")),
            str(row.get("education", "")),
            str(row.get("experience", ""))
        ))
        inserted_count += 1

    conn.commit()
    conn.close()
    return inserted_count


def save_single_student(student_dict: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Saves or updates a single student profile.
    """
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO students (student_id, name, skills, education, experience)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_dict["student_id"],
            student_dict["name"],
            student_dict["skills"],
            student_dict.get("education", "B.Tech"),
            student_dict.get("experience", "Fresher")
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving student: {e}")
        return False


def save_recommendations(student_id: str, recommendations: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Stores recommended jobs and their matching scores for a student.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    count = 0
    for rec in recommendations:
        cursor.execute("""
            INSERT INTO recommendations (student_id, job_id, matching_score)
            VALUES (?, ?, ?)
        """, (
            student_id,
            rec["job_id"],
            float(rec["matching_score"])
        ))
        count += 1

    conn.commit()
    conn.close()
    return count


def get_all_jobs(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves all jobs from the SQLite database as a Pandas DataFrame.
    """
    if not os.path.exists(db_path):
        init_db(db_path)

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM jobs", conn)
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_all_students(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves all students from the SQLite database as a Pandas DataFrame.
    """
    if not os.path.exists(db_path):
        init_db(db_path)

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM students", conn)
    except Exception as e:
        print(f"Error fetching students: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_recent_recommendations(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves recent recommendations joined with job and student details.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = get_connection(db_path)
    query = """
        SELECT 
            r.recommendation_id,
            r.student_id,
            s.name AS student_name,
            r.job_id,
            j.job_title,
            j.company,
            r.matching_score,
            r.recommended_at
        FROM recommendations r
        LEFT JOIN students s ON r.student_id = s.student_id
        LEFT JOIN jobs j ON r.job_id = j.job_id
        ORDER BY r.recommendation_id DESC
        LIMIT ?
    """
    try:
        df = pd.read_sql_query(query, conn, params=(limit,))
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Calculates quick statistical counts from database tables.
    """
    if not os.path.exists(db_path):
        init_db(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    stats = {
        "total_jobs": 0,
        "total_students": 0,
        "total_recommendations": 0,
        "avg_matching_score": 0.0
    }

    try:
        cursor.execute("SELECT COUNT(*) FROM jobs")
        stats["total_jobs"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM students")
        stats["total_students"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM recommendations")
        stats["total_recommendations"] = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(matching_score) FROM recommendations")
        avg_score = cursor.fetchone()[0]
        stats["avg_matching_score"] = round(float(avg_score), 2) if avg_score is not None else 0.0

    except Exception as e:
        print(f"Error computing database statistics: {e}")
    finally:
        conn.close()

    return stats
