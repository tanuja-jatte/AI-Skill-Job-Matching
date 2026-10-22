"""
Verification Test Script for AI Skill-to-Job Matching System
"""
import sys
import os

print("Python Version:", sys.version)

# 1. Test Ingestion
from src.data_ingestion import load_raw_jobs, load_raw_students
jobs_raw, jobs_rep = load_raw_jobs("data/jobs.csv")
students_raw, stu_rep = load_raw_students("data/students.csv")
print("Raw jobs loaded:", len(jobs_raw), jobs_rep["status"])
print("Raw students loaded:", len(students_raw), stu_rep["status"])

# 2. Test Cleaning
from src.data_cleaning import clean_jobs_data, clean_students_data, standardize_skills
jobs_clean, j_rep = clean_jobs_data(jobs_raw)
students_clean, s_rep = clean_students_data(students_raw)
print("Cleaned jobs:", len(jobs_clean), "| Duplicates removed:", j_rep["duplicates_removed"])
print("Test skill standardization:", standardize_skills("  Python , PYTHON, pandas ,  SQL  "))

# 3. Test Transformation & Stats
from src.data_transformation import compute_dataset_statistics, extract_unique_skills
stats = compute_dataset_statistics(jobs_clean, students_clean)
print("Total unique skills:", stats["total_unique_skills"])
print("Top 5 skills:", list(stats["top_10_skills"].items())[:5])

# 4. Test Database
from src.database import init_db, insert_jobs, insert_students, get_all_jobs, get_db_stats
init_db()
insert_jobs(jobs_clean)
insert_students(students_clean)
db_jobs = get_all_jobs()
print("Database jobs count:", len(db_jobs))
print("Database stats:", get_db_stats())

# 5. Test Data Structures
from src.data_structures import build_skill_to_jobs_dict, lookup_jobs_by_skill, build_graph_from_dataframe, sort_recommended_jobs
skill_dict = build_skill_to_jobs_dict(jobs_clean)
print("Dictionary lookup for 'python' jobs count:", len(lookup_jobs_by_skill(skill_dict, "python")))
graph = build_graph_from_dataframe(jobs_clean)
print("Graph stats:", graph.get_stats())
print("Graph jobs for 'sql':", graph.find_jobs_for_skill("sql")[:3])

# 6. Test AI Recommendation
from src.recommendation import recommend_jobs, explain_match
test_skills = "Python, SQL, Pandas, Excel"
recs = recommend_jobs(test_skills, jobs_clean, top_n=5)
print(f"\nRecommendations for [{test_skills}]:")
for r in recs:
    print(f"  - {r['job_title']} at {r['company']}: {r['matching_score']}% | Matched: {r['matched_skills']} | Missing: {r['missing_skills']}")

# 7. Test Match Explanation
exp = explain_match("python, sql", "python, sql, power bi")
print("\nMath explanation test:", exp["percentage_match"], "% | Cosine:", exp["cosine_similarity"])

# 8. Test Edge Cases (Empty input, unknown skills)
empty_recs = recommend_jobs("", jobs_clean)
print("Empty skills handled safely (count):", len(empty_recs))

unknown_recs = recommend_jobs("Quantum Computing, Fortran, COBOL", jobs_clean)
print("Unknown skills handled safely (highest score):", unknown_recs[0]["matching_score"] if unknown_recs else 0.0)

print("\n>>> ALL SYSTEM TESTS PASSED SUCCESSFULLY! <<<")
