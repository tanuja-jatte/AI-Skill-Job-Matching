# AI-Based Skill-to-Job Matching System

> **A 2nd-Year B.Tech Academic Mini Project Demonstrating Data Engineering, Data Structures, and Applied Artificial Intelligence**

---

## Table of Contents
1. [Project Introduction](#1-project-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Existing System](#4-existing-system)
5. [Proposed System](#5-proposed-system)
6. [System Architecture](#6-system-architecture)
7. [Technology Stack](#7-technology-stack)
8. [Data Engineering Pipeline](#8-data-engineering-pipeline)
9. [Data Structures Used](#9-data-structures-used)
10. [AI / Machine Learning Algorithm](#10-ai--machine-learning-algorithm)
11. [Database Design](#11-database-design)
12. [Project Workflow](#12-project-workflow)
13. [Results and Performance](#13-results-and-performance)
14. [Advantages](#14-advantages)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [How to Run the Project](#17-how-to-run-the-project)
18. [Viva Voice & Examiner Defense Guide](#18-viva-voice--examiner-defense-guide)

---

## 1. Project Introduction
The **AI-Based Skill-to-Job Matching System** is an intelligent career advisory web application designed to help engineering students align their technical skills with current job market requirements. By analyzing candidate skills and comparing them against a curated dataset of realistic job postings, the system computes similarity scores and recommends the top 5 most suitable job roles along with identifying missing skills for targeted upskilling.

This project is tailored specifically as a **2nd-Year B.Tech Mini Project**, providing a balance between practical software engineering and explainable academic theory across **Data Engineering**, **Data Structures**, and **Applied Artificial Intelligence**.

---

## 2. Problem Statement
Engineering students often struggle to understand which job roles align with their skill sets. Traditional job portals rely on static keyword queries or opaque algorithms, causing:
- Students applying for mismatched roles with low selection chances.
- Lack of clarity on *why* a job matches and *which specific skills* are missing.
- Complex commercial platforms requiring paid APIs, heavy microservices, or complex setups unsuitable for academic study and offline demonstration.

---

## 3. Objectives
1. **Bridge the Skill-to-Role Gap**: Provide automated, explainable job recommendations based on skill similarity.
2. **Demonstrate Data Engineering**: Implement a complete modular data pipeline (Ingestion ➔ Cleaning ➔ Transformation ➔ Relational Storage).
3. **Demonstrate Fundamental Data Structures**: Visibly utilize Hash Maps (Dictionaries), Dynamic Arrays (Lists), Bipartite Graphs (Adjacency Lists), and Sorting (Timsort).
4. **Demonstrate Machine Learning**: Apply TF-IDF Vectorization and Cosine Similarity to calculate matching percentages (0–100%).
5. **Beginner-Friendly & Self-Contained**: Run locally on standard student hardware without internet dependencies, paid APIs, or cloud containers.

---

## 4. Existing System
- Relies on basic keyword string searching (e.g., `SQL LIKE '%python%'`).
- Fails to distinguish between rare specialized skills vs. ubiquitous skills.
- Fails to calculate graded percentage compatibility scores.
- Highly dependent on complex web microservices, external cloud APIs, or opaque machine learning models that students cannot easily explain during a viva examination.

---

## 5. Proposed System
- **Transparent Mathematical Engine**: Uses Scikit-learn's `TfidfVectorizer` and `cosine_similarity` to calculate mathematically rigorous, explainable scores.
- **Relational Persistence**: Uses embedded `SQLite` to store raw and cleaned jobs, candidate profiles, and historical recommendation logs.
- **Interactive Visual Dashboard**: Powered by `Streamlit` and `Plotly`, showing job distribution across Indian metro tech hubs and demand frequencies for core skills.
- **Bipartite Skill-Job Graph**: Employs an adjacency-list graph structure enabling bidirectional exploration between technical competencies and career profiles.

---

## 6. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                              STREAMLIT WEB FRONTEND                                |
|  [PAGE 2 — Student Profile]  |  [PAGE 3 — Job Recommendations]  |  [PAGE 5 — Skill Graph]   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                             DATA ENGINEERING PIPELINE                             |
|                                                                                   |
|  1. Ingestion (src/data_ingestion.py):                                            |
|     Reads data/jobs.csv & data/students.csv using Pandas                         |
|                                                                                   |
|  2. Cleaning & Standardization (src/data_cleaning.py):                            |
|     • Detects & imputes missing values                                            |
|     • Purges duplicate records based on (job_title, company)                      |
|     • Normalizes skills: lowercase, trim whitespace, deduplicate tokens           |
|                                                                                   |
|  3. Transformation & Features (src/data_transformation.py):                       |
|     • Extracts global skill vocabulary                                            |
|     • Computes skill frequency distribution & location metrics                    |
|                                                                                   |
|  4. Relational Storage (src/database.py):                                         |
|     • Stores processed data into SQLite database (database/job_matching.db)       |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
+---------------------------------------+ +---------------------------------------+
|        DATA STRUCTURES ENGINE         | |           AI / ML ENGINE              |
|        (src/data_structures.py)       | |        (src/recommendation.py)        |
|                                       | |                                       |
| • Hash Map: skill -> [jobs] (O(1))    | | • TF-IDF Vectorizer                   |
| • List: candidate & job skill arrays  | | • Cosine Similarity Calculation       |
| • Graph: Bipartite Adjacency List     | | • Percentage Scoring (0–100%)         |
| • Sorting: Timsort (O(N log N))       | | • Skill Gap Extraction (Upskilling)   |
+---------------------------------------+ +---------------------------------------+
```

---

## 7. Technology Stack
- **Programming Language**: Python 3.10+
- **Frontend / UI**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Database**: SQLite3 (Standard Python Library)
- **Visualizations**: Plotly Express & Matplotlib
- **Graph Modeling**: NetworkX & Pure Python Adjacency Lists

---

## 8. Data Engineering Pipeline

The project follows the standard industry data engineering lifecycle:

$$\text{Raw Data (CSV)} \longrightarrow \text{Ingestion} \longrightarrow \text{Cleaning} \longrightarrow \text{Transformation} \longrightarrow \text{SQLite Storage} \longrightarrow \text{AI Recommender}$$

### Pipeline Modules:
1. `src/data_ingestion.py`:
   - Validates existence of `data/jobs.csv` and `data/students.csv`.
   - Checks presence of required schema columns.
   - Gracefully reports file missing or read exceptions.
2. `src/data_cleaning.py`:
   - **Missing Value Handling**: Imputes missing locations with `'Remote'`, missing experience with `'0-2 years'`, and empty skills with blank strings.
   - **Deduplication**: Removes redundant job postings matching on `(job_title, company)`.
   - **Skill Standardization**: Converts mixed tokens like `" Python ", "PYTHON", "  python  "` into a canonical representation `"python"`.
3. `src/data_transformation.py`:
   - Tokenizes and extracts the complete vocabulary of unique industry skills.
   - Generates summary statistics: top skills, hiring companies, salary ranges, location breakdown.
4. `src/database.py`:
   - Persists cleaned records into SQLite (`database/job_matching.db`).

---

## 9. Data Structures Used

### A. Dictionary / Hash Map (Inverted Index)
- **Usage**: Maps each technical skill to a list of jobs requiring that skill:
  ```python
  skill_map = {
      "python": [{"job_id": "JOB001", "job_title": "Data Analyst"}, ...],
      "sql": [{"job_id": "JOB001", ...}, {"job_id": "JOB002", ...}],
      "machine learning": [{"job_id": "JOB003", ...}]
  }
  ```
- **Time Complexity**: **$O(1)$** average time lookup.
- **Viva Defense**: Avoids an expensive $O(N)$ linear scan across all jobs when searching by skill.

### B. List (Dynamic Array)
- **Usage**:
  - Stores student skills: `['python', 'sql', 'pandas', 'excel']`
  - Stores job requirement skills: `['python', 'sql', 'power bi']`
  - Stores ranked recommendation dictionaries.
- **Properties**: Sequential ordered storage with $O(1)$ amortized append and $O(1)$ indexed access.

### C. Graph (Adjacency List)
- **Usage**: Implemented via `SkillJobGraph` class modeling a **Bipartite Graph**:
  - **Node Set 1 ($V_1$)**: Technical Skills (e.g., Python, SQL, C++)
  - **Node Set 2 ($V_2$)**: Job Roles (e.g., Data Analyst, ML Engineer)
  - **Edges ($E$)**: Connection exists if a skill is required for that job role.
- **Representation**: Stored internally as an Adjacency List using a Python dictionary:
  ```text
  python ─────→ [Data Analyst, Python Developer, ML Engineer, Data Engineer]
  sql ────────→ [Data Analyst, Data Engineer, Database Administrator]
  ```
- **Complexity**: Space complexity is **$O(V + E)$**, far more memory-efficient than an $O(V^2)$ Adjacency Matrix.

### D. Sorting (Timsort)
- **Usage**: Ranks job recommendation objects in descending order of matching score:
  ```python
  sorted_recommendations = sorted(jobs, key=lambda x: x["matching_score"], reverse=True)
  ```
- **Time Complexity**: **$O(N \log N)$** worst-case using Python's stable Timsort algorithm.

---

## 10. AI / Machine Learning Algorithm

The recommendation engine utilizes **TF-IDF Vectorization** and **Cosine Similarity**.

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
Text descriptions of skills are transformed into numerical vector spaces:

$$\text{TF}(t, d) = \frac{\text{Count of skill } t \text{ in job } d}{\text{Total skills in job } d}$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + N}{1 + \text{count of jobs requiring skill } t}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Why it matters**: Rare skills (e.g., *TensorFlow*, *PyTorch*, *CUDA*) receive higher mathematical weighting than ubiquitous skills (e.g., *Git*).

### 2. Cosine Similarity
Calculates the cosine of the angle between the student skill vector ($\vec{A}$) and the job skill vector ($\vec{B}$):

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **Score Range**: Between `0.0` (0% match, orthogonal vectors) and `1.0` (100% match, parallel vectors).
- **Matching Percentage**: $\text{Score} = \text{round}(\text{Cosine Similarity} \times 100, 1)$

### 3. Why Cosine Similarity over Euclidean Distance?
- **Length Independence**: Euclidean distance is distorted by the quantity of items listed. If a student lists 10 skills and a job lists 3 skills, the Euclidean distance is large even if all 3 job skills are completely satisfied.
- Cosine similarity measures **directional orientation**, strictly evaluating skill overlap proportion rather than list magnitude.

---

## 11. Database Design

Database Engine: **SQLite** (`database/job_matching.db`)

### Schema:
```sql
-- 1. Jobs Table
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    job_title TEXT NOT NULL,
    company TEXT NOT NULL,
    required_skills TEXT NOT NULL,
    experience TEXT,
    location TEXT,
    salary_range TEXT,
    job_description TEXT
);

-- 2. Students Table
CREATE TABLE students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    skills TEXT NOT NULL,
    education TEXT,
    experience TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Recommendations Table
CREATE TABLE recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    job_id TEXT,
    matching_score REAL NOT NULL,
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);
```

---

## 12. Project Workflow

```text
1. Startup: App checks if SQLite DB exists; if not, triggers run_pipeline().
2. Ingestion: Reads 70+ jobs and sample student records from data/ CSVs.
3. Cleaning: Imputes missing fields, purges duplicates, standardizes skill tokens.
4. User Input: Student enters name, education, and selects/inputs technical skills.
5. AI Processing:
   - Skills standardized into clean token strings.
   - Fit into TF-IDF vector space with job requirements.
   - Cosine similarity matrix computed.
6. Ranking: Results sorted descending by matching percentage.
7. Skill Gap Analysis: Highlights matched skills (green) and missing skills (amber).
8. Persistence: Student query and recommendation scores saved to SQLite database.
```

---

## 13. Results and Performance
- **Dataset Size**: 70+ realistic technical job postings across leading Indian tech hubs (Bengaluru, Pune, Hyderabad, Delhi NCR, Mumbai, Chennai).
- **Ingestion Time**: < 0.25 seconds.
- **Cleaning & Standardization**: < 0.15 seconds.
- **Recommendation Latency**: < 0.05 seconds per search query.
- **Storage Footprint**: Single SQLite `.db` file (< 200 KB).

---

## 14. Advantages
1. **Explainable AI**: Every score can be traced to exact vector dot-products and skill intersections.
2. **Actionable Upskilling**: Explicitly shows which skills the student is missing to achieve 100% suitability for their dream job.
3. **100% Local & Free**: Requires zero paid API keys, internet connectivity, or cloud credentials.
4. **Academically Rigorous**: Directly demonstrates 3 core computer science subjects (Data Structures, Data Engineering, AI/ML).

---

## 15. Limitations
- Recommendations rely on semantic skill matching rather than soft skills or psychometric evaluations.
- Currently supports English technical vocabulary.
- Does not automatically scrape real-time live job postings from LinkedIn (kept self-contained by design for stability during viva demonstrations).

---

## 16. Future Scope
- Integration with PDF resume parsers using optical character recognition or regex tokenizers.
- Dynamic course and tutorial recommendations mapped to identified missing skills.
- Inclusion of salary negotiation estimators based on regional averages.

---

## 17. How to Run the Project

### Option A: One-Click Windows Batch Script (Recommended)
Double-click:
```cmd
run.bat
```
This batch file will automatically verify Python, install any missing requirements, initialize the SQLite database, and launch the Streamlit web app in your browser!

### Option B: Manual Command Line
1. **Open PowerShell / Command Prompt** in the project directory:
   ```cmd
   cd c:\Users\tanuj\OneDrive\Desktop\Mini_project
   ```

2. **Install Required Packages**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Launch Streamlit Web Application**:
   ```cmd
   streamlit run app.py
   ```
   *(Or if python is not in PATH: `python -m streamlit run app.py`)*

4. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

---

## 18. Viva Voice & Examiner Defense Guide

### Q1: What is Data Engineering?
**Answer:** Data Engineering is the discipline of building pipelines that collect, clean, transform, and store data in a reliable, structured format so that downstream applications and AI models can consume it accurately without errors.

### Q2: What is a Data Structure?
**Answer:** A data structure is a systematic way of organizing and storing data in computer memory so that operations like searching, insertion, deletion, and sorting can be executed with optimal time and space complexity.

### Q3: What is Artificial Intelligence?
**Answer:** Artificial Intelligence is the branch of computer science that develops software capable of performing tasks that traditionally require human cognition, such as pattern recognition, language processing, and personalized recommendations.

### Q4: What is TF-IDF?
**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) measures how significant a term is to a specific document within a collection. Term Frequency measures local occurrence, while Inverse Document Frequency penalizes overly common words and rewards rare, specialized words.

### Q5: What is Cosine Similarity and why is it preferred over Euclidean Distance?
**Answer:** Cosine Similarity measures the cosine of the angle between two non-zero vectors in multi-dimensional space. We chose it because it is **length-independent**: candidate profiles listing 15 skills can be evaluated fairly against job requirements listing 4 skills without the penalty that Euclidean distance would impose.

### Q6: Why did you use a Dictionary (Hash Map)?
**Answer:** A dictionary uses an underlying hash table providing **$O(1)$ average time complexity** for lookups. By indexing `skill -> [related jobs]`, we can retrieve matching roles instantaneously without linearly iterating over every job ($O(N)$).

### Q7: Why did you use a Graph?
**Answer:** The relationship between skills and job roles naturally forms a **Bipartite Graph** where skills and job titles represent two distinct sets of nodes connected by requirement edges. Using an **Adjacency List** representation optimizes memory usage to **$O(V + E)$** compared to an $O(V^2)$ matrix.

### Q8: How was the dataset cleaned?
**Answer:** In `src/data_cleaning.py`:
1. Empty fields were imputed with standard values (e.g. `'Remote'`, `'0-2 years'`).
2. Duplicates were identified by composite key `(job_title, company)` and purged.
3. Skill strings were stripped of whitespace, lowercased, and deduplicated.

### Q9: How is the final recommendation score calculated?
**Answer:**
$$\text{Score} = \text{round}\left(\frac{\vec{v}_{\text{student}} \cdot \vec{v}_{\text{job}}}{\|\vec{v}_{\text{student}}\| \times \|\vec{v}_{\text{job}}\|} \times 100, 1\right)$$
The scores are then ranked in descending order using Timsort.
