"""
================================================================================
AI-Based Skill-to-Job Matching System
Streamlit Main Application
================================================================================
1. Student Profile (Candidate Information & Technical Skills)
2. Job Recommendations (TF-IDF & Cosine Similarity Rankings)
3. Skill Graph (Bipartite Network Visualization & Adjacency List)
================================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modular components
from src.data_cleaning import standardize_skills
from src.database import (
    get_all_jobs, save_single_student,
    save_recommendations, DEFAULT_DB_PATH
)
from src.data_structures import (
    SkillJobGraph, build_graph_from_dataframe
)
from src.recommendation import recommend_jobs, explain_match
from utils.helper import run_pipeline


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Skill-to-Job Matching System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .job-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .skill-tag-matched {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .skill-tag-missing {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# APPLICATION INITIALIZATION (Ensure Database & Tables Exist)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_database_ready():
    """Initializes the database and runs the pipeline once if needed."""
    if not os.path.exists(DEFAULT_DB_PATH):
        run_pipeline()
    else:
        jobs = get_all_jobs()
        if jobs.empty:
            run_pipeline()


ensure_database_ready()


# -----------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "student_name" not in st.session_state:
    st.session_state["student_name"] = "Tanuj Verma"
if "student_education" not in st.session_state:
    st.session_state["student_education"] = "B.Tech Computer Science"
if "student_experience" not in st.session_state:
    st.session_state["student_experience"] = "Fresher (0-1 yrs)"
if "selected_skills" not in st.session_state:
    st.session_state["selected_skills"] = ["Python", "SQL", "Pandas", "Excel"]
if "custom_skills" not in st.session_state:
    st.session_state["custom_skills"] = ""
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = []
if "last_queried_skills" not in st.session_state:
    st.session_state["last_queried_skills"] = "python, sql, pandas, excel"


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (CLEANED TITLES WITHOUT PAGE NUMBERS OR ACADEMIC CONCEPTS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("💼 Job Matcher AI")
    st.caption("AI-Based Skill-to-Job Matching System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Student Profile",
            "Job Recommendations",
            "Skill Graph"
        ],
        index=0
    )


# Load current jobs dataset from SQLite database
jobs_df = get_all_jobs()
if jobs_df.empty:
    run_pipeline()
    jobs_df = get_all_jobs()


# =============================================================================
# STUDENT PROFILE
# =============================================================================
if page == "Student Profile":
    st.markdown('<div class="main-header">Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter your academic background and select technical skills to find matching jobs.</div>', unsafe_allow_html=True)

    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("👤 Full Name", value=st.session_state["student_name"], placeholder="e.g. Tanuj Verma")
            education = st.selectbox(
                "🎓 Highest Education",
                [
                    "B.Tech Computer Science",
                    "B.Tech Information Technology",
                    "B.Tech Artificial Intelligence & Data Science",
                    "B.Tech Electronics & Communication",
                    "B.Tech Mechanical / Other",
                    "BCA / MCA",
                    "B.Sc Computer Science"
                ],
                index=0
            )

        with col2:
            experience = st.selectbox(
                "⏳ Experience Level",
                [
                    "Fresher (0-1 yrs)",
                    "6 Months Internship",
                    "1-2 Years Project Experience",
                    "2+ Years Experience"
                ],
                index=0
            )
            st.text_input("🆔 System Status", value="Connected to Database (job_matching.db)", disabled=True)

        st.markdown("#### 🛠️ Technical Skills")
        
        # Standard default skills listed in requirements
        available_popular_skills = [
            "Python", "Java", "C++", "SQL", "HTML", "CSS", "JavaScript",
            "Pandas", "NumPy", "Machine Learning", "Data Structures",
            "Excel", "Power BI", "Statistics"
        ]

        # Multi-select list
        selected_skills = st.multiselect(
            "Select skills from the standard technical list:",
            options=available_popular_skills,
            default=[s for s in st.session_state["selected_skills"] if s in available_popular_skills]
        )

        # Custom skill input
        custom_skills = st.text_input(
            "➕ Custom Skills (comma-separated):",
            value=st.session_state["custom_skills"],
            placeholder="e.g., React, PyTorch, Docker, Git"
        )

        submitted = st.form_submit_button("🔍 Find Matching Jobs", use_container_width=True)

    if submitted:
        # Combine selected skills with custom skills
        all_skills_combined = list(selected_skills)
        if custom_skills.strip():
            custom_tokens = [s.strip() for s in custom_skills.split(",") if s.strip()]
            all_skills_combined.extend(custom_tokens)

        # Standardize skills
        combined_skills_str = ", ".join(all_skills_combined)
        standardized_skills_str = standardize_skills(combined_skills_str)

        # Error Handling: Check for empty input
        if not standardized_skills_str:
            st.error("⚠️ Please select or type at least one technical skill before searching.")
        elif not name.strip():
            st.error("⚠️ Please enter your student name.")
        else:
            # Save to session state
            st.session_state["student_name"] = name
            st.session_state["student_education"] = education
            st.session_state["student_experience"] = experience
            st.session_state["selected_skills"] = selected_skills
            st.session_state["custom_skills"] = custom_skills
            st.session_state["last_queried_skills"] = standardized_skills_str

            # Save student to database
            student_record = {
                "student_id": "STU_CURRENT",
                "name": name,
                "skills": standardized_skills_str,
                "education": education,
                "experience": experience
            }
            save_single_student(student_record)

            # Trigger AI Recommendation Engine (TF-IDF + Cosine Similarity)
            with st.spinner("🤖 Running TF-IDF Vectorization & Cosine Similarity..."):
                recommendations = recommend_jobs(standardized_skills_str, jobs_df, top_n=5)
                st.session_state["recommendations"] = recommendations

                # Store recommendations in database
                if recommendations:
                    save_recommendations("STU_CURRENT", recommendations)

            st.success(f"✅ Matching complete! Found top {len(recommendations)} recommended jobs for **{name}**.")
            st.info("👉 Click on **Job Recommendations** in the sidebar to view matching scores and missing skills!")


# =============================================================================
# JOB RECOMMENDATIONS
# =============================================================================
elif page == "Job Recommendations":
    st.markdown('<div class="main-header">Job Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Top 5 recommended jobs ranked by matching score.</div>', unsafe_allow_html=True)

    recommendations = st.session_state.get("recommendations", [])
    student_name = st.session_state.get("student_name", "Student")
    query_skills = st.session_state.get("last_queried_skills", "")

    # Default fallback if no search was performed yet
    if not recommendations:
        st.info("💡 Showing recommendations for current skills: `Python, SQL, Pandas, Excel`.")
        query_skills = "python, sql, pandas, excel"
        recommendations = recommend_jobs(query_skills, jobs_df, top_n=5)
        st.session_state["recommendations"] = recommendations

    st.markdown(f"**Candidate:** `{student_name}` &nbsp;|&nbsp; **Input Skills:** `{query_skills}`")
    st.markdown("---")

    # Display Top 5 Recommended Jobs
    for rank, job in enumerate(recommendations, start=1):
        score = job["matching_score"]

        # Color-coded badge
        if score >= 80:
            badge_color = "#10B981"
            match_text = "Strong Match"
        elif score >= 50:
            badge_color = "#F59E0B"
            match_text = "Moderate Match"
        else:
            badge_color = "#6B7280"
            match_text = "Potential Match"

        st.markdown(f"""
        <div class="job-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h3 style="margin: 0; color: #1E3A8A;">#{rank}. {job['job_title']}</h3>
                <span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.95rem;">
                    Match: {score}% ({match_text})
                </span>
            </div>
            <div style="font-size: 1.0rem; font-weight: 600; color: #374151; margin-bottom: 8px;">
                🏢 {job['company']} &nbsp;|&nbsp; 📍 {job['location']} &nbsp;|&nbsp; 💰 {job['salary_range']} &nbsp;|&nbsp; ⏱️ {job['experience']}
            </div>
            <p style="color: #4B5563; font-size: 0.92rem; margin-bottom: 10px;">{job['job_description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar for matching score
        st.progress(float(score) / 100.0)

        # Matched vs Missing skills breakdown
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown("**✅ Matched Skills:**")
            if job["matched_skills"]:
                matched_html = "".join([f'<span class="skill-tag-matched">✓ {s}</span>' for s in job["matched_skills"]])
                st.markdown(matched_html, unsafe_allow_html=True)
            else:
                st.caption("No direct token overlap.")

        with col_m2:
            st.markdown("**💡 Required Skills to Learn (Skill Gap):**")
            if job["missing_skills"]:
                missing_html = "".join([f'<span class="skill-tag-missing">+ {s}</span>' for s in job["missing_skills"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.caption("None! You meet all required skills.")

        # Educational expander explaining the calculation for viva
        with st.expander(f"📐 Mathematical Calculation ({score}% Match)"):
            math_exp = explain_match(query_skills, job["required_skills"])
            st.markdown(f"""
            - **Student Skills:** `{', '.join(math_exp['student_skills'])}`
            - **Job Required Skills:** `{', '.join(math_exp['job_skills'])}`
            - **Overlapping Skills:** `{', '.join(math_exp['overlapping_skills']) if math_exp['overlapping_skills'] else 'None'}`
            - **Cosine Similarity:**
              $$\\text{{Cosine Similarity}}(\\vec{{A}}, \\vec{{B}}) = \\frac{{\\vec{{A}} \\cdot \\vec{{B}}}}{{\\|\\vec{{A}}\\| \\times \\|\\vec{{B}}\\|}} = {math_exp['cosine_similarity']}$$
            - **Final Score:** `{math_exp['cosine_similarity']} × 100 = {math_exp['percentage_match']}%`
            """)

        st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# SKILL GRAPH
# =============================================================================
elif page == "Skill Graph":
    st.markdown('<div class="main-header">Skill → Job Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualization of the relationship between technical skills and job roles.</div>', unsafe_allow_html=True)

    # Build Graph from Database
    graph = build_graph_from_dataframe(jobs_df)
    stats = graph.get_stats()

    # Graph summary metrics
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Skill Nodes", stats["skill_nodes"])
    with g2:
        st.metric("Job Role Nodes", stats["job_nodes"])
    with g3:
        st.metric("Total Connections", stats["total_edges"])

    st.markdown("---")

    # Interactive Skill Exploration
    unique_skills = sorted(list(graph.skills))
    selected_skill = st.selectbox(
        "🔎 Select a Skill to see connected Job Roles:",
        options=unique_skills,
        index=unique_skills.index("python") if "python" in unique_skills else 0
    )

    connected_jobs = graph.find_jobs_for_skill(selected_skill)

    st.markdown(f"### 🔗 Skill: `{selected_skill}` ─────→ Job Roles")
    st.write(f"The skill **'{selected_skill}'** connects to **{len(connected_jobs)}** job roles:")

    # Simple visual ASCII / schematic representation as specified
    ascii_tree = f"{selected_skill}\n" + "\n".join([f"  ├──→ {job}" for job in connected_jobs])
    st.code(ascii_tree, language="text")

    st.markdown("---")
    st.markdown("#### 🕸️ Visual Network Diagram")

    # Clean, spacious Bipartite Graph visualization where badges automatically fit the text
    display_jobs = connected_jobs[:8]  # Limit to top 8 connected roles for a clean, spacious layout
    N = len(display_jobs)

    if N > 0:
        fig, ax = plt.subplots(figsize=(10, max(4.5, N * 0.75)))

        ax.set_xlim(-0.05, 1.25)
        ax.set_ylim(-0.1, 1.1)

        y_skill = 0.5
        x_skill = 0.15

        # Draw Skill Node (Prominent Blue Badge)
        ax.text(
            x_skill, y_skill, f"  SKILL: {selected_skill.upper()}  ",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#EFF6FF", edgecolor="#2563EB", lw=2),
            fontsize=11, fontweight="bold", color="#1E40AF", zorder=4
        )

        # Draw connected Job Roles (Spacious Mint Badges with Directed Arrows)
        for i, job in enumerate(display_jobs):
            y_job = 1.0 - (i + 0.5) / N
            x_job = 0.72

            # Smooth connection arrow pointing from skill badge to job badge
            ax.annotate(
                "",
                xy=(x_job - 0.03, y_job),
                xytext=(x_skill + 0.12, y_skill),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#64748B",
                    lw=1.8,
                    connectionstyle="arc3,rad=0.06"
                ),
                zorder=2
            )

            # Job Role Box that wraps the text with uniform padding
            ax.text(
                x_job, y_job, f"  {job}  ",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.55", facecolor="#F0FDF4", edgecolor="#16A34A", lw=1.5),
                fontsize=10, fontweight="bold", color="#166534", zorder=4
            )

        ax.axis("off")
        ax.set_title(
            f"Bipartite Graph: '{selected_skill.title()}' (Blue) ──→ Associated Career Roles (Green)",
            fontsize=12, fontweight="bold", pad=14, color="#1E3A8A"
        )

        st.pyplot(fig)
    else:
        st.info("No connected jobs found for this skill.")

    st.markdown("---")
    # Adjacency List Structure Display
    with st.expander("📚 Inspect Adjacency List (Python Dictionary Data Structure)"):
        st.write("Adjacency List representation stored as `Dict[str, List[str]]`:")
        sample_adj = {k: graph.adjacency_list[k] for k in list(graph.skills)[:6]}
        st.json(sample_adj)
