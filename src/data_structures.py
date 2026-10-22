"""
DATA STRUCTURES COMPONENT
Module: src/data_structures.py

Viva Explanation:
-----------------
What are Data Structures?
A data structure is a specialized format for organizing, processing, retrieving,
and storing data efficiently in memory.

Why are Data Structures used here?
In this project, 4 core data structures are visibly used:
1. DICTIONARY (Hash Map):
   - Maps each skill -> list of jobs requiring it (Inverted Index).
   - Time Complexity: O(1) average lookup time.
   - Far superior to scanning all jobs sequentially (O(N)).
2. LIST:
   - Stores sequences of student skills, individual job skills, and ranked recommendations.
   - Dynamic array supporting indexed access and sequential iteration.
3. GRAPH (Adjacency List):
   - Models relationships between skills and job roles as a Bipartite Graph.
   - Nodes = Skills & Job Roles. Edges = "Skill is required by Job".
   - Adjacency list representation: Memory efficient O(V + E).
4. SORTING:
   - Sorts recommended jobs descending by similarity score so the best matches appear first.
   - Uses Timsort (Python's sorted()): O(N log N) worst-case time complexity.
"""

from typing import Dict, List, Set, Any, Optional
import pandas as pd


# ============================================================================
# 1. DICTIONARY / HASH MAP COMPONENT
# ============================================================================

def build_skill_to_jobs_dict(jobs_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    DATA STRUCTURE: Hash Map / Dictionary

    Constructs an Inverted Index mapping:
    skill -> list of job records requiring that skill

    Example:
        {
            "python": [{"job_id": "JOB001", "job_title": "Data Analyst", ...}, ...],
            "sql": [{"job_id": "JOB001", ...}, {"job_id": "JOB002", ...}],
            "machine learning": [{"job_id": "JOB003", ...}]
        }

    Viva Point:
    A Python dictionary uses an underlying Hash Table. Lookups by skill key
    take O(1) average time complexity, enabling instantaneous retrieval of
    all jobs associated with a given skill without looping through the entire dataset.
    """
    skill_map: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in jobs_df.iterrows():
        job_info = {
            "job_id": row.get("job_id"),
            "job_title": row.get("job_title"),
            "company": row.get("company"),
            "location": row.get("location"),
            "salary_range": row.get("salary_range")
        }

        raw_skills = str(row.get("required_skills", ""))
        skills_list = [s.strip().lower() for s in raw_skills.split(",") if s.strip()]

        for skill in skills_list:
            if skill not in skill_map:
                skill_map[skill] = []  # List of jobs for this skill
            skill_map[skill].append(job_info)

    return skill_map


def lookup_jobs_by_skill(skill_map: Dict[str, List[Dict[str, Any]]], skill: str) -> List[Dict[str, Any]]:
    """
    O(1) average time complexity lookup of jobs for a specific skill.
    """
    cleaned_skill = skill.strip().lower()
    return skill_map.get(cleaned_skill, [])


# ============================================================================
# 2. LIST COMPONENT
# ============================================================================

def create_skill_list(comma_separated_text: str) -> List[str]:
    """
    DATA STRUCTURE: List (Dynamic Array)

    Converts a comma-separated string into a clean Python List of skill strings.
    Lists are used to store:
    - student skills: ['python', 'sql', 'pandas']
    - job skills: ['python', 'sql', 'excel', 'power bi']
    - recommended jobs: [job_1, job_2, job_3, ...]
    """
    if not comma_separated_text:
        return []
    return [s.strip().lower() for s in comma_separated_text.split(",") if s.strip()]


# ============================================================================
# 3. GRAPH COMPONENT (Skill -> Job Adjacency List)
# ============================================================================

class SkillJobGraph:
    """
    DATA STRUCTURE: Graph (Adjacency List Representation)

    A Bipartite Graph representing the relationship between Skills and Job Roles:
    - Nodes: Skill Nodes (e.g., 'python', 'sql') & Job Nodes (e.g., 'Data Analyst')
    - Edges: Directed/Undirected connection indicating a skill is needed for a job.

    Adjacency List Representation:
    Internally maintained as a dictionary of sets/lists:
    adj_list[node] = [neighbor_1, neighbor_2, ...]

    Viva Point:
    Adjacency List is preferred over Adjacency Matrix because:
    1. Space Complexity: O(V + E) instead of O(V^2), saving memory when the graph is sparse.
    2. Fast Edge Traversals: Finding all jobs for a skill takes O(degree(skill)) time.
    """

    def __init__(self):
        # Master adjacency list mapping node -> list of connected neighbor nodes
        self.adjacency_list: Dict[str, List[str]] = {}
        # Separate sets to distinguish node categories
        self.skills: Set[str] = set()
        self.jobs: Set[str] = set()

    def add_skill(self, skill: str) -> None:
        """Adds a skill node to the graph if not already present."""
        s = skill.strip().lower()
        if s and s not in self.adjacency_list:
            self.adjacency_list[s] = []
            self.skills.add(s)

    def add_job(self, job_title: str) -> None:
        """Adds a job role node to the graph if not already present."""
        j = job_title.strip()
        if j and j not in self.adjacency_list:
            self.adjacency_list[j] = []
            self.jobs.add(j)

    def add_connection(self, skill: str, job_title: str) -> None:
        """
        Creates an edge between a skill and a job title.
        Bi-directional connection for convenient query from either side:
        Skill -> Job and Job -> Skill.
        """
        s = skill.strip().lower()
        j = job_title.strip()

        if not s or not j:
            return

        self.add_skill(s)
        self.add_job(j)

        if j not in self.adjacency_list[s]:
            self.adjacency_list[s].append(j)
        if s not in self.adjacency_list[j]:
            self.adjacency_list[j].append(s)

    def find_jobs_for_skill(self, skill: str) -> List[str]:
        """
        Queries the graph to find all job roles requiring a specific skill.
        Time Complexity: O(degree(skill)).
        """
        s = skill.strip().lower()
        return self.adjacency_list.get(s, [])

    def find_skills_for_job(self, job_title: str) -> List[str]:
        """
        Queries the graph to find all skills required for a specific job.
        Time Complexity: O(degree(job)).
        """
        j = job_title.strip()
        return self.adjacency_list.get(j, [])

    def display_graph(self) -> str:
        """
        Generates a human-readable text representation of the graph.
        Useful for console demonstration and viva presentation.
        """
        lines = []
        lines.append("=== SKILL -> JOB GRAPH (ADJACENCY LIST) ===")
        for skill in sorted(self.skills):
            connected_jobs = self.adjacency_list.get(skill, [])
            if connected_jobs:
                connections = ", ".join(connected_jobs)
                lines.append(f"{skill} ────→ [{connections}]")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, int]:
        """Returns node and edge count statistics."""
        num_skills = len(self.skills)
        num_jobs = len(self.jobs)
        # Each undirected connection is counted twice in adjacency list
        total_edges = sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2
        return {
            "skill_nodes": num_skills,
            "job_nodes": num_jobs,
            "total_nodes": num_skills + num_jobs,
            "total_edges": total_edges
        }


def build_graph_from_dataframe(jobs_df: pd.DataFrame) -> SkillJobGraph:
    """
    Constructs and populates a SkillJobGraph from a jobs DataFrame.
    """
    graph = SkillJobGraph()

    for _, row in jobs_df.iterrows():
        job_title = str(row.get("job_title", "")).strip()
        raw_skills = str(row.get("required_skills", ""))
        skills = [s.strip().lower() for s in raw_skills.split(",") if s.strip()]

        graph.add_job(job_title)
        for skill in skills:
            graph.add_skill(skill)
            graph.add_connection(skill, job_title)

    return graph


# ============================================================================
# 4. SORTING COMPONENT
# ============================================================================

def sort_recommended_jobs(jobs_list: List[Dict[str, Any]], key: str = "matching_score", reverse: bool = True) -> List[Dict[str, Any]]:
    """
    DATA STRUCTURE: Sorting (Timsort)

    Sorts a list of job recommendation dictionaries by matching score.
    Highest matching score appears first (descending order).

    Parameters:
        jobs_list (list): List of job recommendation dicts
        key (str): Key to sort on (default: 'matching_score')
        reverse (bool): True for descending (highest score first)

    Returns:
        list: Sorted list of jobs

    Viva Point:
    Python uses Timsort, an adaptive hybrid of Merge Sort and Insertion Sort.
    Worst-case time complexity: O(N log N)
    Space complexity: O(N)
    It is stable, preserving relative order of jobs with identical matching scores.
    """
    if not jobs_list:
        return []

    return sorted(jobs_list, key=lambda x: x.get(key, 0.0), reverse=reverse)
