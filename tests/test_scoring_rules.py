import pytest
from matching_engine import MatchingEngine
from unittest.mock import MagicMock

def test_score_candidate_python_logic():
    """
    Test the Python implementation of scoring rules.
    This serves as a ground truth for what the Cypher query should produce.
    """
    engine = MatchingEngine()
    engine.graph = MagicMock() # Mock connection used in init
    
    candidate = {
        "skills": [
            {"name": "Python", "proficiency": "Expert"},
            {"name": "AWS", "proficiency": "Intermediate"}
        ],
        "years_experience": 5,
        "availability": 100
    }
    
    rfp_skills = ["Python", "AWS", "Docker"]
    
    # Expected Score Calculation:
    # Python: Base(10) + Expert(8) = 18
    # AWS: Base(10) + Intermediate(3) = 13
    # Docker: 0
    # Experience: 5 * 2 = 10
    # Availability: 100 * 0.5 = 50
    # Total = 18 + 13 + 10 + 50 = 91
    
    score = engine.score_candidate(candidate, rfp_skills)
    
    assert score == 91
