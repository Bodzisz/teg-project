import pytest
from unittest.mock import MagicMock
from parsers.assignment_loader import AssignmentLoader
from models.project_models import ProjectData, Requirement
from models.programmer_models import ProgrammerData, Skill
from datetime import date

# Mock data fixtures
@pytest.fixture
def sample_projects():
    return [
        ProjectData(
            id="PRJ-1", name="Alpha", client="C1", description="D1",
            start_date=date(2025, 1, 1), estimated_duration_months=6,
            status="active", team_size=2, requirements=[], budget="10000"
        ),
        ProjectData(
            id="PRJ-2", name="Beta", client="C1", description="D2",
            start_date=date(2025, 1, 1), estimated_duration_months=6,
            status="active", team_size=2, requirements=[], budget="10000"
        )
    ]

@pytest.fixture
def sample_programmer():
    return ProgrammerData(
        id="Dev1", name="Dev1", email="d@d.com", location="Loc",
        years_experience=5, availability=100.0, skills=[]
    )

def test_allocation_proportions(sample_projects, sample_programmer):
    """Test if availability is distributed proportionally to score."""
    loader = AssignmentLoader("utils/config.toml")
    loader.graph = MagicMock() # Mock Neo4j
    
    # Matches: Dev1 matches Alpha (score 80) and Beta (score 20)
    matches = [
        {"person_id": "Dev1", "rfp_id": "Alpha", "rfp_title": "Alpha", "score": 80},
        {"person_id": "Dev1", "rfp_id": "Beta", "rfp_title": "Beta", "score": 20}
    ]
    
    # Run assignment
    assignments = loader.assign_based_on_matches(matches, sample_projects, [sample_programmer])
    
    # Verify assignments
    assert len(assignments) == 2
    
    alpha_assign = next(a for a in assignments if a["project_id"] == "Alpha")
    beta_assign = next(a for a in assignments if a["project_id"] == "Beta")
    
    # Alpha should get ~80% of 100 = 80
    assert alpha_assign["allocation"] == 80
    # Beta should get ~20% of 100 = 20
    assert beta_assign["allocation"] == 20

def test_zero_availability_skips(sample_projects, sample_programmer):
    """Test that programmers with 0 availability are skipped."""
    loader = AssignmentLoader("utils/config.toml")
    loader.graph = MagicMock()
    
    sample_programmer.availability = 0
    
    matches = [
        {"person_id": "Dev1", "rfp_id": "Alpha", "rfp_title": "Alpha", "score": 80}
    ]
    
    assignments = loader.assign_based_on_matches(matches, sample_projects, [sample_programmer])
    
    assert len(assignments) == 0

def test_missing_project_lookup(sample_projects, sample_programmer):
    """Test correct handling when matched RFP doesn't link to a loaded Project."""
    loader = AssignmentLoader("utils/config.toml")
    loader.graph = MagicMock()
    
    matches = [
        {"person_id": "Dev1", "rfp_id": "Gamma", "rfp_title": "Gamma", "score": 80}
    ]
    
    assignments = loader.assign_based_on_matches(matches, sample_projects, [sample_programmer])
    
    # specific handling might vary, but currently it should skip if project not found
    assert len(assignments) == 0
