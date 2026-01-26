import sys
import types
import pytest
from unittest.mock import patch

# Provide minimal fake modules so importing the production modules doesn't fail
sys.modules.setdefault('langchain_neo4j', types.SimpleNamespace(Neo4jGraph=lambda *a, **k: None))
sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

from parsers.assignment_loader import AssignmentLoader


class MockGraphAssignment:
    def query(self, query, params=None):
        q = (query or "").strip()
        # project query
        if q.startswith("MATCH (p:Project)"):
            return [{"id": "proj-1", "name": "Project 1", "start_date": "2024-01-01", "end_date": "2024-06-01", "team_size": 2}]

        # rfp that generated project
        if "MATCH (r:RFP)-[:GENERATES]->(p:Project)" in q:
            return [{"rfp_id": "rfp-1", "rfp_title": "Test RFP"}]

        # matched candidates for RFP
        if "MATCH (person:Person)-[m:MATCHED_TO]->(r:RFP {id: $rfp_id})" in q:
            return [
                {"person_name": "alice", "person_id": "alice", "score": 90, "mandatory_met": True},
                {"person_name": "bob", "person_id": "bob", "score": 60, "mandatory_met": True}
            ]

        # availability aggregation
        if q.strip().startswith("MATCH (p:Person {name: $person_id})-[r:ASSIGNED_TO]->(:Project)"):
            # return allocated sums based on requested person
            pid = params.get("person_id") if params else None
            if pid == "alice":
                return [{"allocated": 20}]
            if pid == "bob":
                return [{"allocated": 50}]
            return [{"allocated": 0}]

        # accept update queries and others
        return []


def test_assign_candidates_to_single_project_distributes_allocations_and_updates_availability():
    with patch("parsers.assignment_loader.Neo4jGraph", return_value=MockGraphAssignment()):
        loader = AssignmentLoader(config_path="utils/config.toml")
        assignments = loader.assign_candidates_to_single_project("proj-1")

        assert isinstance(assignments, list)
        # team_size is 2 -> expect two assignments
        assert len(assignments) == 2

        allocs = [a["allocation_percentage"] for a in assignments]
        # allocations should sum to ~100
        assert round(sum(allocs), 1) == 100.0

        # check that returned entries have expected keys
        for a in assignments:
            assert "person_name" in a and "project_title" in a and "allocation_percentage" in a
