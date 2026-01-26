import sys
import types
import pytest
from unittest.mock import patch

# Provide minimal fake modules so importing the production modules doesn't fail
sys.modules.setdefault('langchain_neo4j', types.SimpleNamespace(Neo4jGraph=lambda *a, **k: None))
sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

from matching_engine import MatchingEngine


class MockGraph:
    def query(self, query, params=None):
        q = (query or "").strip()
        # RFP requirements query
        if "OPTIONAL MATCH (r)-[needs:NEEDS]->(s:Skill)" in q or "collect({name: s.name" in q:
            return [{"requirements": [{"name": "Python", "mandatory": True, "min_proficiency": 3, "preferred_certifications": None}], "rfp_id": "rfp-1", "title": "Test RFP"}]

        # Candidates query
        if "MATCH (p:Person)" in q and "p.name as person_id" in q:
            return [
                {"person_id": "alice", "email": "a@example.com", "location": "X", "description": "dev", "years_experience": 5, "availability": 80, "skills": [{"name": "Python", "proficiency": 4}], "certifications": []},
                {"person_id": "bob", "email": "b@example.com", "location": "Y", "description": "dev", "years_experience": 3, "availability": 50, "skills": [{"name": "Python", "proficiency": 3}], "certifications": []}
            ]

        # For persistence or other queries, return an empty result
        return []


class FakeScorer:
    def __init__(self, config_path=None):
        # consistent thresholds for tests
        self.threshold_score = 50
        self.min_needed = 20

    def calculate_score(self, candidate, requirements):
        # deterministic scoring: alice > bob
        if candidate.get("person_id") == "alice":
            return {"score": 90, "mandatory_met": True, "breakdown": {}}
        return {"score": 60, "mandatory_met": True, "breakdown": {}}


def test_rank_candidates_orders_by_score_and_returns_rfp_id():
    with patch("matching_engine.Neo4jGraph", return_value=MockGraph()), \
         patch("matching_engine.CandidateScoringEngine", FakeScorer):
        engine = MatchingEngine()
        out = engine.rank_candidates("rfp-1", top_n=2)

        assert isinstance(out, list)
        assert len(out) == 2
        # alice should be first (higher fake score)
        assert out[0]["person_id"] == "alice"
        assert out[1]["person_id"] == "bob"
        # rfp_id should be attached to each result
        assert all(item.get("rfp_id") == "rfp-1" for item in out)


def test_rank_candidates_explainability_and_score_consistency():
    # Mock graph returns proficiency as strings matching prof_map
    class MockGraph2:
        def query(self, query, params=None):
            q = (query or "").strip()
            if "OPTIONAL MATCH (r)-[needs:NEEDS]->(s:Skill)" in q or "collect({name: s.name" in q:
                return [{"requirements": [{"name": "Python", "mandatory": True, "min_proficiency": 3, "preferred_certifications": ["CertA"]}], "rfp_id": "rfp-1", "title": "Test RFP"}]
            if "MATCH (p:Person)" in q and "p.name as person_id" in q:
                return [
                    {"person_id": "alice", "email": "a@example.com", "location": "X", "description": "dev", "years_experience": 5, "availability": 80, "skills": [{"name": "Python", "proficiency": "Advanced"}], "certifications": ["CertA"]},
                    {"person_id": "bob", "email": "b@example.com", "location": "Y", "description": "dev", "years_experience": 3, "availability": 50, "skills": [{"name": "Python", "proficiency": "Intermediate"}], "certifications": []}
                ]
            return []

    with patch("matching_engine.Neo4jGraph", return_value=MockGraph2()):
        # Use real scorer
        from scoring import CandidateScoringEngine
        engine = MatchingEngine()
        out = engine.rank_candidates("rfp-1", top_n=2)

        assert len(out) == 2
        # breakdown present and mandatory flag present
        for item in out:
            assert "breakdown" in item and "mandatory_met" in item
            breakdown = item["breakdown"]
            # breakdown keys
            assert set(breakdown.keys()) >= {"skill_score", "exp_score", "avail_score", "mandatory_bonus"}

        # Verify score equals breakdown components and mandatory bonus
        for item in out:
            breakdown = item["breakdown"]
            computed = round(breakdown.get("skill_score", 0) + breakdown.get("exp_score", 0) + breakdown.get("avail_score", 0) + breakdown.get("mandatory_bonus", 0), 2)
            assert round(item["score"], 2) == computed
