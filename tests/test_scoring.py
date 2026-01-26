import pytest
from scoring import CandidateScoringEngine


def test_calculate_score_with_cert_and_proficiency():
    scorer = CandidateScoringEngine(config_path="utils/config.toml")

    candidate = {
        "skills": [{"name": "Python", "proficiency": "Advanced"}],
        "years_experience": 3,
        "availability": 80,
        "certifications": ["CertA"]
    }

    requirements = [{"name": "Python", "mandatory": True, "min_proficiency": 3, "preferred_certifications": ["CertA"]}]

    res = scorer.calculate_score(candidate, requirements)

    assert isinstance(res, dict)
    assert "score" in res and "breakdown" in res and "mandatory_met" in res

    # Compute expected total using scorer configuration (handles custom prof_map)
    prof_val = scorer.prof_map.get("Advanced", 1)
    skill_score = scorer.skill_match_weight + (prof_val * scorer.proficiency_weight) + scorer.certification_bonus
    exp_score = candidate["years_experience"] * scorer.experience_weight
    avail_score = (candidate["availability"] / 100.0) * (scorer.availability_weight * 100.0)
    expected_total = skill_score + exp_score + avail_score + scorer.mandatory_bonus

    assert round(res["score"], 2) == round(expected_total, 2)
    assert res["mandatory_met"] is True


def test_calculate_score_missing_mandatory_fails_mandatory_met():
    scorer = CandidateScoringEngine(config_path="utils/config.toml")

    candidate = {
        "skills": [],
        "years_experience": 0,
        "availability": 0,
        "certifications": []
    }

    requirements = [{"name": "Python", "mandatory": True, "min_proficiency": 3, "preferred_certifications": []}]

    res = scorer.calculate_score(candidate, requirements)

    assert res["mandatory_met"] is False
    # No skill, no experience, no availability -> score should be 0 (no mandatory bonus)
    assert res["score"] == 0.0
