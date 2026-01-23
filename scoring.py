import tomllib
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("ScoringEngine")


class CandidateScoringEngine:
    def __init__(self, config_path: str = "utils/config.toml"):
        self.config = {}
        p = Path(config_path)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    self.config = tomllib.load(f)
            except Exception:
                self.config = {}

        # defaults
        scoring = self.config.get("scoring", {})
        self.skill_match_weight = float(scoring.get("skill_match_weight", 10))
        self.proficiency_weight = float(scoring.get("proficiency_weight", 2))
        self.mandatory_bonus = float(scoring.get("mandatory_bonus", 20))
        self.certification_bonus = float(scoring.get("certification_bonus", 10))
        self.experience_weight = float(scoring.get("experience_weight", 1.5))
        self.availability_weight = float(scoring.get("availability_weight", 0.5))

        # assignment thresholds
        assignment = self.config.get("assignment", {})
        self.threshold_score = float(self.config.get("assignment", {}).get("threshold_score", 50))
        self.min_needed = int(self.config.get("assignment", {}).get("min_needed", 50))

        # proficiency map
        self.prof_map = {"Beginner": 1, "Intermediate": 3, "Advanced": 5, "Expert": 8}

        # Validate skills/proficiency config if present
        skills_cfg = self.config.get("skills", {})
        levels = skills_cfg.get("proficiency_levels", [])
        weights = skills_cfg.get("proficiency_weights", [])
        if levels and weights:
            if len(levels) != len(weights):
                logger.warning(f"proficiency_levels length ({len(levels)}) != proficiency_weights length ({len(weights)}). Using default prof_map for scoring.")
            else:
                try:
                    # Build prof_map from config
                    self.prof_map = {levels[i]: float(weights[i]) for i in range(len(levels))}
                    logger.info("Using proficiency mapping from config: %s", self.prof_map)
                except Exception as e:
                    logger.warning("Failed to build prof_map from config, using defaults: %s", e)

    def calculate_score(self, candidate: Dict[str, Any], rfp_requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate a numeric score for a candidate dict against rfp_requirements.
        candidate: {skills: [{name, proficiency}], years_experience, availability, certifications: [str]}
        rfp_requirements: [{name, mandatory, min_proficiency, preferred_certifications}]
        Returns dict with keys: score (float), mandatory_met (bool), breakdown (dict)
        """
        skill_score = 0.0
        mandatory_met = True
        certs = set(candidate.get("certifications") or [])

        for req in rfp_requirements:
            req_name = req.get("name") or req.get("skill") or req.get("skill_name")
            req_mand = bool(req.get("mandatory") or req.get("is_mandatory"))
            req_min_prof = req.get("min_proficiency")
            pref_certs = set(req.get("preferred_certifications") or req.get("preferred") or [])

            # find matching skill
            person_skill = None
            for s in candidate.get("skills", []):
                if s.get("name") == req_name:
                    person_skill = s
                    break

            if person_skill:
                base = self.skill_match_weight
                prof = person_skill.get("proficiency")
                prof_val = self.prof_map.get(prof, 1)
                prof_bonus = prof_val * self.proficiency_weight
                cert_bonus = 0.0
                if pref_certs and len(certs.intersection(pref_certs)) > 0:
                    cert_bonus = self.certification_bonus
                skill_score += base + prof_bonus + cert_bonus
            else:
                if req_mand:
                    mandatory_met = False

        exp = float(candidate.get("years_experience") or 0)
        avail = float(candidate.get("availability") or 0)

        exp_score = exp * self.experience_weight
        avail_score = (avail / 100.0) * (self.availability_weight * 100.0)

        total = skill_score + exp_score + avail_score

        # bonus if all mandatory requirements met
        if mandatory_met:
            total += self.mandatory_bonus

        breakdown = {
            "skill_score": skill_score,
            "exp_score": exp_score,
            "avail_score": avail_score,
            "mandatory_bonus": self.mandatory_bonus if mandatory_met else 0.0,
        }

        return {"score": round(total, 2), "mandatory_met": mandatory_met, "breakdown": breakdown}
