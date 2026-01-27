import argparse
import logging
import json
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
from src.core.matching.scoring import CandidateScoringEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MatchingEngine")

class MatchingEngine:
    def __init__(self):
        self.graph = Neo4jGraph()
        logger.info("✅ MatchingEngine initialized.")

    def match_candidates(self, rfp_id: str) -> List[Dict[str, Any]]:
        """
        Fetches RFP requirements and finds matching candidates.
        Returns a list of dicts with skill requirements.
        """
        # First, check if we can find the RFP by ID or Title
        query = """
        MATCH (r:RFP)
        WHERE r.id = $rfp_id OR r.title = $rfp_id
        OPTIONAL MATCH (r)-[:NEEDS]->(s:Skill)
        RETURN r.id as id, r.title as title, collect(s.name) as required_skills
        """
        result = self.graph.query(query, {"rfp_id": rfp_id})
        
        if not result:
            logger.warning(f"❌ RFP not found: {rfp_id}")
            return []
            
        rfp_data = result[0]
        logger.info(f"✅ Found RFP: {rfp_data['title']} with requirements: {rfp_data['required_skills']}")
        return rfp_data['required_skills']

    def rank_candidates(self, rfp_id: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Ranks candidates for a specific RFP using Cypher for efficiency.
        Includes saving the MATCHED_TO relationship in the graph.
        """
        
        # Use the centralized scoring engine to compute scores consistently
        scorer = CandidateScoringEngine(config_path="config/config.toml")

        # Fetch RFP requirements
        req_q = """
        MATCH (r:RFP)
        WHERE r.id = $rfp_id OR r.title = $rfp_id
        WITH r LIMIT 1
        OPTIONAL MATCH (r)-[needs:NEEDS]->(s:Skill)
        RETURN r.id as rfp_id, collect({name: s.name, mandatory: needs.is_mandatory, min_proficiency: needs.min_proficiency, preferred_certifications: needs.preferred_certifications}) as requirements
        """
        rres = self.graph.query(req_q, {"rfp_id": rfp_id})
        if not rres:
            logger.warning(f"❌ RFP not found: {rfp_id}")
            return []
        requirements = rres[0].get("requirements", [])

        # Fetch candidate basics
        cand_q = """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[hs:HAS_SKILL]->(s:Skill)
        OPTIONAL MATCH (p)-[:EARNED]->(c:Certification)
        RETURN p.name as person_id, p.email as email, p.location as location, p.description as description,
               p.years_experience as years_experience, p.availability as availability,
               collect(DISTINCT {name: s.name, proficiency: hs.proficiency}) as skills,
               collect(DISTINCT c.name) as certifications
        """
        try:
            candidates = self.graph.query(cand_q)
        except Exception as e:
            logger.error(f"❌ Error fetching candidates: {e}")
            return []

        all_scored = []
        for c in candidates:
            candidate = {
                "person_id": c.get("person_id"),
                "email": c.get("email"),
                "location": c.get("location"),
                "description": c.get("description"),
                "years_experience": c.get("years_experience") or 0,
                "availability": c.get("availability") or 0,
                "skills": c.get("skills") or [],
                "certifications": c.get("certifications") or []
            }
            score_info = scorer.calculate_score(candidate, requirements)
            score = score_info["score"]
            mandatory_met = score_info["mandatory_met"]
            logger.debug(f"Score for {candidate['person_id']}: {score} breakdown={score_info['breakdown']}")

            # Collect all scored candidates for UI ranking
            all_scored.append({
                "person_id": candidate["person_id"],
                "score": score,
                "mandatory_met": mandatory_met,
                "location": candidate.get("location"),
                "email": candidate.get("email"),
                "description": candidate.get("description"),
                "years_experience": candidate.get("years_experience"),
                "availability": candidate.get("availability"),
                "skills": candidate.get("skills"),
                "breakdown": score_info.get("breakdown")
            })

        # Persist MATCHED_TO for a relaxed set of candidates so the system is useful
        threshold = float(scorer.threshold_score)
        min_avail = int(scorer.min_needed)
        # Relaxed criteria: 70% of score threshold or half availability
        relaxed_score = threshold * 0.7 if threshold > 0 else 0
        relaxed_avail = max(0, int(min_avail * 0.5))
        persisted = 0
        # Ensure we always persist the top-N so UI/assignment can reference them
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        top_ids = {c["person_id"] for c in all_scored[:top_n]}

        for cand in all_scored:
            meets_strict = (cand["score"] >= threshold and cand["availability"] >= min_avail)
            meets_relaxed = (cand["score"] >= relaxed_score and cand["availability"] >= relaxed_avail)
            should_persist = (cand["person_id"] in top_ids) or meets_strict or meets_relaxed
            if not should_persist:
                continue
            try:
                m_q = """
                MATCH (p:Person {name: $person_id})
                MATCH (r:RFP)
                WHERE r.id = $rfp_id OR r.title = $rfp_id
                MERGE (p)-[m:MATCHED_TO]->(r)
                SET m.score = $score,
                    m.mandatory_met = $mandatory_met,
                    m.meets_threshold = $meets_threshold,
                    m.updated_at = datetime()
                RETURN m
                """
                self.graph.query(m_q, {
                    "person_id": cand["person_id"],
                    "rfp_id": rfp_id,
                    "score": cand["score"],
                    "mandatory_met": cand["mandatory_met"],
                    "meets_threshold": meets_strict
                })
                persisted += 1
            except Exception as e:
                logger.warning(f"Failed to persist MATCHED_TO for {cand['person_id']}: {e}")

        # sort and return top_n (return top candidates for UI regardless of persistence)
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        out = all_scored[:top_n]
        logger.info(f"✅ Ranked {len(out)} candidates for {rfp_id} (persisted={persisted}, threshold={scorer.threshold_score}, min_avail={scorer.min_needed})")
        for r in out:
            r["rfp_id"] = rfp_id
        return out

    def get_all_matches(self) -> List[Dict[str, Any]]:
        """
        Retrieves all MATCHED_TO relationships from the graph.
        Returns a list of dicts: {person_id, rfp_id, score}
        """
        query = """
        MATCH (p:Person)-[m:MATCHED_TO]->(r:RFP)
        RETURN p.name as person_id, r.id as rfp_id, r.title as rfp_title, m.score as score
        ORDER BY p.name, m.score DESC
        """
        try:
            results = self.graph.query(query)
            logger.info(f"✅ Retrieved {len(results)} matches from graph")
            return results
        except Exception as e:
            logger.error(f"❌ Error retrieving matches: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match and rank candidates for an RFP.")
    parser.add_argument("--rfp-id", type=str, required=True, help="ID or Title of the RFP")
    args = parser.parse_args()

    engine = MatchingEngine()
    
    # Run ranking
    ranked_candidates = engine.rank_candidates(args.rfp_id)
    
    # Output JSON to stdout
    print(json.dumps(ranked_candidates, indent=2))
