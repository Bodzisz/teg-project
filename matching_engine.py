import argparse
import logging
import json
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
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

    def score_candidate(self, person: Dict[str, Any], rfp_skills: List[str]) -> int:
        """
        Calculates score for a single candidate based on skills, experience and availability.
        This is a helper function if we were to process in Python.
        """
        score = 0
        
        # Skill Match
        person_skills = [s['name'] for s in person.get('skills', [])]
        for r_skill in rfp_skills:
            if r_skill in person_skills:
                score += 10 # Base score for matching skill
                
                # Proficiency bonus
                # Find the specific skill proficiency
                skill_data = next((s for s in person['skills'] if s['name'] == r_skill), None)
                if skill_data:
                    proficiency = skill_data.get('proficiency', 'Beginner')
                    if proficiency == 'Intermediate': score += 3
                    elif proficiency == 'Advanced': score += 5
                    elif proficiency == 'Expert': score += 8
                    else: score += 1 # Beginner

        # Experience
        years_exp = person.get('years_experience', 0)
        if years_exp:
            try:
                score += int(years_exp) * 2
            except:
                pass

        # Availability
        availability = person.get('availability', 0)
        score += int(availability * 0.5) # 50 points for 100% availability

        return score

    def rank_candidates(self, rfp_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Ranks candidates for a specific RFP using Cypher for efficiency.
        Includes saving the MATCHED_TO relationship in the graph.
        """
        
        # We will use a comprehensive Cypher query to score and rank
        query = """
        MATCH (r:RFP)
        WHERE r.id = $rfp_id OR r.title = $rfp_id
        WITH r
        MATCH (p:Person)
        
        // Calculate Skill Score
        OPTIONAL MATCH (r)-[:NEEDS]->(req_skill:Skill)
        OPTIONAL MATCH (p)-[hs:HAS_SKILL]->(p_skill:Skill)
        WITH r, p, collect(req_skill.name) as required_skills, collect({name: p_skill.name, proficiency: hs.proficiency}) as person_skills
        
        WITH r, p, required_skills, person_skills,
             reduce(s = 0.0, req in required_skills | 
                s + CASE 
                    WHEN req IN [ps IN person_skills | ps.name] THEN 
                        10.0 + CASE toInteger([x IN person_skills WHERE x.name = req][0].proficiency)
                            WHEN 5 THEN 8.0
                            WHEN 4 THEN 5.0
                            WHEN 3 THEN 3.0
                            ELSE 1.0
                        END
                    ELSE 0.0
                END
             ) as skill_score
             
        // Experience and Availability Scores
        WITH r, p, skill_score,
             toInteger(coalesce(p.years_experience, 0)) * 2.0 as exp_score,
             coalesce(p.availability, 0) * 0.5 as avail_score
             
        // Total Score
        WITH r, p, skill_score + exp_score + avail_score as total_score
        WHERE total_score > 0
        
        // Persist the score
        MERGE (p)-[m:MATCHED_TO]->(r)
        SET m.score = total_score, m.updated_at = datetime()
        
        RETURN p.name as person_id, total_score as score
        ORDER BY score DESC
        LIMIT $top_n
        """
        
        try:
            results = self.graph.query(query, {"rfp_id": rfp_id, "top_n": top_n})
            logger.info(f"✅ Ranked {len(results)} candidates for {rfp_id}")
            # Enhance results with rfp_id for consistency if needed, though mostly used for display here
            for r in results:
                r['rfp_id'] = rfp_id
            return results
        except Exception as e:
            logger.error(f"❌ Error ranking candidates: {e}")
            return []

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
