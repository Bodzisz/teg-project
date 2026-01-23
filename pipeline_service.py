import logging
from comprehensive_pipeline import GraphPipeline

class PipelineService:
    def __init__(self):
        self.pipeline = GraphPipeline(config_path="utils/config.toml")

    def assign_programmers(self, project_id: str):
        try:
            logger = logging.getLogger("PipelineService")
            logger.info("🔍 Stage 4: Assigning programmers to Projects...")
            programmers = self.pipeline.assignment_loader.load_programmers_from_graph()
            for p in programmers:
                availability = self.pipeline.assignment_loader.calculate_availability(p.id)
                self.pipeline.assignment_loader.update_graph_with_availability(p.id, availability)
            assignments_summary = self.pipeline.assignment_loader.assign_candidates_to_single_project(project_id)
            self.pipeline.assignment_loader.save_assignments_to_neo4j(assignments_summary)

            return assignments_summary
        except Exception as e:
            raise
    
    def assign_programmers_for_rfp(self, rfp_id: str):
        try:
            project_query = """
            MATCH (p:Project)<-[:GENERATES]-(r:RFP {id: $rfp_id})
            RETURN p.id as id, p.name as name
            """
            existing_project = self.pipeline.assignment_loader.graph.query(project_query, {"rfp_id": rfp_id})
            if existing_project:
                project_id = existing_project[0]["id"]
            else:
                project = self.pipeline.rfp_parser.create_project_from_rfp(rfp_id)
                project_id = project.get("id")
            programmers = self.pipeline.assignment_loader.load_programmers_from_graph()
            for p in programmers:
                availability = self.pipeline.assignment_loader.calculate_availability(p.id)
                self.pipeline.assignment_loader.update_graph_with_availability(p.id, availability)
            assignments_summary = self.pipeline.assignment_loader.assign_candidates_to_single_project(project_id)
            self.pipeline.assignment_loader.save_assignments_to_neo4j(assignments_summary)
            return {"project_id": project_id, "assignments": assignments_summary}
        except Exception as e:
            raise
