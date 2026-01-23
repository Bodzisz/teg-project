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
    
    def assign_programmers_for_rfp(self, rfp_id: str, force: bool = False):
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
            # If force is set, temporarily bump assignment probability to 1.0
            if force:
                try:
                    self.pipeline.assignment_loader.config.setdefault("assignment", {})["assignment_probability"] = 1.0
                except Exception:
                    pass

            programmers = self.pipeline.assignment_loader.load_programmers_from_graph()
            for p in programmers:
                availability = self.pipeline.assignment_loader.calculate_availability(p.id)
                self.pipeline.assignment_loader.update_graph_with_availability(p.id, availability)
            assignments_summary = self.pipeline.assignment_loader.assign_candidates_to_single_project(project_id)
            self.pipeline.assignment_loader.save_assignments_to_neo4j(assignments_summary)
            return {"project_id": project_id, "assignments": assignments_summary}
        except Exception as e:
            raise

    def assign_selected_candidates_for_rfp(self, rfp_id: str, selected_person_ids: list, force: bool = False):
        """Assign specific selected persons to the Project generated from an RFP.
        This implements the equal-split + cap-by-availability + round-robin redistribution logic.
        """
        try:
            # Ensure project exists for this RFP
            project_query = """
            MATCH (p:Project)<-[:GENERATES]-(r:RFP {id: $rfp_id})
            RETURN p.id as id, p.name as name, p.start_date as start_date, p.end_date as end_date
            """
            existing_project = self.pipeline.assignment_loader.graph.query(project_query, {"rfp_id": rfp_id})
            if existing_project:
                proj_rec = existing_project[0]
                project_id = proj_rec.get("id") or proj_rec.get("name")
                project_name = proj_rec.get("name")
                project_start = proj_rec.get("start_date")
                project_end = proj_rec.get("end_date")
            else:
                project = self.pipeline.rfp_parser.create_project_from_rfp(rfp_id)
                project_id = project.get("id")
                project_name = project.get("name")
                project_start = project.get("start_date")
                project_end = project.get("end_date")

            # Refresh availability for selected persons
            avail_map = {}
            for pid in selected_person_ids:
                avail = self.pipeline.assignment_loader.calculate_availability(pid)
                # Update the graph with the refreshed availability
                self.pipeline.assignment_loader.update_graph_with_availability(pid, avail)
                avail_map[pid] = avail

            # Allocation algorithm
            n = len(selected_person_ids)
            if n == 0:
                return {"project_id": project_id, "assignments": []}

            desired = round(100 / n)
            alloc = {pid: 0 for pid in selected_person_ids}

            # First pass: allocate up to desired or availability
            for pid in alloc:
                alloc[pid] = min(desired, avail_map.get(pid, 0))

            remaining = 100 - sum(alloc.values())

            # Round-robin distribute remaining by 1% to those with capacity
            if remaining > 0:
                pids_with_capacity = [pid for pid in alloc if avail_map.get(pid, 0) - alloc[pid] > 0]
                i = 0
                while remaining > 0 and pids_with_capacity:
                    pid = pids_with_capacity[i % len(pids_with_capacity)]
                    if avail_map.get(pid, 0) - alloc[pid] > 0:
                        alloc[pid] += 1
                        remaining -= 1
                    else:
                        pids_with_capacity = [p for p in pids_with_capacity if avail_map.get(p, 0) - alloc[p] > 0]
                    i += 1

            assignments = []
            # Build assignment records for those with allocation > 0
            for pid, allocation in alloc.items():
                if allocation <= 0:
                    # Skip zero allocations
                    continue

                # Update person's availability immediately in graph
                new_avail = max(0, int(avail_map.get(pid, 0) - allocation))
                self.pipeline.assignment_loader.update_graph_with_availability(pid, new_avail)

                # Compose assignment record consistent with save_assignments_to_neo4j
                assignments.append({
                    "person_name": pid,
                    "project_title": project_name or project_id,
                    "allocation_percentage": allocation,
                    "start_date": project_start,
                    "end_date": project_end
                })

            # Persist assignments
            if assignments:
                self.pipeline.assignment_loader.save_assignments_to_neo4j(assignments)

            return {"project_id": project_id, "assignments": assignments}
        except Exception as e:
            raise
