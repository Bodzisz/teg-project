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

            # Refresh availability for selected persons (read-only here)
            avail_map = {}
            for pid in selected_person_ids:
                avail = self.pipeline.assignment_loader.calculate_availability(pid)
                # do NOT persist this refresh yet — avoid changing graph unless we create an assignment
                avail_map[pid] = avail

            # Fetch existing assignments for this project so we don't double-assign
            existing_assignments = []
            try:
                exist_q = """
                MATCH (pr:Project)
                WHERE pr.id = $project_id OR pr.name = $project_id
                OPTIONAL MATCH (p:Person)-[a:ASSIGNED_TO]->(pr)
                RETURN collect({person_name: p.name, allocation_percentage: a.allocation_percentage, start_date: a.start_date, end_date: a.end_date}) as assignments
                """
                exist_res = self.pipeline.assignment_loader.graph.query(exist_q, {"project_id": project_id})
                if exist_res:
                    existing_assignments = exist_res[0].get("assignments") or []
            except Exception:
                existing_assignments = []

            # Compute existing total allocation and remaining needed
            existing_total = 0
            assigned_persons = set()
            for a in existing_assignments:
                try:
                    alloc_val = float(a.get("allocation_percentage") or 0)
                except Exception:
                    alloc_val = 0
                existing_total += alloc_val
                if a.get("person_name"):
                    assigned_persons.add(a.get("person_name"))

            remaining_needed = max(0, 100 - existing_total)

            # If project already fully assigned, do not proceed and do not update any availability
            if existing_total >= 100:
                return {"project_id": project_id, "assignments": existing_assignments, "message": "Project already fully assigned (100%). No changes made."}

            # Allocation algorithm (only distribute remaining_needed across newly-selected persons)
            new_selected = [pid for pid in selected_person_ids if pid not in assigned_persons]
            if len(new_selected) == 0:
                # Nothing new to assign; return existing assignments
                return {"project_id": project_id, "assignments": existing_assignments, "message": "Selected persons are already assigned to this project."}

            if remaining_needed <= 0:
                return {"project_id": project_id, "assignments": existing_assignments, "message": "Project already fully assigned."}

            desired = round(remaining_needed / len(new_selected))
            alloc = {pid: 0 for pid in new_selected}

            # First pass: allocate up to desired or availability
            for pid in alloc:
                alloc[pid] = min(desired, avail_map.get(pid, 0))

            remaining = remaining_needed - sum(alloc.values())

            # Round-robin distribute remaining by 1% to those with capacity
            if remaining > 0:
                pids_with_capacity = [pid for pid in alloc if avail_map.get(pid, 0) - alloc[pid] > 0]
                if not pids_with_capacity:
                    pass
                else:
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
