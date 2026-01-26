import pytest
from unittest.mock import patch


class MockNeo4j:
    def __init__(self):
        # nodes keyed by label then key
        self.persons = {
            "Amanda Rivas": {"name": "Amanda Rivas", "availability": 100}
        }
        self.rfps = {
            "rfp-1": {
                "id": "rfp-1",
                "title": "Test RFP",
                "description": "Test description",
                "start_date": "2024-01-01",
                "duration_months": 6,
                "team_size": 1,
                "client": "ACME",
                "budget_range": "10k-20k",
                "location": "Remote",
                "remote_allowed": True
            }
        }
        self.projects = {}
        self.relationships = []

    def query(self, query, params=None):
        q = (query or "").strip()
        p = params or {}

        # calculate allocated for a person
        if q.startswith("MATCH (p:Person {name: $person_id})-[r:ASSIGNED_TO]->(:Project)"):
            pid = p.get("person_id")
            allocated = 0
            for rel in self.relationships:
                if rel["type"] == "ASSIGNED_TO" and rel["from_label"] == "Person" and rel["from_key"] == pid:
                    allocated += rel["props"].get("allocation_percentage", 0)
            return [{"allocated": allocated}]

        # update person availability
        if q.startswith("MATCH (p:Person {name: $person_id})\n        SET p.availability = $availability") or "SET p.availability = $availability" in q:
            pid = p.get("person_id")
            val = int(p.get("availability")) if p.get("availability") is not None else int(p.get("value", 100))
            # support both param names
            if p.get("availability") is None and p.get("value") is not None:
                val = int(p.get("value"))
            if pid in self.persons:
                self.persons[pid]["availability"] = val
            return [{"updated": 1}]

        # check project by rfp
        if "MATCH (p:Project)<-[:GENERATES]-(r:RFP {id: $rfp_id})" in q:
            rfp_id = p.get("rfp_id")
            for proj in self.projects.values():
                if proj.get("generated_from") == rfp_id:
                    return [{"id": proj["id"], "name": proj.get("name", proj["id"])}]
            return []

        # fetch RFP data and return for create_project_from_rfp
        if "MATCH (r:RFP {id: $rfp_id})" in q and "collect({skill" in q:
            rfp_id = p.get("rfp_id")
            r = self.rfps.get(rfp_id)
            if not r:
                return []
            return [{"r": r, "requirements": []}]

        # project create (detect CREATE (p:Project)
        if q.startswith("CREATE (p:Project"):
            proj_id = p.get("project_id")
            self.projects[proj_id] = {"id": proj_id, "name": p.get("name"), "generated_from": p.get("rfp_id")}
            return []

        # collect existing assignments for project
        if "OPTIONAL MATCH (p:Person)-[a:ASSIGNED_TO]->(pr)" in q and "collect({person_name: p.name" in q:
            project_id = p.get("project_id")
            rows = []
            for rel in self.relationships:
                if rel["type"] == "ASSIGNED_TO" and rel["to_label"] == "Project" and (rel["to_key"] == project_id or rel["to_key"] == project_id):
                    rows.append({"person_name": rel["from_key"], "allocation_percentage": rel["props"].get("allocation_percentage", 0)})
            return [{"assignments": rows}]

        # create ASSIGNED_TO relationship
        if "MERGE (p)-[a:ASSIGNED_TO]->(pr)" in q or "CREATE (p)-[a:ASSIGNED_TO]->(pr)" in q:
            person_name = p.get("person_name")
            project_title = p.get("project_title")
            alloc = p.get("allocation_percentage")
            # find project key by id/entity/name
            proj_key = project_title
            if project_title not in self.projects and project_title.startswith("PRJ-"):
                # project might be created earlier
                proj_key = project_title
            self.relationships.append({"type": "ASSIGNED_TO", "from_label": "Person", "from_key": person_name, "to_label": "Project", "to_key": proj_key, "props": {"allocation_percentage": float(alloc)}})
            return []

        # fallback
        return []


def test_end_to_end_assign_flow():
    # Patch Neo4jGraph to use MockNeo4j for all components
    with patch("langchain_neo4j.Neo4jGraph", return_value=MockNeo4j()):
        # Import here to ensure patched class is used in constructors
        from pipeline_service import PipelineService

        service = PipelineService()

        # Ensure RFP exists in mock (mock contains rfp-1)
        result = service.assign_selected_candidates_for_rfp("rfp-1", ["Amanda Rivas"], force=True)

        # Check result structure
        assert isinstance(result, dict)
        assert result.get("project_id") is not None
        assignments = result.get("assignments")
        assert isinstance(assignments, list)
        # One assignment expected for team_size=1
        assert len(assignments) == 1

        # Verify that mock graph recorded the relationship and updated availability
        mock_graph = service.pipeline.assignment_loader.graph
        # find assignment relationship
        rels = [r for r in mock_graph.relationships if r["type"] == "ASSIGNED_TO" and r["from_key"] == "Amanda Rivas"]
        assert len(rels) == 1
        # availability should be reduced to 0
        assert mock_graph.persons["Amanda Rivas"]["availability"] == 0
