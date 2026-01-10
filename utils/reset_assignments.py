import logging
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Ładowanie zmiennych środowiskowych
load_dotenv(override=True)

# Konfiguracja loggera
logger = logging.getLogger("ResetAssignments")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AssignmentResetter:
    def __init__(self):
        self.graph = Neo4jGraph()
        logger.info("✅ AssignmentResetter initialized.")

    def reset_assignments(self):
        """
        Usuwa wszystkie relacje ASSIGNED_TO między projektami a programistami.
        """
        query = """
        MATCH (proj:Project)-[r:ASSIGNED_TO]->(p:Person)
        DELETE r
        """
        self.graph.query(query)
        logger.info("✅ Wszystkie przypisania zostały usunięte.")

    def reset_projects(self):
        """
        Usuwa wszystkie węzły Project wraz z ich relacjami.
        """
        query = """
        MATCH (proj:Project)
        DETACH DELETE proj
        """
        self.graph.query(query)
        logger.info("✅ Wszystkie projekty zostały usunięte.")

    def reset_programmers(self):
        """
        Usuwa wszystkie węzły Person wraz z ich relacjami.
        """
        query = """
        MATCH (p:Person)
        DETACH DELETE p
        """
        self.graph.query(query)
        logger.info("✅ Wszyscy programiści zostali usunięci.")

    def reset_specific_project(self, project_id: str):
        """
        Usuwa przypisania tylko dla wybranego projektu.
        """
        query = """
        MATCH (proj:Project {id: $project_id})-[r:ASSIGNED_TO]->(p:Person)
        DELETE r
        """
        self.graph.query(query, {"project_id": project_id})
        logger.info(f"✅ Przypisania dla projektu {project_id} zostały usunięte.")

    def delete_specific_project(self, project_id: str):
        """
        Usuwa węzeł Project wraz z jego relacjami.
        """
        query = """
        MATCH (proj:Project {id: $project_id})
        DETACH DELETE proj
        """
        self.graph.query(query, {"project_id": project_id})
        logger.info(f"✅ Projekt {project_id} został usunięty.")

    def reset_specific_programmer(self, programmer_id: str):
        """
        Usuwa przypisania tylko dla wybranego programisty.
        """
        query = """
        MATCH (proj:Project)-[r:ASSIGNED_TO]->(p:Person {id: $programmer_id})
        DELETE r
        """
        self.graph.query(query, {"programmer_id": programmer_id})
        logger.info(f"✅ Przypisania dla programisty {programmer_id} zostały usunięte.")

    def delete_specific_programmer(self, programmer_id: str):
        """
        Usuwa węzeł Person wraz z jego relacjami.
        """
        query = """
        MATCH (p:Person {id: $programmer_id})
        DETACH DELETE p
        """
        self.graph.query(query, {"programmer_id": programmer_id})
        logger.info(f"✅ Programista {programmer_id} został usunięty.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resetuj przypisania w Neo4j.")
    parser.add_argument("--mode", choices=["all", "project", "programmer", "delete_projects", "delete_programmers", "delete_project", "delete_programmer"], default="all", help="Tryb resetu")
    parser.add_argument("--id", type=str, help="ID projektu lub programisty (w trybie project/programmer/delete_project/delete_programmer)")
    args = parser.parse_args()

    resetter = AssignmentResetter()

    if args.mode == "all":
        resetter.reset_assignments()
    elif args.mode == "project" and args.id:
        resetter.reset_specific_project(args.id)
    elif args.mode == "programmer" and args.id:
        resetter.reset_specific_programmer(args.id)
    elif args.mode == "delete_projects":
        resetter.reset_projects()
    elif args.mode == "delete_programmers":
        resetter.reset_programmers()
    elif args.mode == "delete_project" and args.id:
        resetter.delete_specific_project(args.id)
    elif args.mode == "delete_programmer" and args.id:
        resetter.delete_specific_programmer(args.id)
    else:
        logger.error("❌ Niepoprawne argumenty. Użyj --mode [all|project|programmer|delete_projects|delete_programmers|delete_project|delete_programmer] oraz --id w odpowiednich trybach.")
