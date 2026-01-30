# Data Model (Neo4j)

This document describes the **minimum** node labels and relationships used by the matching and assignment flow. Confirm the full schema in the graph builder and queries under [src/rag/graph/](src/rag/graph/).

## Node labels (minimum expected)

- **Person**: candidate profile (id, name, availability, seniority)
- **Skill**: normalized skill (id, name, category)
- **RFP**: request for proposal (id, title, start_date, team_size)
- **Project**: generated project (id, name/title, start_date, end_date)

## Relationships (minimum expected)

- **(Person)-[:HAS_SKILL]->(Skill)**
  - properties: level, years (if available)
- **(RFP)-[:REQUIRES]->(Skill)**
  - properties: mandatory, min_level
- **(RFP)-[:GENERATES]->(Project)**
- **(Person)-[:MATCHED_TO]->(RFP)**
  - properties: score, mandatory_met
- **(Person)-[:ASSIGNED_TO]->(Project)**
  - properties: allocation_percentage, start_date, end_date

## Key properties used in queries

- Person: id, name
- Skill: id
- RFP: id, title
- Project: id, name/title, start_date, end_date

## Indexes and constraints (recommended)

- Unique: Person.id, Skill.id, RFP.id, Project.id
- Index: Skill.id (case-insensitive matching via `toLower()` in queries)

## Notes

- Matching and assignment logic uses labels and relationships referenced in tests, e.g. [tests/test_assignment_loader.py](tests/test_assignment_loader.py).
- If you extend the schema (e.g., certifications, roles, availability calendars), update queries in [src/rag/graph/querier.py](src/rag/graph/querier.py) and matching logic in [src/core/matching/](src/core/matching/).
