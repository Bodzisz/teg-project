### TEG Project

[Project description](https://github.com/wodecki/TEG_2025/blob/main/src/3.%20Retrieval%20Augmented%20Generation/07_Your_Project_TalentMatch/PRD.md)


This repository contains the Talent Matcher project (TEG). It provides tools to
ingest CVs and RFPs into Neo4j, rank candidates for RFPs, and assign programmers
to projects. The web UI uses Streamlit for interactive workflows.

## Quicklinks

- Source RFP/Projects/CVs: `data/`
- Pipeline loader: `comprehensive_pipeline.py` (loads CVs and RFPs into Neo4j)
- Streamlit UI: `streamlit_app.py`

---

## Docker

We use Docker Compose to run a local Neo4j instance. From the repository root (where
`docker-compose.yml` is located) run:

```bash
docker-compose up -d
```

To list running containers:

```bash
docker ps
```

To follow Neo4j logs:

```bash
docker-compose logs -f neo4j
```

To stop and remove the containers:

```bash
docker-compose down
```

Note: the Neo4j data is stored in a Docker volume defined in `docker-compose.yml`.

---

## Neo4j Browser

Once the Neo4j container is running, open the browser at:

```
http://localhost:7474
```

Log in with the username/password you configured in `docker-compose.yml`.

You can also run Cypher commands from the container:

```bash
docker exec -it <neo4j_container_name> bash
cypher-shell -u <user> -p <password>
# then run Cypher queries, exit with :exit
```

---

## Load data from `data/` into Neo4j

The repository includes a pipeline script that processes CVs and RFP PDFs and
creates nodes and relationships in Neo4j. To load data into the running Neo4j
instance, do the following:

1. Use the `uv` package manager to create an environment and install dependencies from `pyproject.toml`.

```bash
# Install uv if you don't have it (global install)
python -m pip install --user uv

# Sync dependencies and create the project environment
uv sync
```

`uv sync` creates an isolated virtual environment and installs the packages
from `pyproject.toml`. After that, run commands inside the environment with
`uv run`.

2. Run the comprehensive pipeline to process CVs and RFPs into Neo4j. By
  default the pipeline only loads CVs and RFPs (matching and assignments are
  handled interactively through the Streamlit UI):

```bash
uv run python comprehensive_pipeline.py --config utils/config.toml
```


---

## Run the Streamlit UI

Start the Streamlit app from the repository root:

```bash
uv run streamlit run streamlit_app.py
```

Open the provided local URL (usually `http://localhost:8501`) in your browser.

The Streamlit UI exposes buttons to run matching and assignments once you have
ingested data and selected a target RFP.

---

## Unit tests (pytest)

There are unit tests that verify matching and assignment logic. Tests mock the
Neo4j connection and scoring components, so a running Neo4j instance is not
required to run the tests.

1. Make sure you ran `uv sync` to create the project environment (see above).

2. Run the test suite from the repository root:

```bash
uv run python -m pytest -q
```

---

## Notes and troubleshooting

- If imports for `langchain_neo4j` or other third-party libraries fail during
  tests, the unit tests include light mocks so tests can run without the
  service dependencies.
- Adjust `utils/config.toml` to point to your Neo4j connection settings if
  the defaults do not match your environment.

