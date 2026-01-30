### Elite Match AI

This repository implements the Elite Match AI system: it ingests programmer
profiles and RFPs, builds a knowledge graph, ranks candidates for RFPs, and
provides an interactive Streamlit UI for reviewing matches and assignments.

Project layout (top-level highlights):

- **Files**: [README.md](README.md), [main.py](main.py), [streamlit_app.py](streamlit_app.py), [pyproject.toml](pyproject.toml), [docker-compose.yml](docker-compose.yml)
- **Data**: [data/](data/) and [data_extended/](data_extended/) (source CVs, projects, RFPs)
- **Source**: [src/](src/) — core code, matching engine, RAG modules, parsers, and services
- **Scripts**: [scripts/](scripts/) — helpers and demos
- **Neo4j**: [neo4j/](neo4j/) — local DB state when using Docker Compose
- **Tests**: [tests/](tests/) — unit and e2e tests

Quick pointers

- **Streamlit UI**: Run the app from the repository root using Streamlit:

```bash
uv run streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (typically http://localhost:8501).

- **Pipeline loader**: The ingestion and pipeline logic lives under the `src` package. To run the pipeline module directly (loads/parses data and writes to the graph), run:

```bash
uv run python -m src.services.pipeline --config config/config.toml
```

Adjust the command if you prefer calling a different runner (for example, a
custom script in `scripts/`).

- **Neo4j (Docker Compose)**: A Docker Compose file is provided for a local
  Neo4j instance. From the repository root:

```bash
docker-compose up -d
```

To follow logs:

```bash
docker-compose logs -f neo4j
```

To stop and remove containers:

```bash
docker-compose down
```

Neo4j Browser is usually available at http://localhost:7474 (or the Bolt
endpoint at 7687). Update connection settings in `config/config.toml` if needed.

- **Development environment**: This project uses `uv` to create and manage an
  isolated environment from `pyproject.toml`. Install and sync dependencies:

```bash
python -m pip install --user uv
uv sync
```

Run commands inside the environment using `uv run`, e.g. `uv run python ...`.

- **Tests (pytest)**: Run the test suite from the repository root:

```bash
uv run python -m pytest -q
```

Notes and troubleshooting

- Tests include mocks for external services so a running Neo4j instance is not
  required for unit tests.
- If connectors or optional libraries (e.g. langchain-related packages) cause
  import errors during local development, install the extras or adjust tests
  that mock those modules.
- If you want, I can also:
  - run the test suite locally and report failures
  - add example commands for common workflows
  - commit this README change for you

See [streamlit_app.py](streamlit_app.py) and the code under [src/](src/) for
implementation details and example usage.

