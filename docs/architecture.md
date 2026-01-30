# Architecture Overview

## Components

- **Streamlit UI**: entry point for interactive workflows in [streamlit_app.py](streamlit_app.py) and UI modules under [ui/](ui/).
- **Parsing layer**: RFP parsing and loading in [src/data/parsers/rfp_parser.py](src/data/parsers/rfp_parser.py).
- **Graph ingestion/query**: graph build and query utilities in [src/rag/graph/](src/rag/graph/).
- **Matching engine**: scoring and ranking logic in [src/core/matching/](src/core/matching/).
- **Assignments**: allocation logic in [src/data/parsers/assignment_loader.py](src/data/parsers/assignment_loader.py).
- **Persistence**: Neo4j connection and write helpers in [src/core/proxies/](src/core/proxies/).

## Data flow (PDFs → parsing → graph → matching → assignment)

1. **PDF/RFP input**
   - Source files live under [data/](data/) and [data_extended/](data_extended/).
2. **Parsing**
   - `RFPParser` extracts and structures RFP data in [src/data/parsers/rfp_parser.py](src/data/parsers/rfp_parser.py).
3. **Graph build**
   - Parsed data is stored in Neo4j via graph ingestion utilities in [src/rag/graph/](src/rag/graph/).
4. **Matching**
   - `MatchingEngine` and scoring logic in [src/core/matching/engine.py](src/core/matching/engine.py) and [src/core/matching/scoring.py](src/core/matching/scoring.py) compute candidate fit.
5. **Assignments**
   - Assignment logic (allocations, availability) runs in [src/data/parsers/assignment_loader.py](src/data/parsers/assignment_loader.py) and is exposed in the UI.

## Runtime entry points

- UI: [streamlit_app.py](streamlit_app.py)
- Pipeline module: [src/services/pipeline.py](src/services/pipeline.py)
- Experiment scripts: [scripts/](scripts/)
