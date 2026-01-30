# UI Flows (Streamlit)

This describes the typical user journey in the Streamlit UI.

## 1) Select RFP

- User chooses a target RFP to analyze.
- UI logic lives in [ui/rfp.py](ui/rfp.py) and is orchestrated by [streamlit_app.py](streamlit_app.py).

## 2) Run matching

- User triggers candidate matching for the selected RFP.
- Scoring and ranking are executed by `MatchingEngine` in [src/core/matching/engine.py](src/core/matching/engine.py) and `scoring.py` in [src/core/matching/scoring.py](src/core/matching/scoring.py).
- Results are displayed in [ui/candidates.py](ui/candidates.py).

## 3) Review candidates

- UI displays ranked candidates with scores and (optionally) mandatory requirement checks.
- The matching output is backed by data stored in Neo4j.

## 4) Approve / assign to project

- User accepts assignments and allocations.
- Assignment operations are implemented in [src/data/parsers/assignment_loader.py](src/data/parsers/assignment_loader.py) and written to Neo4j.

## 5) Optional chat/Q&A

- The chat/QA flow is implemented in [ui/chat.py](ui/chat.py) and typically uses RAG to answer questions about the graph and data.
