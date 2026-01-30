# Testing

## Run tests

From the repository root:

```bash
uv run python -m pytest -q
```

## What tests cover

- **Matching engine & scoring**: [tests/test_matching_engine.py](tests/test_matching_engine.py), [tests/test_scoring.py](tests/test_scoring.py)
- **Assignment loader**: [tests/test_assignment_loader.py](tests/test_assignment_loader.py)
- **End-to-end assignment flow**: [tests/test_e2e_assignment.py](tests/test_e2e_assignment.py)

## Notes

- Tests mock external services (Neo4j and LLMs) so they can run locally without Docker.
- If you change graph schema or scoring logic, update the relevant tests.
