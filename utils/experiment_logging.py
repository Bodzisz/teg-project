"""Experiment metadata logging utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import sys
import os


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_experiment_metadata(run_name: str, metadata: Dict[str, Any], output_path: Path | str = "results/experiment_metadata.json") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": run_name,
        "cwd": os.getcwd(),
        "python_version": sys.version,
        **metadata,
    }

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    else:
        existing = []

    if isinstance(existing, dict):
        existing = [existing]

    existing.append(entry)
    output_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def collect_prompt_sources(paths: Dict[str, str]) -> Dict[str, str]:
    """Read prompt contents from a mapping of name -> path string."""
    out: Dict[str, str] = {}
    for key, path_str in paths.items():
        out[key] = _safe_read_text(Path(path_str))
    return out
