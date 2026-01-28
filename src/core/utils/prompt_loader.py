import functools
from pathlib import Path

@functools.lru_cache(maxsize=None)
def load_prompt(file_path: Path) -> str:
    """
    Reads a text file and caches the content.
    Useful for loading prompt templates from disk without re-reading files on every call.
    """
    return file_path.read_text()
