"""Pytest configuration: make the in-repo python/ importable without install."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # python/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
