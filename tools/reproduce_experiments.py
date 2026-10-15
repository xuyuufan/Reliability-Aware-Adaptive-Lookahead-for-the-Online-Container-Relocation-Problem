"""Portable entrypoint: preflight, smoke, or full frozen experiment rerun."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
