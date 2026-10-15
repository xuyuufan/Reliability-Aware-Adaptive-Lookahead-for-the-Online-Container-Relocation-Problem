"""Refresh reviewable source/input and release hash inventories after edits."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    snapshot = ROOT / "baselines/_snapshot"
    inputs = [p for p in snapshot.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    inputs += [ROOT/p for p in ["configs/reproduction.json", "reproduction/runtime.py", "reproduction/runner.py",
                               "reproduction/reporting.py", "tools/reproduce_experiments.py", "tools/reproduce_results.py",
                               "requirements.txt", "requirements-experiments.txt", "data/reference_trajectories.csv",
                               "data/layouts/select20.json", "models/GP_RULE.json", "models/RIDGE_MODEL.json"]]
    inputs += list((ROOT/"baselines").glob("*/__init__.py")) + [ROOT/"baselines/__init__.py"]
    record = {"schema": "two_noise_reproduction_input_manifest_v1", "paths_are_repository_relative": True,
              "scope": ["subset_shuffle", "time_window_shuffle", "CLEAN_separate"],
              "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(inputs))}}
    (ROOT/"reproduction/INPUT_MANIFEST.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    ignored_roots = {".git", "runs", "reproduced", "validation_tmp", ".venv", ".pytest_cache"}
    destination = ROOT/"provenance/RELEASE_MANIFEST.csv"
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p == destination: continue
        relative = p.relative_to(ROOT)
        if relative.parts[0] in ignored_roots or "__pycache__" in relative.parts: continue
        files.append(p)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows({"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
                        for p in sorted(files))
    print(f"Bound {len(record['sha256'])} reproduction inputs; indexed {len(files)} release files")


if __name__ == "__main__":
    main()
