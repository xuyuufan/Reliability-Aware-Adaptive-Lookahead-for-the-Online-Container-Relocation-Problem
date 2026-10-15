"""Fresh trajectories under the frozen protocol, compared with archived rows."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproduction.runtime import (ALGORITHMS, METHODS, REFERENCE, ROOT, SNAPSHOT,
                                  audit_imports, load_runtime, make_policy)

LOCK = ROOT / "reproduction/INPUT_MANIFEST.json"
RAW = REFERENCE
CONFIG = ROOT / "configs/reproduction.json"
_CONTEXT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    record = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in record["sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen reproduction input changed or is missing: {relative}")
    return {"manifest_sha256": sha(LOCK), "files_verified": len(record["sha256"])}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def plan(scope, selected_algorithms):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(config["algorithms"]) != ALGORITHMS or tuple(config["methods"]) != METHODS:
        raise ValueError("Configuration differs from the registered solver/method order")
    rows = json.loads((ROOT / "data/layouts/select20.json").read_text(encoding="utf-8-sig"))
    context = load_runtime()
    conditions = [c for c in context["comparison"].condition_specs()
                  if c["family"] in config["noise_families"] or c["family"] == "CLEAN"]
    if len(rows) != config["full_layouts"] or len(conditions) != config["full_conditions"]:
        raise ValueError("The complete frozen layout/condition matrix changed")
    if scope == "smoke":
        rows = [next(r for r in rows if r["parameter_group"] == geometry) for geometry in ("G1", "G2")]
        conditions = [c for c in conditions if c["condition_id"] in config["smoke_condition_ids"]]
    expected = len(rows)*len(conditions)*len(selected_algorithms)*len(METHODS)
    expected_default = config["smoke_rows"] if scope == "smoke" else config["full_rows"]
    if selected_algorithms == list(ALGORITHMS) and expected != expected_default:
        raise ValueError("Planned row count differs from the frozen reproduction configuration")
    reference = {r["trajectory_key"]: r for r in read_csv(RAW)}
    if len(reference) != config["full_rows"]:
        raise ValueError("Archived comparison data have missing or duplicate keys")
    jobs = []
    forecast_pairs = 0
    for row in rows:
        episode = context["runtime"]["OnlineCRPEpisode"].from_record(row)
        fingerprint = context["runtime"]["label_invariant_fingerprint"](episode)
        if fingerprint != row["label_invariant_fingerprint"]:
            raise ValueError("Layout fingerprint differs from the included registry")
        for condition in conditions:
            forecast, seed = context["comparison"]._forecast_for(episode, fingerprint, condition, context["runtime"])
            forecast_hash = context["comparison"]._public_hash(forecast)
            for algorithm in selected_algorithms:
                for method in METHODS:
                    key = "|".join((fingerprint, condition["condition_id"], algorithm, method))
                    archived = reference[key]
                    if archived["F_hash"] != forecast_hash or archived["forecast_seed_evaluator_only"] != ("" if seed is None else str(seed)):
                        raise ValueError(f"Forecast/seed differs from archive: {key}")
            jobs.append({"index": len(jobs), "row": row, "condition": condition,
                         "forecast": forecast, "seed": seed, "algorithms": selected_algorithms})
            forecast_pairs += 1
    return config, jobs, expected, forecast_pairs


def initialize_worker():
    global _CONTEXT
    _CONTEXT = load_runtime()
    import torch
    torch.set_num_threads(1)


def run_cell(job):
    """Each cell uses one identical full forecast across all 36 fresh runs."""
    from R.scrp.r_order_actor import HistoryBoundController
    from R.experiments.fixed_horizon_analysis import RAW_COLUMNS
    context = _CONTEXT
    rt, comparison = context["runtime"], context["comparison"]
    row, condition = job["row"], job["condition"]
    episode = rt["OnlineCRPEpisode"].from_record(row)
    fp = row["label_invariant_fingerprint"]
    directory = Path(job["output"])/"cells"/f'{job["index"]:04d}'
    directory.mkdir(parents=True)
    public = gzip.open(directory/"public_decision_trace.jsonl.gz", "wt", encoding="utf-8") if job["traces"] else None
    private = gzip.open(directory/"evaluator_physics.jsonl.gz", "wt", encoding="utf-8") if job["traces"] else None
    output = []
    try:
        for algorithm in job["algorithms"]:
            for method in METHODS:
                if time.monotonic() >= job["deadline"]:
                    raise TimeoutError("Reproduction wall-clock budget exhausted")
                started = time.monotonic()
                key = "|".join((fp, condition["condition_id"], algorithm, method))
                policy = make_policy(algorithm, method, context)
                controller = HistoryBoundController.fresh(policy)
                environment, provider = comparison._make_context(episode, job["forecast"], job["seed"], rt)
                frame = rt["public_frame"](episode, environment, provider)
                decisions, actor_seconds = 0, 0.0
                actions = hashlib.sha256()
                if private:
                    private.write(json.dumps(comparison._json_safe({
                        "record_type": "trajectory_start", "trajectory_key": key,
                        "initial_stacks_private": episode.initial_stacks,
                        "true_retrieval_sequence_private": episode.retrieval_sequence,
                        "fixed_forecast_private": job["forecast"],
                        "forecast_seed_evaluator_only": job["seed"],
                        "condition_evaluator_only": condition,
                    }), separators=(",", ":"))+"\n")
                while not environment.state.terminated:
                    if time.monotonic()-started > job["trajectory_cap"] or time.monotonic() >= job["deadline"]:
                        raise TimeoutError(f"Trajectory or run budget exhausted: {key}")
                    physical = rt["private_physical_stacks"](environment) if private else None
                    tick = time.perf_counter()
                    action, trace, decorated = controller.select(frame)
                    actor_seconds += time.perf_counter()-tick
                    if int(action) not in frame.legal_destinations:
                        raise ValueError(f"Illegal solver action: {key}")
                    before_retrieval = int(environment.state.retrieval_count)
                    before_moves = int(environment.relocation_count)
                    environment.step(int(action))
                    rt["sync_retrieval_events_with_audit"](provider, episode.retrieval_sequence,
                                                          before_retrieval, int(environment.state.retrieval_count))
                    if int(environment.relocation_count) != before_moves+1:
                        raise ValueError("Physical relocation conservation failed")
                    next_frame = None if environment.state.terminated else rt["public_frame"](episode, environment, provider)
                    actions.update(f"{decisions}:{int(action)}:{before_retrieval};".encode())
                    if public:
                        public.write(json.dumps(comparison._json_safe({
                            "trajectory_key": key, "decision_index": decisions,
                            "algorithm": algorithm, "method": method,
                            "public_frame_before": comparison._frame_record(decorated),
                            "own_action": int(action), "actor_trace": trace,
                            "public_frame_after": None if next_frame is None else comparison._frame_record(next_frame),
                            "terminal_after_action": bool(environment.state.terminated),
                        }), separators=(",", ":"))+"\n")
                        private.write(json.dumps(comparison._json_safe({
                            "record_type": "physical_step", "trajectory_key": key, "decision_index": decisions,
                            "physical_stacks_before_private": physical,
                            "physical_stacks_after_private": rt["private_physical_stacks"](environment),
                            "retrieval_count_before": before_retrieval,
                            "retrieval_count_after": int(environment.state.retrieval_count),
                            "relocation_count_before": before_moves,
                            "relocation_count_after": int(environment.relocation_count),
                            "provider_history_hits_private": provider.history.hits,
                        }), separators=(",", ":"))+"\n")
                    decisions += 1
                    if decisions > int(episode.num_containers)*max(1, int(episode.max_tiers)-1):
                        raise ValueError("Trajectory exceeded physical relocation bound")
                    frame = next_frame
                if int(environment.state.retrieval_count) != int(episode.num_containers):
                    raise ValueError("Trajectory terminated before all retrievals")
                output.append({
                    "trajectory_key": key, "layout_fingerprint": fp, "geometry": episode.parameter_group,
                    "condition_id": condition["condition_id"], "family": condition["family"],
                    "strength": condition["strength"], "p": condition["p"], "window_size": condition["window_size"],
                    "repeat": condition["repeat"], "algorithm": algorithm, "method": method,
                    "R_raw": int(environment.relocation_count), "N": int(episode.num_containers),
                    "R_over_N": int(environment.relocation_count)/int(episode.num_containers),
                    "forecast_seed_evaluator_only": job["seed"], "F_hash": comparison._public_hash(job["forecast"]),
                    "complete": True, "decision_count": decisions,
                    "wall_seconds": time.monotonic()-started, "actor_time_seconds": actor_seconds,
                    "eg_cached_moves": int(getattr(policy, "cached_moves", 0)),
                    "source_origin": "fresh_portable_rerun", "source_stage": job["scope"],
                    "action_sequence_sha256": actions.hexdigest(),
                })
        fields = (*RAW_COLUMNS, "action_sequence_sha256")
        with (directory/"raw_trajectories.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(output)
        audit_imports()
        return {"index": job["index"], "rows": len(output)}
    finally:
        if public: public.close()
        if private: private.close()


def compare_rows(rows, expected_count):
    reference = {r["trajectory_key"]: r for r in read_csv(RAW)}
    keys = set()
    # Timings and provenance necessarily change on a fresh run; scientific
    # observables, forecast pairing, completion, decisions and EG reuse must agree.
    fields = ("layout_fingerprint", "geometry", "condition_id", "family", "strength", "p",
              "window_size", "repeat", "algorithm", "method", "R_raw", "N", "R_over_N",
              "forecast_seed_evaluator_only", "F_hash", "complete", "decision_count", "eg_cached_moves")
    for row in rows:
        key = row["trajectory_key"]
        if key in keys or key not in reference:
            raise ValueError(f"Duplicate/unregistered rerun trajectory: {key}")
        keys.add(key)
        for field in fields:
            x, y = row[field], reference[key][field]
            if x.lower() == y.lower(): continue
            try:
                equal = math.isfinite(float(x)) and math.isfinite(float(y)) and abs(float(x)-float(y)) <= 1e-10
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(f"Archive mismatch: {key}: {field}: {x} != {y}")
    if len(keys) != expected_count:
        raise ValueError("Reproduction matrix is incomplete")
    return {"trajectory_rows_verified": len(keys), "fields_per_row": len(fields),
            "total_fields_verified": len(keys)*len(fields), "deterministic_mismatches": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cap-seconds", type=float, default=7200)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-traces", action="store_true", help="omit new per-step trace copies; counts and decisions are still checked")
    parser.add_argument("--analyze", action="store_true", help="full six-solver run only: regenerate and compare all tables/figures")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 32 or not math.isfinite(args.cap_seconds) or not 0 < args.cap_seconds <= 7200:
        parser.error("workers must be 1..32 and wall cap must be finite, positive and <=7200 seconds")
    if len(set(args.algorithms)) != len(args.algorithms): parser.error("duplicate algorithms")
    if args.analyze and (args.scope != "full" or args.algorithms != list(ALGORITHMS)):
        parser.error("--analyze requires the complete six-solver full scope")
    start = time.monotonic()
    inputs = verify_inputs()
    config, jobs, expected, pairs = plan(args.scope, args.algorithms)
    preflight = {"status": "PASS_PORTABLE_PREFLIGHT", "scope": args.scope, "planned_rows": expected,
                 "forecast_pairs_verified": pairs, "algorithms": args.algorithms,
                 "model_selection_or_training": False, "inputs": inputs,
                 "loaded_modules": audit_imports(), "new_trajectories": 0}
    if args.preflight:
        print(json.dumps(preflight, indent=2)); return 0
    target = (args.output_dir or ROOT/f"runs/{args.scope}").resolve()
    if target.exists(): raise ValueError(f"Output already exists: {target}; choose a new directory")
    target.mkdir(parents=True)
    write_json(target/"PREFLIGHT.json", preflight)
    deadline = start+args.cap_seconds
    for job in jobs:
        job.update(output=str(target), traces=not args.no_traces, deadline=deadline,
                   trajectory_cap=config["single_trajectory_cap_seconds"], scope=args.scope)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    receipt = {"scope": args.scope, "expected_rows": expected, "workers": args.workers,
               "inputs": inputs, "new_fits": 0, "new_model_selections": 0,
               "new_per_step_traces": not args.no_traces}
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
            futures = [pool.submit(run_cell, job) for job in jobs]
            for count, future in enumerate(as_completed(futures, timeout=max(1, deadline-time.monotonic())), start=1):
                future.result()
                if count == 1 or count % 20 == 0 or count == len(jobs):
                    print(f"Completed {count}/{len(jobs)} layout-condition cells", flush=True)
        paths = [target/"cells"/f'{job["index"]:04d}'/"raw_trajectories.csv" for job in jobs]
        rows = [row for path in paths for row in read_csv(path)]
        checks = compare_rows(rows, expected)
        destination = target/"MERGED_RAW_TRAJECTORIES.csv"
        with destination.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if not args.no_traces:
            for name in ("public_decision_trace.jsonl.gz", "evaluator_physics.jsonl.gz"):
                with (target/name).open("xb") as output:
                    for job in jobs:
                        with (target/"cells"/f'{job["index"]:04d}'/name).open("rb") as source:
                            shutil.copyfileobj(source, output)
        if verify_inputs() != inputs: raise ValueError("Reproduction inputs changed during execution")
        if args.scope == "full" and args.algorithms == list(ALGORITHMS):
            from R.experiments import fixed_horizon_analysis as a
            bank = a.validate_rows(rows, a.load_layout_registry())
            if bank.forecast_pair_count != 300: raise ValueError("Full forecast matrix incomplete")
        receipt.update(status="PASS_FULL_TRAJECTORY_REPRODUCTION" if expected == config["full_rows"] else "PASS_PARTIAL_TRAJECTORY_REPRODUCTION",
                       new_trajectories=len(rows), elapsed_seconds=time.monotonic()-start,
                       raw_sha256=sha(destination), comparison=checks, analysis_executed=False)
        write_json(target/"RUN_RECEIPT.json", receipt)
        if args.analyze:
            subprocess.run([sys.executable, "-B", str(ROOT/"tools/reproduce_results.py"),
                            "--input", str(destination), "--output-dir", str(target/"analysis")], check=True)
            receipt.update(status="PASS_FULL_EXPERIMENT_REPRODUCTION", analysis_executed=True,
                           analysis_receipt_sha256=sha(target/"analysis/REPRODUCTION_RECEIPT.json"),
                           elapsed_seconds=time.monotonic()-start)
            write_json(target/"RUN_RECEIPT.json", receipt)
        print(json.dumps(receipt, indent=2)); return 0
    except BaseException as error:
        receipt.update(status="FAIL_REPRODUCTION", error=f"{type(error).__name__}: {error}",
                       elapsed_seconds=time.monotonic()-start)
        write_json(target/"RUN_RECEIPT.json", receipt)
        raise
