"""Independently replay physical moves and verify paired public/private traces."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FAMILIES = {"subset_shuffle", "time_window_shuffle", "CLEAN"}


def verify(directory):
    with (directory/"MERGED_RAW_TRAJECTORIES.csv").open(encoding="utf-8", newline="") as f:
        rows = {r["trajectory_key"]: r for r in csv.DictReader(f)}
    layouts = {r["label_invariant_fingerprint"]: r for r in json.loads(
        (ROOT/"data/layouts/select20.json").read_text(encoding="utf-8-sig"))}
    starts, step_count, current = set(), 0, None
    with gzip.open(directory/"public_decision_trace.jsonl.gz", "rt", encoding="utf-8") as public, \
         gzip.open(directory/"evaluator_physics.jsonl.gz", "rt", encoding="utf-8") as private:
        def finish():
            if current is None: return
            assert retrieved == len(truth) and not any(stacks), current
            assert moves == int(rows[current]["R_raw"]) == int(rows[current]["decision_count"]), current
            assert actions.hexdigest() == rows[current]["action_sequence_sha256"], current

        def auto_retrieve():
            nonlocal retrieved
            while retrieved < len(truth):
                found = next((s for s in stacks if s and s[-1] == truth[retrieved]), None)
                if found is None: break
                found.pop(); retrieved += 1

        def check_frame(frame):
            assert frame["retrieval_count"] == retrieved and frame["relocation_count"] == moves
            assert frame["max_tiers"] == tiers
            assert not ({"environment", "provider", "forecast", "seed", "p", "family", "window_size", "layout_fingerprint"} & frame.keys())
            for stack, ranks, occupied in zip(stacks, frame["rank_tensor"], frame["occupied_mask"]):
                assert sum(occupied) == len(stack) and occupied == [True]*len(stack)+[False]*(tiers-len(stack))
                assert all(1 <= rank <= 22 if occupied[i] else rank == 0 for i, rank in enumerate(ranks))
            if retrieved < len(truth):
                source = next(i for i, s in enumerate(stacks) if truth[retrieved] in s)
                assert frame["rank_tensor"][source][stacks[source].index(truth[retrieved])] == 1
                legal = [i for i, s in enumerate(stacks) if i != source and len(s) < tiers]
                assert frame["legal_destinations"] == legal
                assert frame["legal_destination_mask"] == [i in legal for i in range(len(stacks))]

        for line in private:
            event = json.loads(line)
            key = event["trajectory_key"]
            if event["record_type"] == "trajectory_start":
                finish()
                assert key in rows and key not in starts
                starts.add(key); current = key
                row = rows[key]; layout = layouts[row["layout_fingerprint"]]
                assert row["family"] in ALLOWED_FAMILIES
                assert event["condition_evaluator_only"]["family"] == row["family"]
                assert event["initial_stacks_private"] == layout["initial_stacks"]
                assert event["true_retrieval_sequence_private"] == layout["retrieval_sequence"]
                forecast = event["fixed_forecast_private"]
                forecast_hash = hashlib.sha256(json.dumps(forecast, separators=(",", ":")).encode()).hexdigest()
                assert forecast_hash == row["F_hash"]
                assert str(event["forecast_seed_evaluator_only"] or "") == row["forecast_seed_evaluator_only"]
                stacks = [list(s) for s in layout["initial_stacks"]]
                truth, tiers = layout["retrieval_sequence"], layout["max_tiers"]
                retrieved, moves = 0, 0; actions = hashlib.sha256(); auto_retrieve()
                continue
            assert event["record_type"] == "physical_step" and key == current
            observed = json.loads(next(public))
            assert observed["trajectory_key"] == current and observed["decision_index"] == moves
            assert event["decision_index"] == moves
            assert event["physical_stacks_before_private"] == stacks
            assert event["retrieval_count_before"] == retrieved and event["relocation_count_before"] == moves
            check_frame(observed["public_frame_before"])
            source = next(i for i, s in enumerate(stacks) if truth[retrieved] in s)
            destination = observed["own_action"]
            assert destination != source and len(stacks[destination]) < tiers
            assert stacks[source][-1] != truth[retrieved]
            actions.update(f"{moves}:{destination}:{retrieved};".encode())
            stacks[destination].append(stacks[source].pop()); moves += 1; auto_retrieve()
            assert event["physical_stacks_after_private"] == stacks
            assert event["retrieval_count_after"] == retrieved and event["relocation_count_after"] == moves
            assert observed["terminal_after_action"] == (retrieved == len(truth))
            if retrieved < len(truth): check_frame(observed["public_frame_after"])
            else: assert observed["public_frame_after"] is None
            step_count += 1
        finish()
        assert next(public, None) is None
    assert starts == set(rows)
    return {"status": "PASS_INDEPENDENT_PHYSICAL_TRACE_REPLAY", "trajectories_verified": len(starts),
            "physical_steps_verified": step_count, "public_private_pairing_verified": True,
            "allowed_families_only": sorted(ALLOWED_FAMILIES), "raw_sha256": hashlib.sha256(
                (directory/"MERGED_RAW_TRAJECTORIES.csv").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"results/two_noise_families/reproduced_run")
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    payload = json.dumps(result, indent=2)+"\n"
    if args.receipt_out: args.receipt_out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
