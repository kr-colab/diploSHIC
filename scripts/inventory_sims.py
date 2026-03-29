#!/usr/bin/env python
"""Scan exampleApplication/ and write a simulation manifest to benchmarks/sim_manifest.json."""

import gzip
import json
import os
import sys


def parse_ms_header(filepath):
    """Read the first line of an ms-format file and return (program, n_samples, n_sims, seq_len).

    seq_len is extracted from the discoal positional argument (3rd token).
    """
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt") as f:
        header = f.readline().strip()
    tokens = header.split()
    program = tokens[0]
    n_samples = int(tokens[1])
    n_sims = int(tokens[2])
    seq_len = int(tokens[3])
    return program, n_samples, n_sims, seq_len


def count_reps(filepath):
    """Count the number of '//' lines (simulation replicates) in an ms-format file."""
    opener = gzip.open if filepath.endswith(".gz") else open
    count = 0
    with opener(filepath, "rt") as f:
        for line in f:
            if line.strip() == "//":
                count += 1
    return count


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_dir = os.path.join(repo_root, "exampleApplication")
    out_dir = os.path.join(repo_root, "benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(sim_dir):
        sys.exit(f"Simulation directory not found: {sim_dir}")

    ms_files = sorted(
        f for f in os.listdir(sim_dir) if f.endswith(".msOut.gz") or f.endswith(".msOut")
    )

    if not ms_files:
        sys.exit(f"No ms output files found in {sim_dir}")

    manifest = {"sim_dir": sim_dir, "files": []}

    for fname in ms_files:
        filepath = os.path.join(sim_dir, fname)
        program, n_samples, n_sims_header, seq_len = parse_ms_header(filepath)
        n_reps = count_reps(filepath)

        # Classify sweep type from filename
        if fname.startswith("hard_"):
            sweep_type = "hard"
            sweep_index = int(fname.replace("hard_", "").replace(".msOut.gz", "").replace(".msOut", ""))
        elif fname.startswith("soft_"):
            sweep_type = "soft"
            sweep_index = int(fname.replace("soft_", "").replace(".msOut.gz", "").replace(".msOut", ""))
        elif fname.startswith("neut"):
            sweep_type = "neutral"
            sweep_index = None
        else:
            sweep_type = "unknown"
            sweep_index = None

        entry = {
            "filename": fname,
            "sweep_type": sweep_type,
            "sweep_index": sweep_index,
            "simulator": program,
            "n_samples": n_samples,
            "n_reps_header": n_sims_header,
            "n_reps_actual": n_reps,
            "seq_len": seq_len,
        }
        manifest["files"].append(entry)
        status = "OK" if n_reps == n_sims_header else "MISMATCH"
        print(f"  {fname}: {n_reps}/{n_sims_header} reps, {n_samples} samples, {seq_len} bp [{status}]")

    # Summary
    by_type = {}
    for f in manifest["files"]:
        t = f["sweep_type"]
        by_type.setdefault(t, {"n_files": 0, "total_reps": 0})
        by_type[t]["n_files"] += 1
        by_type[t]["total_reps"] += f["n_reps_actual"]

    manifest["summary"] = by_type

    out_path = os.path.join(out_dir, "sim_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSummary:")
    for sweep_type, info in sorted(by_type.items()):
        print(f"  {sweep_type}: {info['n_files']} files, {info['total_reps']} total reps")
    print(f"\nManifest written to {out_path}")


if __name__ == "__main__":
    main()
