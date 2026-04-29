#!/usr/bin/env python3
"""
Verification Suite Runner.

Reads the experiment registry from verification.yaml, creates the shared
output directory, and invokes each verification script with --output-dir
so that every script writes its own artifacts (JSONL, CSV, numpy) into
its designated sub-directory.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

from synapse.verification.config import load_verification_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_script(
    experiment_id: str,
    script_path: str,
    output_dir: str,
    stop_on_failure: bool = False,
) -> bool:
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        print(f"[ERROR] Script not found: {full_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running {experiment_id}: {full_path.name}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [sys.executable, str(full_path), "--output-dir", output_dir],
        env=env,
    )

    if result.returncode != 0:
        print(f"\n{experiment_id} FAILED (exit code {result.returncode})")
        if stop_on_failure:
            print("Stopping due to failure (--strict mode).")
            sys.exit(1)
        return False

    print(f"\n{experiment_id} COMPLETED")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verification Suite Runner")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        help="Experiment ID (e.g. VZ3-01) or 'all' to run everything.",
    )
    parser.add_argument(
        "--output-dir",
        default="verification_outputs",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop execution immediately if a script fails.",
    )
    args = parser.parse_args()

    config = load_verification_config()
    target = args.experiment.upper()

    if target == "ALL":
        print(f"Running {len(config)} verification experiments...")
        failures: list[str] = []
        for eid, exp_cfg in config.items():
            script = exp_cfg.get("script", "")
            success = run_script(eid, script, args.output_dir, args.strict)
            if not success:
                failures.append(eid)

        print(f"\n{'='*60}")
        if failures:
            print(f"SUITE FINISHED WITH {len(failures)} FAILURES: {', '.join(failures)}")
        else:
            print("SUITE FINISHED — all experiments completed.")
        print(f"{'='*60}")

    else:
        if target not in config:
            valid = ", ".join(sorted(config.keys()))
            print(f"Unknown experiment ID: {target}. Valid options: {valid}, or 'all'.")
            sys.exit(1)

        run_script(target, config[target]["script"], args.output_dir, args.strict)


if __name__ == "__main__":
    main()
