from __future__ import annotations

import os
import sys
import subprocess
import argparse
from pathlib import Path

from synapse.empirical.config import load_empirical_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_script(
    experiment_id: str,
    script_path: str,
    output_dir: str,
    stop_on_failure: bool = False,
) -> bool:
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"  [SKIP] {experiment_id}: script not found at {full_path}")
        return False

    cmd = [sys.executable, str(full_path), "--output-dir", output_dir]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  [FAIL] {experiment_id} (exit {result.returncode})")
        if stop_on_failure:
            raise RuntimeError(f"Experiment {experiment_id} failed; stopping.")
        return False
    print(f"  [OK]   {experiment_id}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Z3 empirical suite.")
    parser.add_argument("--output-dir", default="empirical_outputs")
    parser.add_argument("--experiment", default="all", help="Experiment ID (e.g. EZ3-01) or 'all'.")
    parser.add_argument("--strict", action="store_true", help="Stop execution immediately if a script fails.")
    args = parser.parse_args()

    config = load_empirical_config()
    target = args.experiment.upper()

    if target == "ALL":
        print(f"Running {len(config)} empirical experiments...")
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
            return 1
        exp_cfg = config[target]
        script = exp_cfg.get("script", "")
        success = run_script(target, script, args.output_dir, args.strict)
        if not success:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
