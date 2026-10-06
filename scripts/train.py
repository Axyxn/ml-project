"""Run reproducible experiments from the project directory."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, TrainingConfig
from src.data import DatasetError
from src.training import run_baselines, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--stage", choices=["baselines", "all"], default="all")
    parser.add_argument("--tuning-iterations", type=int, default=3)
    parser.add_argument("--tuning-max-rows", type=int, default=60_000)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--resume-baselines", action="store_true", help="Reuse a checkpoint for the same exact data/config.")
    args = parser.parse_args()
    config = TrainingConfig(tuning_iterations=args.tuning_iterations,
                            tuning_max_rows=args.tuning_max_rows, n_jobs=args.n_jobs)
    try:
        if args.stage == "baselines":
            run_baselines(args.data, config)
        else:
            run_training(args.data, config, resume_baselines=args.resume_baselines)
    except (DatasetError, ValueError) as exc:
        print(f"Training cannot proceed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
