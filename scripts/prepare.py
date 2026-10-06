"""Milestone 2: audit, training-only EDA, feature selection, and split checks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reporting import prepare_project

if __name__ == "__main__":
    splits, audit, ranking, prep = prepare_project()
    print(f"Removed {audit['duplicates_removed']:,} duplicates; {audit['clean_rows']:,} records remain.")
    for name, (X, y) in splits.items():
        print(f"{name}: {len(X):,} rows, {int(y.sum())} frauds ({100*y.mean():.3f}%).")
    print(ranking.to_string(index=False))
    print("Selected:", ", ".join(prep.named_steps["select"].selected_features_))
