"""Project paths and explicit, reproducible experiment settings."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"
SEED = 42
DATASET_PAGE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DATASET_MIRROR = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


@dataclass
class TrainingConfig:
    seed: int = SEED
    test_size: float = 0.20
    validation_size: float = 0.20  # Fraction of the entire cleaned dataset.
    selected_k: int = 20
    selection_max_rows: int = 60_000
    smote_ratio: float = 0.10  # Fraud/legitimate ratio AFTER SMOTE; not 50:50.
    cv_folds: int = 3
    tuning_iterations: int = 3
    tuning_max_rows: int = 60_000
    n_jobs: int = 2

    def to_dict(self):
        return asdict(self)


def make_output_dirs(root=ROOT):
    for name in ("data", "models", "figures", "results", "notebooks"):
        (Path(root) / name).mkdir(parents=True, exist_ok=True)
