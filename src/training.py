"""Six fair baselines, followed by fold-local tuning and frozen test evaluation."""
import time
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Memory
from joblib import parallel_config
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import ROOT, TrainingConfig, make_output_dirs
from src.data import audit_and_clean, load_dataset, split_dataset, stratified_sample
from src.evaluation import compute_metrics, plot_evaluation, plot_final_threshold, select_threshold
from src.preprocessing import make_preprocessing_steps
from src.reporting import prepare_project, write_json, write_findings

ALGORITHMS = ["Logistic Regression", "Decision Tree", "Random Forest"]


def make_model(algorithm, config):
    if algorithm == "Logistic Regression":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=config.seed)
    if algorithm == "Decision Tree":
        return DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=config.seed)
    if algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=2,
                                      max_samples=0.7, n_jobs=config.n_jobs, random_state=config.seed)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def make_pipeline(algorithm, use_smote, config=None, root=ROOT):
    config = config or TrainingConfig()
    sampler = SMOTE(sampling_strategy=config.smote_ratio, k_neighbors=5, random_state=config.seed) if use_smote else "passthrough"
    # Cached fold-local transformations are shared across compatible candidates.
    # Cache reuse never shares a fitted transformer between different training folds.
    return Pipeline(
        make_preprocessing_steps(config) + [("smote", sampler), ("classifier", make_model(algorithm, config))],
        memory=Memory(Path(root) / ".cache/pipelines", verbose=0))


def load_splits(data_path, config):
    clean, audit = audit_and_clean(load_dataset(data_path))
    with Path(data_path).open("rb") as handle:
        audit["data_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return split_dataset(clean, config), audit


def describe_fit(pipeline, algorithm, use_smote, stage, seconds, X_train, y_train, X_val, y_val):
    name = f"{algorithm} | {'SMOTE' if use_smote else 'None'} | {stage}"
    train_scores = pipeline.predict_proba(X_train)[:, 1]
    val_scores = pipeline.predict_proba(X_val)[:, 1]
    row = {"experiment": name, "algorithm": algorithm, "imbalance": "SMOTE" if use_smote else "None",
           "stage": stage, "partition": "validation", "fit_seconds": seconds,
           "train_average_precision": compute_metrics(y_train, train_scores)["average_precision"],
           "train_f1": compute_metrics(y_train, train_scores)["f1"],
           **compute_metrics(y_val, val_scores)}
    print(f"  Validation AP={row['average_precision']:.4f}, precision={row['precision']:.4f}, "
          f"recall={row['recall']:.4f}, F1={row['f1']:.4f}; fit {seconds:.1f}s", flush=True)
    return name, row, val_scores


def run_baselines(data_path=ROOT / "data/creditcard.csv", config=None, root=ROOT):
    config = config or TrainingConfig()
    root = Path(root)
    make_output_dirs(root)
    splits, audit = load_splits(data_path, config)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    experiments, rows, scores = {}, [], {}
    for algorithm in ALGORITHMS:
        for use_smote in (False, True):
            print(f"Fitting {algorithm}, {'SMOTE' if use_smote else 'no resampling'} ...", flush=True)
            pipeline = make_pipeline(algorithm, use_smote, config, root)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            seconds = time.perf_counter() - start
            name, row, val_scores = describe_fit(
                pipeline, algorithm, use_smote, "baseline", seconds, X_train, y_train, X_val, y_val)
            experiments[name], scores[name] = pipeline, val_scores
            rows.append(row)
            pd.DataFrame(rows).to_csv(root / "results/baseline_validation.csv", index=False)
    checkpoint = {"config": config.to_dict(), "audit": audit, "rows": rows, "experiments": experiments}
    joblib.dump(checkpoint, root / "models/baseline_checkpoint.joblib", compress=3)
    write_json(root / "results/training_config.json", config.to_dict())
    plot_evaluation(y_val, scores, root, prefix="baseline_validation")
    print("Milestone 3 complete: all six baselines fitted and validation plots saved.", flush=True)
    return checkpoint


def parameter_space(algorithm):
    if algorithm == "Logistic Regression":
        return {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__tol": [1e-4, 1e-3]}
    if algorithm == "Decision Tree":
        return {"classifier__max_depth": [4, 8, 12, None],
                "classifier__min_samples_leaf": [1, 2, 5, 10],
                "classifier__criterion": ["gini", "entropy"]}
    return {"classifier__n_estimators": [40, 60, 80],
            "classifier__max_depth": [8, 14, None],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features": ["sqrt", 0.5],
            "classifier__max_samples": [0.5, 0.7]}


def run_tuning(checkpoint, splits, config, root):
    root = Path(root)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_search, y_search = stratified_sample(X_train, y_train, config.tuning_max_rows, config.seed)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    fold_table = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_search, y_search), 1):
        fold_table.append({"fold": fold, "train_rows": len(train_idx),
                           "train_frauds": int(y_search.iloc[train_idx].sum()),
                           "validation_rows": len(val_idx), "validation_frauds": int(y_search.iloc[val_idx].sum())})
    pd.DataFrame(fold_table).to_csv(root / "results/cv_fold_summary.csv", index=False)
    tuning_rows, candidates = [], []
    experiments = checkpoint["experiments"].copy()
    rows = checkpoint["rows"].copy()
    for algorithm in ALGORITHMS:
        for use_smote in (False, True):
            print(f"Tuning {algorithm}, {'SMOTE' if use_smote else 'no resampling'}: "
                  f"{config.tuning_iterations} candidates x {config.cv_folds} folds, "
                  f"{len(X_search):,} real rows ({int(y_search.sum())} frauds) ...", flush=True)
            pipeline = make_pipeline(algorithm, use_smote, config, root)
            search = RandomizedSearchCV(
                pipeline, parameter_space(algorithm), n_iter=config.tuning_iterations,
                cv=cv, scoring="average_precision", n_jobs=config.n_jobs,
                random_state=config.seed, refit=False, return_train_score=True,
                error_score="raise", pre_dispatch=config.n_jobs)
            start = time.perf_counter()
            # Bound native threads within CV workers and avoid uncontrolled oversubscription.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                search.fit(X_search, y_search)
            search_seconds = time.perf_counter() - start
            tuned = clone(pipeline).set_params(**search.best_params_)
            start = time.perf_counter()
            tuned.fit(X_train, y_train)
            refit_seconds = time.perf_counter() - start
            name, row, _ = describe_fit(tuned, algorithm, use_smote, "tuned", refit_seconds,
                                       X_train, y_train, X_val, y_val)
            experiments[name] = tuned
            rows.append(row)
            best_idx = search.best_index_
            tuning_rows.append({"experiment": name, "algorithm": algorithm,
                                "imbalance": "SMOTE" if use_smote else "None",
                                "search_rows": len(X_search), "search_frauds": int(y_search.sum()),
                                "candidates": config.tuning_iterations, "folds": config.cv_folds,
                                "cv_average_precision": search.best_score_,
                                "cv_ap_std": search.cv_results_["std_test_score"][best_idx],
                                "search_seconds": search_seconds, "refit_seconds": refit_seconds,
                                "best_params": json.dumps(search.best_params_, sort_keys=True)})
            for idx, params in enumerate(search.cv_results_["params"]):
                candidates.append({"experiment": name, "candidate": idx + 1,
                                   "params": json.dumps(params, sort_keys=True),
                                   "mean_train_ap": search.cv_results_["mean_train_score"][idx],
                                   "mean_cv_ap": search.cv_results_["mean_test_score"][idx],
                                   "std_cv_ap": search.cv_results_["std_test_score"][idx]})
            pd.DataFrame(rows).to_csv(root / "results/validation_comparison.csv", index=False)
            pd.DataFrame(tuning_rows).to_csv(root / "results/tuning_summary.csv", index=False)
            pd.DataFrame(candidates).to_csv(root / "results/tuning_candidates.csv", index=False)
            print(f"  Best CV AP={search.best_score_:.4f}; parameters={search.best_params_}", flush=True)
    return experiments, rows


def finalize_experiments(experiments, validation_rows, splits, audit, config, root):
    root = Path(root)
    validation = pd.DataFrame(validation_rows)
    # Selection is complete before computing ANY test prediction or metric.
    chosen = validation.sort_values(["average_precision", "f1", "experiment"],
                                    ascending=[False, False, True]).iloc[0]
    chosen_name = chosen.experiment
    best_model = experiments[chosen_name]
    X_val, y_val = splits["validation"]
    threshold, threshold_table = select_threshold(y_val, best_model.predict_proba(X_val)[:, 1])
    threshold_table.to_csv(root / "results/threshold_analysis.csv", index=False)
    selection = {"experiment": chosen_name, "validation_ap": float(chosen.average_precision),
                 "threshold": threshold, "criterion": "Highest validation AP; threshold maximizes validation F1",
                 "test_used_for_selection": False}
    write_json(root / "results/frozen_selection.json", selection)
    print(f"Frozen choice: {chosen_name}; validation threshold={threshold:.6f}. Evaluating test now.", flush=True)
    X_test, y_test = splits["test"]
    rows = validation_rows.copy()
    test_predictions = {}
    validation_predictions = {}
    for name, pipeline in experiments.items():
        source_row = validation[validation.experiment == name].iloc[0]
        scores = pipeline.predict_proba(X_test)[:, 1]
        test_predictions[name] = scores
        validation_predictions[name] = pipeline.predict_proba(X_val)[:, 1]
        rows.append({**source_row.to_dict(), "partition": "test", **compute_metrics(y_test, scores)})
    for partition, (X, y) in [("validation", (X_val, y_val)), ("test", (X_test, y_test))]:
        rows.append({"experiment": "Always legitimate", "algorithm": "Dummy reference", "imbalance": "None",
                     "stage": "reference", "partition": partition, "fit_seconds": 0,
                     **compute_metrics(y, np.zeros(len(y)))})
        scores = best_model.predict_proba(X)[:, 1]
        rows.append({**chosen.to_dict(), "experiment": f"{chosen_name} | selected threshold",
                     "stage": "selected threshold", "partition": partition, **compute_metrics(y, scores, threshold)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "results/model_comparison.csv", index=False)
    plot_evaluation(y_val, validation_predictions, root, prefix="validation")
    plot_evaluation(y_test, test_predictions, root, prefix="test")
    final_scores = test_predictions[chosen_name]
    plot_final_threshold(y_test, final_scores, threshold, threshold_table, root)
    baseline = validation[validation.stage == "baseline"]
    before_after = baseline.merge(validation[validation.stage == "tuned"], on=["algorithm", "imbalance"],
                                  suffixes=("_before", "_after"))
    cols = ["algorithm", "imbalance"] + [f"{metric}_{stage}" for metric in
            ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc"]
            for stage in ["before", "after"]]
    before_after[cols].to_csv(root / "results/before_after_tuning.csv", index=False)
    # Remove training cache paths from deployable artifacts.
    best_model.memory = None
    selector = best_model.named_steps["select"]
    selected_features = selector.selected_features_
    from sklearn.pipeline import Pipeline as SklearnPipeline
    preprocessor = SklearnPipeline([(name, best_model.named_steps[name])
                                   for name in ("imputer", "scaler", "select")])
    joblib.dump(best_model, root / "models/best_model.joblib", compress=3)
    joblib.dump(best_model.named_steps["scaler"], root / "models/scaler.joblib", compress=3)
    joblib.dump(preprocessor, root / "models/preprocessor.joblib", compress=3)
    joblib.dump(selected_features, root / "models/selected_features.joblib")
    from src.config import FEATURES
    import sklearn
    import imblearn
    metadata = {**selection, "algorithm": chosen.algorithm, "imbalance": chosen.imbalance,
                "stage": chosen.stage, "config": config.to_dict(), "raw_features": FEATURES,
                "selected_features": selected_features, "data_audit": audit,
                "final_test_metrics": compute_metrics(y_test, final_scores, threshold),
                "default_threshold_test_metrics": compute_metrics(y_test, final_scores),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "imblearn": imblearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
                "probability_calibrated": False, "training_rows": len(splits["train"][0])}
    write_json(root / "models/metadata.json", metadata)
    # Reproducible real held-out examples; labels are included for classroom demonstration.
    examples = pd.concat([X_test[y_test == 0].head(3), X_test[y_test == 1].head(3)])
    examples.assign(Class=y_test.loc[examples.index]).to_csv(root / "results/example_transactions.csv", index=False)
    write_findings(comparison, metadata, root)
    print("Milestone 4 complete: tuned comparisons, frozen test metrics, figures, and artifacts saved.", flush=True)
    return metadata


def run_training(data_path=ROOT / "data/creditcard.csv", config=None, root=ROOT, resume_baselines=False):
    config = config or TrainingConfig()
    root = Path(root)
    if config.tuning_iterations < 1 or config.tuning_max_rows < 10_000 or config.n_jobs < 1:
        raise ValueError("Use >=1 tuning iteration, >=10,000 search rows, and >=1 worker.")
    splits, audit = load_splits(data_path, config)
    audit_path = root / "results/data_audit.json"
    previous_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    if previous_audit.get("config") != config.to_dict() or previous_audit.get("data_sha256") != audit["data_sha256"]:
        print("Preparing dataset and training-only analysis ...", flush=True)
        prepare_project(data_path, config, root)
    checkpoint_path = root / "models/baseline_checkpoint.joblib"
    checkpoint = None
    if resume_baselines and checkpoint_path.exists():
        checkpoint = joblib.load(checkpoint_path)
        if checkpoint["config"] != config.to_dict() or checkpoint["audit"].get("data_sha256") != audit["data_sha256"]:
            raise ValueError("Baseline checkpoint does not match this dataset/config. Run without --resume-baselines.")
        print("Resuming the six verified baseline fits for this exact dataset/config.", flush=True)
    if checkpoint is None:
        checkpoint = run_baselines(data_path, config, root)
    experiments, rows = run_tuning(checkpoint, splits, config, root)
    return finalize_experiments(experiments, rows, splits, audit, config, root)
