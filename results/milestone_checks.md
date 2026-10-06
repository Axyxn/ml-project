# Milestone checks

All reported experiments use the real ULB CSV with SHA256
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

| Milestone | Evidence |
|---|---|
| 1. Setup/data | Original 284,807 rows / 492 frauds verified; five loader/split tests passed; pip check passed |
| 2. Preprocessing/EDA | 1,081 duplicate rows removed; 283,726 unique rows / 473 frauds; scaler/selection check passed; seven EDA/selection figures saved |
| 3. Baselines | Six full-training fits completed; two pipeline checks verify fold-local fits and no inference resampling; three metric checks passed |
| 4. Tuning/evaluation | All six configurations searched with 3 candidates × 3 folds; refitted on all 170,235 training rows; three real-artifact checks reproduce saved metrics and input transformations |
| 5. Notebook/report | All 28 notebook cells executed; zero error outputs; eight ordered sections and eleven embedded figures checked |
| 6. Streamlit | Manual and real-example single inference, batch example, valid CSV upload path, invalid upload message; local health and page HTTP 200 |

Final deployment choice: Random Forest without SMOTE, baseline hyperparameters, selected
by validation AP 0.8659. Validation F1 selected threshold 0.259462. Final test: 56,746
records, TN=56,641 / FP=10 / FN=22 / TP=73; precision 0.8795, recall 0.7684,
F1 0.8202, ROC-AUC 0.9538, AP 0.7792, trapezoidal PR-AUC 0.7791.

The threshold catches four more frauds and raises seven more false alerts than 0.5 on
test, while test F1 falls slightly. This was not used to revise the validation choice.

The CSV upload check injects actual file bytes at Streamlit's uploader boundary because
AppTest 1.50 lacks an upload setter. It exercises parsing, validation, scoring, display,
and the download control. HTTP startup is a separate check; no manual browser walkthrough
or screenshot is claimed. Synthetic fixtures serve unit checks only.

Reproduction: `python -m pytest -q --junitxml results/test_results.xml`,
`python scripts/execute_notebook.py`, and `python scripts/verify_server.py`.
The XML contains the final full-suite result. Additional execution records are saved
in `notebook_execution.json` and `server_verification.json`.
