"""Execute all notebook cells with this interpreter and save executed outputs."""
import json
import asyncio
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = ROOT / "notebooks/credit_card_fraud_detection.ipynb"
    runtime = ROOT / ".cache/jupyter-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    nb = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Use the current venv without requiring a global kernelspec installation.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, timeout=1800, kernel_manager=manager,
                            resources={"metadata": {"path": str(ROOT)}})
    start = time.perf_counter()
    print(f"Executing {path} with {sys.executable}", flush=True)
    client.execute()
    nbformat.write(nb, path)
    errors = [output for cell in nb.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    record = {"notebook": path.relative_to(ROOT).as_posix(), "cells": len(nb.cells), "errors": len(errors),
              "seconds": time.perf_counter() - start, "python": sys.version.split()[0],
              "environment": "project virtual environment"}
    (ROOT / "results/notebook_execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
