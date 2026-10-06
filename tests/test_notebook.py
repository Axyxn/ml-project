import nbformat
from src.config import ROOT


def test_notebook_has_eight_ordered_sections_and_executed_outputs():
    nb = nbformat.read(ROOT / "notebooks/credit_card_fraud_detection.ipynb", as_version=4)
    nbformat.validate(nb)
    text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    sections = ["Problem Definition", "Dataset", "Data Preprocessing", "Exploratory Data Analysis",
                "Model Development", "Model Evaluation", "Model Comparison", "Deployment / Demonstration"]
    positions = [text.index(f"## {number}. {title}") for number, title in enumerate(sections, 1)]
    assert positions == sorted(positions)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) >= 11
