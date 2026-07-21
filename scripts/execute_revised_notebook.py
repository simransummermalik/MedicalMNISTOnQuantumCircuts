"""Execute the research notebook, retain outputs, and save a categorized copy."""

from pathlib import Path
import shutil
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "MedicalMSINTstuff.ipynb"
OUTPUT_COPY = ROOT / "output" / "notebooks" / "MedicalMSINTstuff.executed.ipynb"
LOG_PATH = ROOT / "output" / "logs" / "notebook_execution.log"

OUTPUT_COPY.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

started = time.time()
notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
client = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
    record_timing=True,
)

try:
    client.execute(cwd=str(ROOT))
except Exception:
    nbformat.write(notebook, NOTEBOOK_PATH)
    LOG_PATH.write_text(
        f"status=failed\nelapsed_seconds={time.time() - started:.2f}\n"
    )
    raise
else:
    nbformat.write(notebook, NOTEBOOK_PATH)
    shutil.copy2(NOTEBOOK_PATH, OUTPUT_COPY)
    LOG_PATH.write_text(
        f"status=success\nelapsed_seconds={time.time() - started:.2f}\n"
        f"notebook={NOTEBOOK_PATH.name}\ncopy={OUTPUT_COPY.relative_to(ROOT)}\n"
    )
    print(f"Executed notebook in {time.time() - started:.1f} seconds")
    print(f"Saved executed copy to {OUTPUT_COPY}")
