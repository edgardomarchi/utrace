"""Execute a jupytext notebook of this directory for one model and save it under notebooks/_build/.

    UTRACE_NB_MODEL=fc UTRACE_NB_QUICK=0 python notebooks/run_notebook.py periodic_verification_demo

Cells tagged ``only-<model>`` are kept only when that model is selected (the hand-written Inferences are
model-specific). The executed notebook is ``<name>_<model>_<mode>.ipynb`` for a non-default model and
``<name>_<mode>.ipynb`` for the default one, as before.
"""
import os
import sys
from pathlib import Path

import jupytext
import nbclient

NOTEBOOKS = {  # name: (default model, available models)
    "periodic_verification_demo": ("cnn", ("cnn", "fc")),
    "acdc_periodic_verification": ("km", ("km", "bm")),
    "model_comparison": (None, (None,)),       # reads the four full-mode results files; no model, no mode
}

name = sys.argv[1]
default, available = NOTEBOOKS[name]
model = os.environ.get("UTRACE_NB_MODEL", default) if default else None
if model not in available:
    raise SystemExit(f"{name}: UTRACE_NB_MODEL must be one of {available}, not {model!r}")
mode = "quick" if os.environ.get("UTRACE_NB_QUICK", "1") != "0" else "full"
here = Path(__file__).resolve().parent

nb = jupytext.read(here / f"{name}.py")
nb.cells = [c for c in nb.cells
            if all(t == f"only-{model}" for t in c.metadata.get("tags", []) if t.startswith("only-"))]
nbclient.NotebookClient(nb, timeout=-1, resources={"metadata": {"path": str(here.parent)}}).execute()
tag = "" if model == default else f"{model}_"
out = here / "_build" / (f"{name}.ipynb" if default is None else f"{name}_{tag}{mode}.ipynb")
jupytext.write(nb, out)
print("wrote", out)
