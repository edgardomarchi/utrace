"""Properties of the package as an importable artifact, not of the API surface.

These assert what `import utrace` does to the process: which global JAX configuration it leaves in place, and which heavy dependencies it does NOT pull in - torch, and (since the imports in utils.py were deferred into the two functions that actually need them) matplotlib and pandas. They are grouped because they share that subject, not their mechanism - the torch/matplotlib/pandas absence tests run in a subprocess because pytest's own suite imports all three into this process elsewhere.
"""

import subprocess, sys

import pytest

def test_core_does_not_import_torch():
    code = (
        "import sys; import utrace; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_core_does_not_import_matplotlib_or_pandas():
    """`import utrace` reaches utils.py only for `_bucket_size`; plot_scores (matplotlib) and
    class_wise_performance (pandas) are the only two functions in that module that need either
    package, and their imports are deferred into the function bodies. This pins that the core
    does not pay for plotting/reporting dependencies at import time.
    """
    code = (
        "import sys; import utrace; "
        "assert 'matplotlib' not in sys.modules and 'pandas' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith('matplotlib') or m.startswith('pandas'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_x64_is_enabled():
    import utrace  # noqa: F401
    import jax.numpy as jnp
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64


def test_deferred_matplotlib_and_pandas_imports_resolve_on_call():
    """Companion to test_core_does_not_import_matplotlib_or_pandas: absence at import time is
    only useful if the deferred imports still resolve correctly once plot_scores /
    class_wise_performance are actually called. Runs in-process (unlike the subprocess test
    above) since it only needs to prove the functions work and that calling them makes
    matplotlib/pandas importable on demand, not that they were absent beforehand. Uses the
    non-interactive 'Agg' matplotlib backend since this runs headless under pytest.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from utrace.utils import plot_scores, class_wise_performance

    fig, ax = plt.subplots()
    plot_scores(
        alpha=0.1,
        scores=np.array([0.1, 0.2, 0.3, 0.4]),
        quantiles=np.array([0.2]),
        method="lac",
        ax=ax,
    )
    plt.close(fig)
    assert "matplotlib" in sys.modules

    df = class_wise_performance(
        y_new=np.array([0, 0, 1, 1]),
        y_set=np.array([
            [True, False],
            [True, True],
            [False, True],
            [True, True],
        ]),
        classes=[0, 1],
    )
    assert "pandas" in sys.modules
    assert list(df["class"]) == [0, 1]
    assert list(df["avg. set size"]) == [1.5, 1.5]


def test_torch_helpers_import_does_not_pull_matplotlib():
    """utils/pytorch/helpers.py is reached only when a caller explicitly imports it (e.g.
    tests/integration/torch/test_golden_mnist.py importing flatten_batch) - it is not on the
    `import utrace` path, so it is a different property from the ones above. It used to have
    its own unconditional `import matplotlib.pyplot as plt`, which meant `utrace[torch]` alone
    was not enough to use flatten_batch, a tensor-reshaping helper that never touches
    matplotlib; `utrace[viz]` was needed too. That import is now deferred into the one function
    (view_classify) that actually needs it. This pins that importing the helpers module itself
    does not pull matplotlib in - it says nothing about `import utrace` staying torch-free,
    which is test_core_does_not_import_torch's job.

    tests/core/ is torch-free by convention, but this one test necessarily needs torch present
    to import a torch-only helper module at all, so it skips cleanly via pytest.importorskip
    rather than failing when torch is absent.
    """
    pytest.importorskip("torch")
    code = (
        "import sys; from utrace.utils.pytorch import helpers; "
        "assert 'matplotlib' not in sys.modules, sorted(m for m in sys.modules if m.startswith('matplotlib'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr