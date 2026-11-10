"""`import utrace` must not load scipy: it is imported only when a bound is computed."""
import subprocess
import sys


def test_import_utrace_does_not_import_scipy():
    code = (
        "import sys; import utrace; "
        "assert 'scipy' not in sys.modules, sorted(m for m in sys.modules if m.startswith('scipy'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: PLW1510 - check=True would raise before this assert could surface r.stderr
    assert r.returncode == 0, r.stderr


def test_delta_none_does_not_import_scipy_but_delta_does():
    code = (
        "import sys, numpy as np; from utrace import UncertaintyQuantifier; "
        "r = np.random.default_rng(0); y = r.integers(0, 4, 200); s = r.dirichlet(np.ones(4), 200); "
        "uq = UncertaintyQuantifier(N=200); uq.calibrate(s, y); "
        "uq.uncertainty_at(s, y, 0.2); "
        "assert 'scipy' not in sys.modules, 'imported with delta=None'; "
        "uq.uncertainty_at(s, y, 0.2, delta=0.05); "
        "assert 'scipy.special' in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: PLW1510 - check=True would raise before this assert could surface r.stderr
    assert r.returncode == 0, r.stderr
