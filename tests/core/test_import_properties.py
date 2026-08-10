"""Properties of the package as an importable artifact, not of the API surface.

These assert what `import utrace` does to the process: which global JAX configuration it leaves in place, and which heavy dependencies it does NOT pull in. They are grouped because they share that subject, not their mechanism - the torch test runs in a subprocess because pytest's integration suite imports torch into this one.
"""

import subprocess, sys

def test_core_does_not_import_torch():
    code = (
        "import sys; import utrace; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_x64_is_enabled():
    import utrace  # noqa: F401
    import jax.numpy as jnp
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64