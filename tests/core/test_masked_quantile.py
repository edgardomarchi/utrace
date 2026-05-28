import numpy as np
import jax.numpy as jnp
import pytest
from utrace.utils import _masked_quantile_higher


@pytest.mark.parametrize("n_valid", [1, 2, 5, 17, 100])
@pytest.mark.parametrize("pad", [0, 50])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 0.999, 1.0])
def test_masked_quantile_matches_jnp(n_valid, pad, q):
    rng = np.random.default_rng(0)
    valid = np.sort(rng.standard_normal(n_valid))
    padded = np.full(n_valid + pad, np.inf)
    padded[:n_valid] = valid

    expected = jnp.quantile(jnp.asarray(valid, dtype=jnp.float64),
                            q, method='higher')
    got = _masked_quantile_higher(jnp.asarray(padded, dtype=jnp.float64),
                                  jnp.int32(n_valid),
                                  jnp.asarray(q, dtype=jnp.float64))
    np.testing.assert_allclose(float(got), float(expected),
                               rtol=1e-12, atol=1e-12,
                               err_msg=f"n={n_valid} pad={pad} q={q}")