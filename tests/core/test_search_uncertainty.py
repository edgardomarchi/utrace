import numpy as np
import jax.numpy as jnp
from utrace.uncertaintyQuantifier import _search_uncertainty
from utrace.scores import lac

def test_cs_padding_does_not_recompile_on_N_change():
    """After Phase 2.1: _search_uncertainty must NOT recompile when only n_cs
    changes, holding sample-related shapes (y, p, valid_mask) fixed."""
    import jax.numpy as jnp
    import numpy as np
    from utrace.uncertaintyQuantifier import _search_uncertainty
    from utrace.scores import lac

    _search_uncertainty.clear_cache()

    K = 10
    rng = np.random.default_rng(0)
    max_N = 2000
    B = 300

    # Sample shapes fijas en ambas llamadas:
    y    = jnp.asarray(rng.integers(0, K, size=B).astype(np.int32))
    p    = jnp.asarray(rng.random((B, K)))
    mask = jnp.ones(B, dtype=bool)   # NEW: mask with fixed shape (B,)

    for n_cs in (500, 1200):
        cs = jnp.full((max_N,), jnp.inf, dtype=jnp.float64)
        cs = cs.at[:n_cs].set(jnp.sort(jnp.asarray(rng.random(n_cs))))
        _search_uncertainty(y, p, mask, cs, jnp.int32(n_cs), 30, lac)
        #                       ^^^^ posición del nuevo arg

    assert _search_uncertainty._cache_size() == 1, (
        f"Expected 1 compilation, got {_search_uncertainty._cache_size()}. "
        "cs padding is not shape-stable."
    )

def _reference_filtered(y, p, valid, cs_padded, n_cs, max_iters, score_fn):
    """Versión 'filtra antes' (equivalente a 2.1) para comparar."""
    y_f = jnp.asarray(y[valid])
    p_f = jnp.asarray(p[valid])
    mask_f = jnp.ones(y_f.shape[0], dtype=bool)  # todos válidos tras filtrar
    return _search_uncertainty(y_f, p_f, mask_f, cs_padded, n_cs, max_iters, score_fn)


def test_masking_equals_filtering():
    rng = np.random.default_rng(0)
    K, B, max_N = 10, 500, 2000
    C = 3  # clase de interés

    y = rng.integers(0, K, size=B).astype(np.int32)
    p = rng.random((B, K)); p /= p.sum(axis=1, keepdims=True)
    valid = np.isin(y, [C])

    n_cs = 800
    cs = jnp.full((max_N,), jnp.inf, dtype=jnp.float64)
    cs = cs.at[:n_cs].set(jnp.sort(jnp.asarray(rng.random(n_cs))))
    n_cs_j = jnp.int32(n_cs)

    # 1. versión filtrada (referencia)
    a_ref, U_ref = _reference_filtered(y, p, valid, cs, n_cs_j, 30, lac)

    # 2. versión paddeada+enmascarada (la de 2.2)
    B_max = 512
    y_pad = np.zeros(B_max, dtype=np.int32)
    p_pad = np.zeros((B_max, K), dtype=np.float64)
    m_pad = np.zeros(B_max, dtype=bool)
    y_pad[:B] = y; p_pad[:B] = p; m_pad[:B] = valid
    y_safe = np.where(m_pad, y_pad, 0)

    a_mask, U_mask = _search_uncertainty(
        jnp.asarray(y_safe), jnp.asarray(p_pad), jnp.asarray(m_pad),
        cs, n_cs_j, 30, lac)

    np.testing.assert_allclose(float(a_mask), float(a_ref), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(U_mask), float(U_ref), rtol=1e-10, atol=1e-12)