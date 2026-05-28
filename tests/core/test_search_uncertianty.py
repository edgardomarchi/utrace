def test_cs_padding_does_not_recompile_on_N_change():
    """After 2.1: _search_uncertainty must NOT recompile when only the number
    of calibration scores (n_cs) changes, holding sample shape fixed."""
    import numpy as np
    import jax.numpy as jnp
    from utrace.uncertaintyQuantifier import _search_uncertainty
    from utrace.scores import lac

    _search_uncertainty.clear_cache()  # arrancar limpio

    K = 10
    rng = np.random.default_rng(0)
    max_N = 2000

    # Misma shape de samples en ambas llamadas:
    y = jnp.asarray(rng.integers(0, K, size=300))
    p = jnp.asarray(rng.random((300, K)))

    for n_cs in (500, 1200):  # distinto número de scores válidos
        cs = jnp.full((max_N,), jnp.inf, dtype=jnp.float64)
        cs = cs.at[:n_cs].set(jnp.sort(jnp.asarray(rng.random(n_cs))))
        _search_uncertainty(y, p, cs, jnp.int32(n_cs), 30, lac)

    # Una sola compilación: la shape no cambió, n_cs es tracer
    assert _search_uncertainty._cache_size() == 1, (
        f"Expected 1 compilation, got {_search_uncertainty._cache_size()}. "
        "cs padding is not shape-stable."
    )