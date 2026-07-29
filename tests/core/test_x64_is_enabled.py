def test_x64_is_enabled():
    import utrace  # noqa: F401
    import jax.numpy as jnp
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64