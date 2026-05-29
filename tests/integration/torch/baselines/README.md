# Test Baselines

The `.npy` files are reference outputs for the golden tests in `test_golden_mnist.py` and `test_golden_mnist_new_api.py` respectively.
They are committed to the repository and treated as part of the test specification.

## Regenerating

Only regenerate when you have made an intentional change to the algorithm or test setup. Otherwise, a test failure indicates a regression to fix.

    uv run --extra=cpu python tests/integration/torch/regenerate_baselines.py --api new

Commit the new baselines together with the code change that motivated them, and document the reason in the commit message.