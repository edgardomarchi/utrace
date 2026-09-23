"""Regression test for Defect 2 (last-iterate-vs-feasible), on real data known
to trigger it.

The 2026-09-22 diagnostic found that classes 0 and 3 of the golden MNIST
configuration (untrained CNN, N=18/N=26 calibration scores) converge to an
interior alpha where consecutive bisection iterates straddle a quantile grid
jump -- the pre-fix search's returned U depended on `max_iters`' parity there,
by up to ~0.1. `tests/core/test_search_status.py`'s synthetic cases could not
be made to reproduce that specific grid alignment (it depends on structure in
the untrained CNN's near-degenerate output, not just "any converging case"),
so this test reuses the real golden data-generation path instead -- scoped
down to the two classes that actually exhibit it, and to noise=0.0 only, to
stay fast. Verified separately (against the pre-fix code, loaded from git, in a scratch
script -- not part of this suite) that this same data makes the OLD
`_search_uncertainty` max_iters-sensitive.

Does not modify test_golden_mnist.py; imports its helpers.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from test_golden_mnist import _compute_logits_for_loader, set_all_seeds
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from utrace import SearchStatus, SearchStatusWarning, UncertaintyQuantifier
from utrace.uncertaintyQuantifier import _ensure_sorted, _q_hat_from_alpha
from utrace.utils.pytorch.example_models import ImageClassifierCNN
from utrace.utils.pytorch.helpers import flatten_batch
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

# The two classes the diagnostic found convergent-with-grid-alignment at
# noise=0.0, and their calibration set sizes at the time of that finding.
_GRID_SENSITIVE_CLASSES = (0, 3)


@pytest.fixture(scope="module")
def grid_sensitive_uqs():
    """Calibrates UncertaintyQuantifier instances for classes 0 and 3 exactly
    as `_compute_golden_run` does at noise=0.0 (same seeds, same split, same
    untrained CNN) and returns them along with the tuning set. Scoped down
    from the golden test's full 10-class x 2-noise sweep to just what this
    test needs, for speed.
    """
    seed = 42
    set_all_seeds(seed)
    device = torch.device("cpu")
    model = ImageClassifierCNN().to(device)
    model.eval()
    classes = np.arange(10)
    classifier = Pytorch_wrapper(model, classes=classes, device=device)

    whole_dataset = datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            AddGaussianNoise(0., 0.0),
        ]))
    whole_dataset = Subset(whole_dataset, list(range(1000)))

    g_split = torch.Generator().manual_seed(seed + 1)
    g_cal = torch.Generator().manual_seed(seed + 2)
    g_tune = torch.Generator().manual_seed(seed + 3)
    cal_dataset, tune_dataset, _test_dataset = random_split(
        whole_dataset, [0.2, 0.2, 0.6], generator=g_split)

    cal_loader = DataLoader(cal_dataset, batch_size=500, shuffle=True, generator=g_cal)
    tune_loader = DataLoader(tune_dataset, batch_size=500, shuffle=True, generator=g_tune)

    uqs = {
        C: UncertaintyQuantifier(N=2000, classes=[C], max_batch_size=512)
        for C in _GRID_SENSITIVE_CLASSES
    }
    with torch.inference_mode():
        for X_cal, y_cal in cal_loader:
            smx_cal = classifier.predict_proba(X_cal).cpu().numpy()
            y_cal_arr = flatten_batch(y_cal).ravel().numpy().astype(int)
            for C in _GRID_SENSITIVE_CLASSES:
                uqs[C].calibrate(smx_cal, y_cal_arr, batched=True)

    tune_smx, tune_y = _compute_logits_for_loader(tune_loader, classifier)
    return uqs, tune_smx, tune_y


@pytest.mark.parametrize("C", _GRID_SENSITIVE_CLASSES)
def test_q_hat_and_EC_stable_across_max_iters(C, grid_sensitive_uqs):
    """`q_hat` at the returned alpha, and the conditioned mean EC (recovered
    from `U`/`alpha`), must be identical across max_iters in [25, 30] -- the
    exact range the diagnostic showed the pre-fix search flipping on for
    these classes. Alpha itself may differ, but only by the final step width
    (2**-max_iters); U may then differ only through the `(1 - alpha)` factor.
    """
    uqs, tune_smx, tune_y = grid_sensitive_uqs
    uq = uqs[C]
    N = uq._state.N

    q_hats = set()
    ECs = set()
    alphas = {}
    for max_iters in range(25, 31):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SearchStatusWarning)
            U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=max_iters)
        assert uq.search_status_ == SearchStatus.CONVERGED, (
            f"class {C} drifted out of CONVERGED at max_iters={max_iters}; "
            f"this test's premise (a known interior/grid-sensitive case) no "
            f"longer holds for this data"
        )
        alphas[max_iters] = alpha

        uq._state = _ensure_sorted(uq._state)
        q_hat = float(_q_hat_from_alpha(
            uq._state.conformity_scores, jnp.int32(N), alpha
        ))
        q_hats.add(round(q_hat, 12))
        EC = (1.0 - U) / (1.0 - alpha)
        ECs.add(round(EC, 9))

    assert len(q_hats) == 1, f"q_hat varied across max_iters 25..30 for class {C}: {q_hats}"
    assert len(ECs) == 1, f"EC varied across max_iters 25..30 for class {C}: {ECs}"

    # Alpha may differ, but only by the finest step width actually reached.
    max_step = 2.0 ** -25
    alpha_values = list(alphas.values())
    assert max(alpha_values) - min(alpha_values) <= max_step
