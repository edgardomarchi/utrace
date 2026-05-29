"""Golden test using the *_from_proba API.

This test exercises the statistically correct usage of U-TraCE and has its
OWN baselines (tests/baselines/new_api/), distinct from the legacy golden.

Why separate baselines: the legacy golden (test_golden_mnist.py) averaged
per-batch alphas during tuning, which is incorrect — alpha is a nonlinear
function of the tuning data, so mean(alpha_per_batch) != alpha(all_data).
This test computes a single alpha over the entire tuning set, as the method
requires. The numbers therefore differ from the legacy baselines by design.

Equivalence between the legacy and new APIs *given identical inputs* is
proven in test_legacy_equivalence.py (synthetic inputs, no batching).

Memory note: calibration is streamed (one batch of logits at a time, scores
accumulated) so it scales to large calibration sets. Tuning materializes its
set, which is cheap because conformal prediction converges with few samples.
"""
import warnings

import numpy as np
import pytest
import torch
import time

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils import flatten_batch, get_coverage
from utrace.utils.pytorch.example_models import ImageClassifierCNN
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

from _baselines import NEW_API_BASELINE_DIR

def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_logits_for_loader(loader, classifier):
    """Materialize logits + labels for a whole loader. Used for tuning and
    test sets (small/bounded). Logits are N x K floats — much smaller than
    the underlying images. NOT used for calibration, which is streamed.
    """
    probs_chunks, labels_chunks = [], []
    with torch.inference_mode():
        for X, y in loader:
            probs_chunks.append(classifier.predict_proba(X).cpu().numpy())
            labels_chunks.append(flatten_batch(y).ravel().numpy().astype(int))
    return np.concatenate(probs_chunks), np.concatenate(labels_chunks)


def _compute_golden_run_new_api(seed=42):
    set_all_seeds(seed)

    device = torch.device("cpu")
    model = ImageClassifierCNN().to(device)
    model.eval()

    classes = np.arange(10)
    classifier = Pytorch_wrapper(model, classes=classes, device=device)

    uqs = [UncertaintyQuantifier(N=2000, classes=[C], max_batch_size=512)
           for C in classifier.classes_]

    noises = np.array([0.0, 0.5])
    iterations = 1
    BATCH_SIZE = 500
    splits = [0.2, 0.2, 0.6]

    iter_coverages_, iter_alphas_, iter_uncertainties_ = [], [], []

    for iteration in range(iterations):
        coverages    = [[] for _ in classifier.classes_]
        alphas_tuned = [[] for _ in classifier.classes_]
        Us_tuned     = [[] for _ in classifier.classes_]

        for noise in noises:
            for C in classifier.classes_:
                uqs[C].reset()

            wholeDataset = datasets.MNIST(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    AddGaussianNoise(0., noise),
                ]))
            wholeDataset = Subset(wholeDataset, list(range(1000)))

            g_split = torch.Generator().manual_seed(seed + 1)
            g_cal   = torch.Generator().manual_seed(seed + 2)
            g_tune  = torch.Generator().manual_seed(seed + 3)
            g_test  = torch.Generator().manual_seed(seed + 4)

            cal_dataset, tune_dataset, test_dataset = random_split(
                wholeDataset, splits, generator=g_split)

            calDataLoader  = DataLoader(cal_dataset,  batch_size=BATCH_SIZE,
                                        shuffle=True, generator=g_cal)
            tuneDataLoader = DataLoader(tune_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, generator=g_tune)
            testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, generator=g_test)

            # ----- CALIBRATION: streamed (batch-outer, class-inner) -----
            # Honors lazy loading: only one batch of logits in memory at a
            # time. Scores are accumulated via batched=True. Each uq filters
            # its own class internally. Iterating the loader once (rather than
            # once per class) is both more efficient and more memory-friendly
            # than precomputing the whole calibration set.
            for X_cal, y_cal in calDataLoader:
                p_cal = classifier.predict_proba(X_cal).cpu().numpy()
                y_cal_arr = flatten_batch(y_cal).ravel().numpy().astype(int)
                for C in classifier.classes_:
                    uqs[C].calibrate_from_proba(p_cal, y_cal_arr, batched=True)

            # ----- TUNING + TEST: materialize (small / bounded sets) -----
            tune_probs, tune_y = _compute_logits_for_loader(tuneDataLoader, classifier)
            test_probs, test_y = _compute_logits_for_loader(testDataLoader, classifier)

            for C in classifier.classes_:
                # Tuning: ONE call over the entire tuning set. Pure, no mutation.
                # Averaging per-batch alphas would be statistically incorrect.
                U, alpha = uqs[C].get_uncertainty_from_proba(tune_probs, tune_y)

                # Caller sets alpha explicitly (get_uncertainty does not mutate)
                uqs[C].alpha = alpha

                # Test: prediction over the whole set; coverage is a GLOBAL
                # proportion, not an average of per-batch proportions.
                y_p, y_s = uqs[C].predict_from_proba(test_probs)
                valid = np.isin(test_y, np.array([C]))
                if valid.sum() > 0:
                    coverage = get_coverage(test_y[valid], y_s[valid])
                else:
                    coverage = np.float64('nan')

                coverages[C].append(coverage)
                alphas_tuned[C].append(alpha)
                Us_tuned[C].append(U)

        iter_coverages_.append(coverages)
        iter_alphas_.append(alphas_tuned)
        iter_uncertainties_.append(Us_tuned)

    return {
        'coverages':     np.mean(iter_coverages_,    axis=0),
        'alphas':        np.mean(iter_alphas_,       axis=0),
        'uncertainties': np.mean(iter_uncertainties_, axis=0),
    }


@pytest.fixture(scope="module")
def golden_run_new_api():
    """Run the new-API golden, asserting NO DeprecationWarning fires
    (any leftover legacy call would raise here)."""
    from utrace.uncertaintyQuantifier import _search_uncertainty
    _search_uncertainty.clear_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return _compute_golden_run_new_api()


def test_coverages_match_baseline_new_api(golden_run_new_api):
    expected = np.load(NEW_API_BASELINE_DIR / 'mean_coverages.npy')
    np.testing.assert_allclose(golden_run_new_api['coverages'], expected,
                               rtol=1e-5, atol=1e-7)


def test_alphas_match_baseline_new_api(golden_run_new_api):
    expected = np.load(NEW_API_BASELINE_DIR / 'mean_alphas.npy')
    np.testing.assert_allclose(golden_run_new_api['alphas'], expected,
                               rtol=1e-5, atol=1e-7)


def test_uncertainties_match_baseline_new_api(golden_run_new_api):
    expected = np.load(NEW_API_BASELINE_DIR / 'mean_uncertainties.npy')
    np.testing.assert_allclose(golden_run_new_api['uncertainties'], expected,
                               rtol=1e-5, atol=1e-7)
    
def test_jax_cache_does_not_grow(golden_run_new_api):
    """Detect if _search_uncertainty is recompiling on every call."""
    from utrace.uncertaintyQuantifier import _search_uncertainty
    cache_size = _search_uncertainty._cache_size() # type:ignore
    # Allow up to 2: one for warmup, one "stable"
    # After fix, should be exactly 1
    assert cache_size <= 3, (
        f"_search_uncertainty was compiled {cache_size} times. "
        "Likely cause: variable shapes in JIT arguments. "
        "Use padding + mask pattern instead of filtering before JIT."
    )

def test_golden_run_under_threshold():
    """Smoke test: detect catastrophic performance regressions."""
    start = time.perf_counter()
    _compute_golden_run_new_api()
    elapsed = time.perf_counter() - start
    # Arbitrary threshold based on initial tests
    assert elapsed < 120, f"Golden run took {elapsed:.1f}s, suspect regression"