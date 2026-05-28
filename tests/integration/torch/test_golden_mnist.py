from pathlib import Path

import numpy as np
import pytest
import time

import torch
import random
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


from utrace import UncertaintyQuantifier
from utrace.utils import flatten_batch, get_coverage
from utrace.utils.pytorch.example_models import ImageClassifierCNN
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

BASELINE_DIR = Path(__file__).parent / 'baselines'

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Detect and prevent non-deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _compute_golden_run(seed=42):
    """
    Golden snapshot test for UncertaintyQuantifier refactor safety.
    
    NOTE: Uses an UNTRAINED model with fixed seed. The numeric outputs
    are not meaningful as ML metrics — they are arbitrary reproducible
    numbers used to detect regressions in the conformal prediction logic.
    
    To regenerate baselines: python tests/regenerate_baselines.py
    """

    set_all_seeds(seed)

    g_split  = torch.Generator().manual_seed(seed + 1)
    g_cal    = torch.Generator().manual_seed(seed + 2)
    g_tune   = torch.Generator().manual_seed(seed + 3)
    g_test   = torch.Generator().manual_seed(seed + 4)

    device = torch.device("cpu")
    model = ImageClassifierCNN().to(device)
    model.eval()

    classes = np.arange(10)
    classifier = Pytorch_wrapper(model, classes=classes, device=device)

    uqs = []
    for C in classifier.classes_:
        # Small N for fast test
        uqs.append(UncertaintyQuantifier(model=classifier, N=2000, classes=[C]))

    # Reducir noises a 2-3 valores e iterations a 1 (para test rápido)
    noises = np.array([0.0, 0.5])
    iterations = 1
    
    BATCH_SIZE = 500
    splits = [0.2, 0.2, 0.6]

    iter_coverages_ = []
    iter_alphas_ = []
    iter_uncertainties_ = []

    # Correr la lógica completa, devolver:
    for iteration in range(iterations):
        coverages: list[list[float]] = [[] for _ in classifier.classes_]
        alphas_tuned: list[list[float]] = [[] for _ in classifier.classes_]
        Us_tuned: list[list[float]] = [[] for _ in classifier.classes_]

        for noise in noises:
            torch.manual_seed(seed + int(noise * 1000))
            
            for C in classifier.classes_:
                uqs[C].reset()

            wholeDataset = datasets.MNIST(root='./data', train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              AddGaussianNoise(0., noise)
                                          ]))
            # To make tests fast, we use a small subset of the dataset
            subset_indices = list(range(1000))
            wholeDataset = Subset(wholeDataset, subset_indices)
            
            cal_dataset, tune_dataset, test_dataset = random_split(wholeDataset, splits, generator=g_split)

            calDataLoader = DataLoader(cal_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g_cal)
            tuneDataLoader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g_tune)
            testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g_test)


            for C in classifier.classes_:
                for X_cal, y_cal in calDataLoader:
                    uqs[C].calibrate(X_cal, y_cal, batched=True)

                alphas_ = []
                U_ = []
                for X_tune, y_tune in tuneDataLoader:
                    U, alpha = uqs[C].get_uncertainty_jit(X_tune, y_tune)
                    alphas_.append(alpha)
                    U_.append(U)
                
                alpha = np.nanmean(alphas_)
                U = np.nanmean(U_)

                uqs[C].alpha = alpha
                batch_coverages = []
                for X_n, y_n in testDataLoader:
                    y_p, y_s = uqs[C].predict(X_n, force_non_empty_sets=False)
                    
                    y_n = flatten_batch(y_n).ravel()
                    valid_indexes = np.isin(y_n, np.array([C]))
                    y_n = y_n[valid_indexes]
                    y_s = y_s[valid_indexes]

                    if len(y_n) > 0:
                        i_cov = get_coverage(y_n.numpy(), y_s)
                        batch_coverages.append(i_cov)

                coverage = np.mean(batch_coverages) if batch_coverages else 0.0

                coverages[C].append(coverage)
                alphas_tuned[C].append(alpha)
                Us_tuned[C].append(U)

        iter_coverages_.append(coverages)
        iter_alphas_.append(alphas_tuned)
        iter_uncertainties_.append(Us_tuned)

    mean_coverages = np.mean(iter_coverages_, axis=0)
    mean_alphas = np.mean(iter_alphas_, axis=0)
    mean_uncertainties = np.mean(iter_uncertainties_, axis=0)

    return {
        'coverages':     mean_coverages,
        'uncertainties': mean_uncertainties,
        'alphas':        mean_alphas,
    }

@pytest.fixture(scope="module")
def golden_run():
    from utrace.uncertaintyQuantifier import _search_uncertainty
    _search_uncertainty.clear_cache()
    return _compute_golden_run()


def test_coverages_match_baseline(golden_run):
    expected = np.load(BASELINE_DIR / 'mean_coverages.npy')
    np.testing.assert_allclose(
        golden_run['coverages'], expected, rtol=1e-5, atol=1e-7,
        err_msg="Coverages drifted vs baseline. If intentional, regenerate baselines."
    )

def test_alphas_match_baseline(golden_run):
    expected = np.load(BASELINE_DIR / 'mean_alphas.npy')
    np.testing.assert_allclose(golden_run['alphas'], expected, rtol=1e-5, atol=1e-7)

def test_uncertainties_match_baseline(golden_run):
    expected = np.load(BASELINE_DIR / 'mean_uncertainties.npy')
    np.testing.assert_allclose(golden_run['uncertainties'], expected, rtol=1e-5, atol=1e-7)

def test_jax_cache_does_not_grow(golden_run):
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
    _compute_golden_run()
    elapsed = time.perf_counter() - start
    # Arbitrary threshold based on initial tests
    assert elapsed < 120, f"Golden run took {elapsed:.1f}s, suspect regression"