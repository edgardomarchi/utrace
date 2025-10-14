import numpy as np


def lac_cal(
    y: np.ndarray,
    smx: np.ndarray,
    ) -> np.ndarray:
    return 1 - smx[np.arange(len(y)), y].cpu().numpy().astype(np.float64) # changed .astype to .cpu().numpy().astype

def lac(smx:np.ndarray) -> np.ndarray:
    """LAC score.
    Args:
        smx (np.array): model output of the softmax function
    Returns:
        np.array: LAC score
    """
    return 1 - smx.astype(np.float64)

def aps_cal(
    y: np.ndarray,
    smx: np.ndarray,
    ) -> np.ndarray:
    print(f'softmax of sample 0: {smx[0]}')
    sorted_proba_idx = smx.argsort(axis=1)[:, ::-1]    # Sort the probabilities in descending order
    print(f'sorted_proba_idx of sample 0: {sorted_proba_idx[0]}')
    accumulated_probas = np.take_along_axis(smx.astype(np.float64), sorted_proba_idx, axis=1).cumsum(axis=1)    # Cumulative sum of the sorted probabilities
    print(f'accumulated_probas of sample 0: {accumulated_probas[0]}')
    cal_scores = np.take_along_axis(accumulated_probas, sorted_proba_idx.argsort(axis=1), axis=1)[range(y.shape[0]), y]  # Get the cumulative sum of the sorted probabilities for the true class
    print(f'cal_scores of sample 0: {cal_scores[0]}')
    return cal_scores

def aps(smx:np.ndarray) -> np.ndarray:
    print(f'softmax of sample 0: {smx[0]}')
    sorted_proba_idx = smx.argsort(axis=1)[:, ::-1]
    print(f'sorted_proba_idx of sample 0: {sorted_proba_idx[0]}')
    accumulated_probas = np.take_along_axis(smx.astype(np.float64), sorted_proba_idx, axis=1).cumsum(axis=1)
    print(f'accumulated_probas of sample 0: {accumulated_probas[0]}')
    scores = np.take_along_axis(accumulated_probas, sorted_proba_idx.argsort(axis=1), axis=1) # Get the cumulative sum of the sorted probabilities for all classes
    print(f'scores of sample 0: {scores[0]}')
    return scores
