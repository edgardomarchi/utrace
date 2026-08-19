import logging

import numpy as np

logger = logging.getLogger(__name__)

def _bucket_size(B, base=256):
    """Siguiente múltiplo de `base` >= B (bucketing para shape-stability)."""
    return int(np.ceil(B / base) * base)

def relabel(mask):
    """
    Reordering to match labels for AI model and Ground Truth.
    (Borrowed from Maik's code)
    """
    mask[mask == 3] = 10
    mask[mask == 1] = 3
    mask[mask == 10] = 1
    return mask

def check_row_sums(matrix, tol=1e-6) -> bool:
    """
    Helper function to check if each row of matrix sums to 1.

    Args:
        matrix (np.ndarray): Input matrix of size (M, N).
        tol (float): Tolerance to consider the sum as 1 (to handle numerical errors).

    Returns:
        None
    """
    row_sums = matrix.sum(axis=1)
    invalid_rows = np.where(np.abs(row_sums - 1) > tol)[0]

    if len(invalid_rows) > 0:
        logger.debug("The following rows do not sum to 1 (max. 10 rows):\n %s",
                     invalid_rows[:10].tolist())
        return False
    else:
        logger.debug("All rows sum to approximately 1.")
        return True

def plot_scores(
    alpha: float,
    scores: np.ndarray,
    quantiles: np.ndarray,
    method: str,
    ax: "plt.Axes",  # noqa: F821 - quoted forward reference on purpose: matplotlib's import
    # is deferred into this function's body (see below) precisely so `import utrace` stops
    # paying for matplotlib; the annotation can't reference `plt` unquoted at module-evaluation
    # time, since `plt` isn't bound at module scope.
) -> None:
    """
    Plots the distribution of scores and overlays quantile lines.
    (Borrowed from "Introduction to Conformal Prediction with Python" by C. Molnar)

    Parameters:
    alpha (float): The alpha value used.
    scores (np.ndarray): An array of score values to be plotted in the histogram.
    quantiles (np.ndarray): An array of quantile values.
    method (str): The method name.
    ax (plt.Axes): The matplotlib Axes object where the plot will be drawn.

    Returns:
    None
    """
    import matplotlib.pyplot as plt  # noqa: F401 - deferred import kept for its side effect
    # (loading matplotlib on call, not on `import utrace`); see MIGRATION.md "Import structure:
    # matplotlib and pandas deferred out of `import utrace`". Not referenced by name below.
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    n, _, _ = ax.hist(scores, bins="auto")
    for quantile in quantiles:
        ax.vlines(
            x=quantile,
            ymin=0,
            ymax=n.max(),  # type:ignore
            color=colors[1],
            linestyles="dashed",
            label=f"alpha = {alpha}",
        )

    ax.set_title(f"Distribution of scores for '{method}' method")
    ax.legend()
    ax.set_xlabel("scores")
    ax.set_ylabel("count")

def class_wise_performance(y_new, y_set, classes):
    """
    Evaluate the performance of classification for each class.
    (Borrowed from "Introduction to Conformal Prediction with Python" by C. Molnar)

    Parameters:
    y_new (pd.Series or np.ndarray): The true class labels.
    y_set (pd.Series or np.ndarray): The cp-predicted class sets.
    classes (list): List of class names.

    Returns:
    pd.DataFrame: A dataframe containing the coverage and average set size for each class.
    """
    import pandas as pd
    df = pd.DataFrame()
    # Loop through the classes
    for i,C in enumerate(classes):
        # Calculate the coverage and set size for the current class
        ynew = y_new[y_new == C]
        yscore = y_set[y_new == C]
        cov = get_coverage(ynew, yscore)
        size = get_average_set_size(yscore)
        # Create a new dataframe with the calculated values
        temp_df = pd.DataFrame({
            "class": C,
            "coverage": [cov],
            "avg. set size": [size]
            }, index = [i])
        # Concatenate the new dataframe with the existing one
        df = pd.concat([df, temp_df])
    return(df)

def get_coverage(values: np.ndarray, sets: np.ndarray) -> float:
    V = len(values)
    logger.debug("\n------------ V: %d ------------\n", V)
    if V < 1:
        return -1.0
    is_in = sets[np.arange(V), values]
    coverage = is_in.sum() / V
    if coverage > 0.99:
        print(f'get_coverage(): Coverage of: {coverage}!!!, V: {V}')
        print(f'True values: {sets.flatten().sum()}, total values: {len(sets.flatten())}')
    return coverage

def get_average_set_size(sets: np.ndarray) -> float:
    return sets.sum(axis=1).mean()
