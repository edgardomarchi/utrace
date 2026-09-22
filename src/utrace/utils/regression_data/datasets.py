import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
#from ucimlrepo import fetch_ucirepo


def simulate_gaussian(n, sigma = 0.5, random_state = None):

    rng = np.random.default_rng(random_state)
    X = rng.uniform(low = -3.0, high = 3.0, size = (n, 1))
    y = np.sin(X[:, 0]) + 0.5*X[:, 0]**2 - 0.1*X[:, 0]
    noise = sigma*rng.standard_normal(n)

    y = y + noise
    return X, y 

def simulate_mixture(n, sigma = (0.5, 0.2), mix_prob = 0.5,
                     random_state = None):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(low = -3.0, high = 3.0, size = (n, 1))
    y = np.sin(X[:, 0]) + 0.5*X[:, 0]**2 - 0.1*X[:, 0]
    p = rng.random(n) < mix_prob
    noise = np.where(p, sigma[0]*rng.standard_normal(n),
                     sigma[1]*rng.standard_normal(n))
    y = y + noise
    return X, y

def simulate_heteroskedastic(n, random_state = None):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(low = -3.0, high = 3.0, size = (n, 1))
    y = np.sin(X[:, 0]) + 0.5*X[:, 0]**2 - 0.1*X[:, 0]
    noise_std = 0.2 + 0.3*np.abs(X[:, 0])
    noise = noise_std*rng.standard_normal(n)
    y = y + noise
    return X, y 
'''
def load_uci_dataset(dataset_name):
    """
    Fetches and preprocess standard regression benchmarks
    """

    if dataset_name == "concrete":
        # Concrete Compressive Strength (ID = 165)
        data = fetch_ucirepo(id = 165)
        X = data.data.features.values
        y = data.data.targets.values.flatten()

    elif dataset_name == "housing":
        # California Hosing
        data = fetch_california_housing()
        X = data.data
        y = data.target

    elif dataset_name == "wine":
        # Wine Quality - White and Red combines (ID = 186)
        data = fetch_ucirepo(id = 186)
        X = data.data.features.values
        y = data.data.targets.values.flatten()

    elif dataset_name == "bike":
        # Bike sharing demans (ID = 275)
        data = fetch_ucirepo(id = 275)
        x_df = data.data.features.copy()
        if 'dteday' in x_df.columns:
            x_df = x_df.drop(columns = ['dteday'])

        X = x_df.values
        y = data.data.targets.values.flatten()

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Clean NANs is any exist in the raw fetch
    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis = 1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Standarize features
    X = StandardScaler().fit_transform(X)
    return X, y 
'''
