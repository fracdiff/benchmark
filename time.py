import time
from itertools import product

import fracdiff
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.fracdiff_official import fracDiff_FFD


def generate_dataframe(
    n_samples: int, n_features: int, seed: int = None
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    return pd.DataFrame(np.random.randn(n_samples, n_features).cumsum(0))


def timeit(f, n_iter=1, **kwargs) -> tuple:
    times = []
    for _ in range(n_iter):
        t = time.time()
        f(**kwargs)
        times.append(time.time() - t)
    return np.mean(times), np.std(times)


def time_fracdiff(
    method: str,
    n_samples: int,
    n_features: int,
    n: float = 0.5,
    window: int = 10,
    n_iter: int = 10,
    seed: int = 42,
) -> tuple:
    a = generate_dataframe(n_samples=n_samples, n_features=n_features, seed=seed)
    if method == "fracdiff":
        f = fracdiff.fdiff
    if method == "official":
        f = fracDiff_FFD

    return timeit(f, a=a, n_iter=n_iter, n=n, window=window)


if __name__ == "__main__":
    results = []

    params_methods = ["fracdiff", "official"]

    # params_n_samples = [100, 1000, 10000, 100000]
    params_n_samples = [100, 1000]
    params = product(params_n_samples, params_methods)
    for n_samples, method in tqdm(list(params)):
        n_features = 1
        time_mean, time_std = time_fracdiff(method, n_samples, n_features)
        results.append(
            {
                "method": method,
                "n_samples": n_samples,
                "n_features": n_features,
                "time_mean": time_mean,
                "time_std": time_std,
            }
        )

    # params_n_features = [10, 100, 1000]
    params_n_features = [10, 100]
    params = product(params_n_features, params_methods)
    for n_features, method in tqdm(list(params)):
        n_samples = 1000
        time_mean, time_std = time_fracdiff(method, n_samples, n_features)
        results.append(
            {
                "method": method,
                "n_samples": n_samples,
                "n_features": n_features,
                "time_mean": time_mean,
                "time_std": time_std,
            }
        )

    pd.DataFrame(results).to_csv("out/results.csv")
