import numpy as np
import pandas as pd


def getWeights(d: int, size: int) -> np.ndarray:
    """
    Return array of weights

    Returns
    -------
    weithts : np.array, shape (size, 1)

    Examples
    --------
    >>> getWeights(0.5, 5)
    array([[-0.0390625],
           [-0.0625   ],
           [-0.125    ],
           [-0.5      ],
           [ 1.       ]])
    """
    # thres>0 drops insignificant weights
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff(series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
    """

    Parameters
    ----------
    series : pandas.Dataframe, shape (n_samples, n_features)
        dataframe to differentiate.

    Returns
    -------
    diff : pandas.Dataframe, shape (n_samples, n_features)

    Examples
    --------
    # >>> np.random.seed(42)
    # >>> dataframe = pd.DataFrame(np.random.randn(10, 2))
    # >>> fracDiff(dataframe, 0.5, 0.1)

    ---

    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series w=getWeights(d,series.shape[0])
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold w_=np.cumsum(abs(w))
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        # --- Edited by simaki ---
        # seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        seriesF = series[[name]].fillna(method="ffill").dropna()
        df_ = pd.Series(np.empty_like(range(skip, seriesF.shape[0])), dtype="float64")
        # ---
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
            # --- Edited by simaki ---
            df_[loc] = np.dot(w[-(iloc + 1) :, :].T, seriesF.loc[:loc])[0, 0]
            # ---
        # --- Edited by simaki ---
        df[name] = df_.copy(deep=True)
        # ---
    df = pd.concat(df, axis=1)
    return df


# edited by simaki; thres -> size, series -> a
# def fracDiff_FFD(series, d, thres=1e-5):
def fracDiff_FFD(a: pd.DataFrame, n, window=10) -> pd.DataFrame:
    """
    >>> np.random.seed(42)
    >>> dataframe = pd.DataFrame(np.random.randn(10, 2))
    >>> fracDiff_FFD(dataframe, 0.5)

    ---

    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. ’’’
    """
    # 1) Compute weights for the longest series
    # --- edited by simaki ---
    # w = getWeights_FFD(d, thres)
    w = getWeights(n, window)
    # ---
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}
    for name in a.columns:
        # --- edited by simaki ---
        # seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        seriesF = a[[name]].fillna(method="ffill").dropna()
        df_ = pd.Series(np.empty_like(range(width, seriesF.shape[0])), dtype="float64")
        # ---
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(a.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df
