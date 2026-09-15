"""Metric functions Module."""

# Author: Graeme Kelly, Emilio J. Palacios-Garcia
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def w1_per_feature(real: np.ndarray, synth: np.ndarray) -> np.ndarray:
    """Wasserstein 1 distance per feature betweewn two datasets.

    Args:
        real (np.ndarray): Real data array of shape (N, D).
        synth (np.ndarray): Synthetic data array of shape (M, D).

    Returns:
        np.ndarray: Array of Wasserstein distances for each feature of shape (D,).
    """
    return np.array([wasserstein_distance(real[:, d], synth[:, d]) for d in range(real.shape[1])])


def sliced_wasserstein(
    real: np.ndarray, synth: np.ndarray, n_proj: int = 200, seed: int = 0
) -> float:
    """Sliced Wasserstein distance between two datasets.

    Args:
        real (np.ndarray): Real data array of shape (N, D).
        synth (np.ndarray): Synthetic data array of shape (M, D).
        n_proj (int, optional): Number of random projections. Defaults to 200.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        float: Sliced Wasserstein distance between the two datasets.
    """
    rng = np.random.default_rng(seed)
    D = real.shape[1]
    dirs = rng.normal(size=(n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    vals = []
    for w in dirs:
        vals.append(wasserstein_distance(real @ w, synth @ w))
    return float(np.mean(vals))


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix for a given dataset.

    Args:
        x (np.ndarray): Input data array of shape (N, D).

    Returns:
        np.ndarray: Spearman correlation matrix of shape (D,
    """
    D = x.shape[1]
    R = np.zeros((D, D), dtype=np.float32)
    for i in range(D):
        for j in range(D):
            R[i, j] = spearmanr(x[:, i], x[:, j]).statistic
    return np.nan_to_num(R, nan=0.0)


def corr_mae(real: np.ndarray, synth: np.ndarray) -> float:
    """Spearman correlation matrix mean absolute error (MAE) between two datasets.

    Args:
        real (np.ndarray): Real data array of shape (N, D).
        synth (np.ndarray): Synthetic data array of shape (M, D).

    Returns:
        float: MAE between the Spearman correlation matrices of the two
    """
    Rr = spearman_matrix(real)
    Rs = spearman_matrix(synth)
    return float(np.mean(np.abs(Rr - Rs)))


def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray | None:
    """Autocorrelation function (ACF) for a 1D array.

    Args:
        x (np.ndarray): Input 1D array.
        max_lag (int): Maximum lag for which to compute the ACF.

    Returns:
        np.ndarray | None: ACF values for lags 0 to max_lag,
            or None if the input array is too short.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < max_lag + 2:
        return None
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf


def build_sorted_sequence(X_scaled_block: np.ndarray, x_cols: list) -> np.ndarray:
    """Build a sorted sequence of data based on the week-angle inferred from sine/cosine.

    Args:
        X_scaled_block (np.ndarray): Input data block of shape (N, D).
        x_cols (list): List of column names corresponding to the features in X_scaled_block.

    Returns:
        np.ndarray: Sorted data block based on the week-angle.
    """
    # sort by week-angle inferred from sin/cos
    si = x_cols.index("WeekTime_sin")
    co = x_cols.index("WeekTime_cos")
    ang = np.arctan2(X_scaled_block[:, si], X_scaled_block[:, co])  # [-pi, pi]
    order = np.argsort(ang)
    return X_scaled_block[order]


def mean_acf_over_blocks(
    X: np.ndarray,
    day_arr: np.ndarray,
    managed_arr: np.ndarray,
    x_cols: list,
    feature: str,
    max_lag: int = 24,
) -> np.ndarray | None:
    """Mean (ACF) over blocks of data defined by day and managed status.

    Args:
        X (np.ndarray): Input data array of shape (N, D).
        day_arr (np.ndarray): Array indicating the day of the week for each sample.
        managed_arr (np.ndarray): Array indicating the managed status for each sample.
        x_cols (list): List of column names corresponding to the features in X.
        feature (str): Feature name for which to compute the ACF.
        max_lag (int, optional): Maximum lag for which to compute the ACF. Defaults to 24.

    Returns:
        np.ndarray: Mean ACF values for the specified feature across all blocks,
            or None if no valid blocks were found.
    """
    fi = x_cols.index(feature)
    acfs = []
    for d in range(7):
        for m in [0, 1]:
            mask = (day_arr == d) & (managed_arr == m)
            block = X[mask]
            if block.shape[0] < max_lag + 2:
                continue
            block = build_sorted_sequence(block, x_cols)
            a = acf_1d(block[:, fi], max_lag=max_lag)
            if a is not None:
                acfs.append(a)
    if not acfs:
        return None
    return np.mean(np.stack(acfs, axis=0), axis=0)


def acf_mae(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    C_real: np.ndarray,
    C_synth: np.ndarray,
    x_cols: list,
    feature: str,
    max_lag: int = 24,
) -> float | None:
    """Calculate MAE) between ACFs of a specified feature in real and synthetic datasets.

    Args:
        X_real (np.ndarray): Real data array of shape (N, D).
        X_synth (np.ndarray): Synthetic data array of shape (M, D).
        C_real (np.ndarray): Condition array for real data of shape (N, 8).
        C_synth (np.ndarray): Condition array for synthetic data of shape (M, 8).
        x_cols (list): List of column names corresponding to the features in X.
        feature (str): Feature name for which to compute the ACF.
        max_lag (int, optional): Maximum lag for which to compute the ACF. Defaults to 24.

    Returns:
        float | None: MAE between the mean ACFs of the specified feature in real
            and synthetic datasets, or None if no valid blocks were found in either dataset.
    """
    day_real = np.argmax(C_real[:, :7], axis=1).astype(int)
    man_real = C_real[:, 7].round().astype(int)
    ar = mean_acf_over_blocks(X_real, day_real, man_real, x_cols, feature, max_lag=max_lag)

    day_s = np.argmax(C_synth[:, :7], axis=1).astype(int)
    man_s = C_synth[:, 7].round().astype(int)
    as_ = mean_acf_over_blocks(X_synth, day_s, man_s, x_cols, feature, max_lag=max_lag)

    if ar is None or as_ is None:
        return None
    return float(np.mean(np.abs(ar - as_)))


# -------------------------
# 4) TSTR utility (regression)
#    Default: predict ConsumedkWh from the other features + condition
# -------------------------
def tstr_regression(
    X_real_train: np.ndarray,
    C_real_train: np.ndarray,
    X_real_test: np.ndarray,
    C_real_test: np.ndarray,
    X_synth: np.ndarray,
    C_synth: np.ndarray,
    x_cols: list,
    feature: str,
    seed: int = 0,
) -> tuple[dict, RandomForestRegressor, RandomForestRegressor]:
    """Perform TSTR and TRTR regression using Random Forests and compute MAE for both.

    Args:
        X_real_train (np.ndarray): Real training data array of shape (N_train, D).
        C_real_train (np.ndarray): Condition array for real training data of shape (N_train, 8).
        X_real_test (np.ndarray): Real test data array of shape (N_test, D).
        C_real_test (np.ndarray): Condition array for real test data of shape (N_test, 8).
        X_synth (np.ndarray): Synthetic data array of shape (M, D).
        C_synth (np.ndarray): Condition array for synthetic data of shape (M, 8).
        x_cols (list): List of column names corresponding to the features in X.
        feature (str): Feature name to predict.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        tuple[dict, RandomForestRegressor, RandomForestRegressor]: A tuple containing:
            - A dictionary with MAE for TSTR and TRTR, and their ratio.
            - The trained Random Forest model on real data (TRTR).
            - The trained Random Forest model on synthetic data (TSTR).
    """
    tgt_i = x_cols.index(feature)

    # Real train data
    Xr_tr = np.concatenate([np.delete(X_real_train, tgt_i, axis=1), C_real_train], axis=1)
    yr_tr = X_real_train[:, tgt_i]

    # Synthetic train data
    Xs = np.concatenate([np.delete(X_synth, tgt_i, axis=1), C_synth], axis=1)
    ys = X_synth[:, tgt_i]

    # Real test data
    Xr_te = np.concatenate([np.delete(X_real_test, tgt_i, axis=1), C_real_test], axis=1)
    yr_te = X_real_test[:, tgt_i]

    m_syn = RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)
    m_syn.fit(Xs, ys)
    pred_tstr = m_syn.predict(Xr_te)
    mae_tstr = mean_absolute_error(yr_te, pred_tstr)

    m_real = RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)
    m_real.fit(Xr_tr, yr_tr)
    pred_trtr = m_real.predict(Xr_te)
    mae_trtr = mean_absolute_error(yr_te, pred_trtr)

    return (
        {
            "MAE_TSTR_scaled": float(mae_tstr),
            "MAE_TRTR_scaled": float(mae_trtr),
            "ratio": float(mae_tstr / (mae_trtr + 1e-12)),
        },
        m_real,
        m_syn,
    )


def tstr_prediction(
    m_real: RandomForestRegressor,
    m_syn: RandomForestRegressor,
    X_real_test: np.ndarray,
    C_real_test: np.ndarray,
    x_cols: list,
    feature: str,
) -> dict:
    """Predict using trained Random Forest models and compute MAE for both TSTR and TRTR.

    Args:
        m_real (RandomForestRegressor): Trained Random Forest model on real data (TRTR).
        m_syn (RandomForestRegressor): Trained Random Forest model on synthetic data (TSTR).
        X_real_test (np.ndarray): Real test data array of shape (N_test, D).
        C_real_test (np.ndarray): Condition array for real test data of shape (N_test, 8).
        x_cols (list): List of column names corresponding to the features in X.
        feature (str): Feature name to predict.

    Returns:
        dict: A dictionary with MAE for TSTR and TRTR, and their ratio.
    """
    tgt_i = x_cols.index(feature)

    # Real test data
    Xr_te = np.concatenate([np.delete(X_real_test, tgt_i, axis=1), C_real_test], axis=1)
    yr_te = X_real_test[:, tgt_i]
    pred_tstr = m_syn.predict(Xr_te)
    mae_tstr = mean_absolute_error(yr_te, pred_tstr)

    pred_trtr = m_real.predict(Xr_te)
    mae_trtr = mean_absolute_error(yr_te, pred_trtr)
    return {
        "MAE_TSTR_scaled": float(mae_tstr),
        "MAE_TRTR_scaled": float(mae_trtr),
        "ratio": float(mae_tstr / (mae_trtr + 1e-12)),
    }
