"""Data transformation module."""

# Author: Graeme Kelly, Emilio J. Palacios-Garcia
# SPDX-License-Identifier: MIT

import numpy as np


def scale(Xs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Normalise the data by subtracting the mean and dividing by the standard deviation.

    Args:
        Xs (np.ndarray): The data to normalise.
        mu (np.ndarray): The mean of the data.
        sigma (np.ndarray): The standard deviation of the data.

    Returns:
        np.ndarray: The normalised data.
    """
    return (Xs - mu) / sigma


def inv_scale(Xs, mu, sigma):
    """Remove the normalisation by multiplying by the standard deviation and adding the mean.

    Args:
        Xs (np.ndarray): The normalised data.
        mu (np.ndarray): The mean of the data.
        sigma (np.ndarray): The standard deviation of the data.

    Returns:
        np.ndarray: The unscaled data.
    """
    return Xs * sigma + mu


def log1p_cols(X: np.ndarray, x_cols: list, pos_cols: list) -> np.ndarray:
    """Apply log1p transformation to the specified columns of the data.

    Args:
        X (np.ndarray): The data to transform.
        x_cols (list): The list of column names.
        pos_cols (list): The list of positive features to apply the log1p transformation to.

    Returns:
        np.ndarray: The transformed data.
    """
    X = X.copy()
    pos_idx = [x_cols.index(c) for c in pos_cols]
    X[:, pos_idx] = np.log1p(np.clip(X[:, pos_idx], 0, None))
    return X


def inv_log1p_cols(X: np.ndarray, x_cols: list, pos_cols: list) -> np.ndarray:
    """Revert the log1p transformation applied to the specified columns of the data.

    Args:
        X (np.ndarray): The transformed data.
        x_cols (list): The list of column names.
        pos_cols (list): The list of positive features to revert the log1p transformation.

    Returns:
        np.ndarray: The original data.
    """
    X = X.copy()
    pos_idx = [x_cols.index(c) for c in pos_cols]
    X[:, pos_idx] = np.expm1(X[:, pos_idx])
    return X
