from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, rankdata
from .core import UncertainFunction


def correlate(
    uvs: Sequence[UncertainFunction], correlation_matrix: ArrayLike
) -> list[UncertainFunction]:
    """
    Force a correlation matrix on a set of statistically distributed objects.
    This function returns new `UncertainFunction` objects with correlated data.

    Parameters
    ----------
    uvs : list or tuple of UncertainFunction
        A sequence of UncertainFunction objects.
    correlation_matrix : array_like
        The correlation matrix to be imposed (e.g., 2D NumPy array or list of lists).

    Returns
    -------
    list of UncertainFunction
        A list of new UncertainFunction objects with the imposed correlation.
        The order corresponds to the input `uvs`.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., wrong types, incompatible shapes,
        invalid correlation matrix).
    """
    # Handle empty input list
    if not uvs:
        return []

    # Validate uvs
    if not all(isinstance(uv, UncertainFunction) for uv in uvs):
        raise ValueError('All inputs in "uvs" must be of type "UncertainFunction".')

    first_uv_npts = uvs[0].shape[0]
    if not all(uv.shape == (first_uv_npts,) for uv in uvs):  # More specific shape check
        raise ValueError(
            "All UncertainFunction inputs must be 1D and have the same number of samples (length)."
        )
    if first_uv_npts == 0:
        raise ValueError("Input UncertainFunction objects cannot be empty (0 samples).")

    # Validate and prepare correlation_matrix
    try:
        correlation_matrix_arr = np.asarray(correlation_matrix, dtype=float)
    except Exception as e:
        raise ValueError(f"Could not convert correlation_matrix to a NumPy array: {e}")

    if correlation_matrix_arr.ndim != 2:
        raise ValueError("Correlation matrix must be 2-dimensional.")
    if correlation_matrix_arr.shape[0] != correlation_matrix_arr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if correlation_matrix_arr.shape[0] != len(uvs):
        raise ValueError(
            "Correlation matrix dimensions must match the number of uncertain variables."
        )

    # Prepare data for induce_correlations: (n_samples, n_variables)
    data_for_induce = np.vstack(uvs).T

    # Apply the correlation (assuming induce_correlations is defined/imported)
    # induce_correlations should handle potential errors from Cholesky etc.
    correlated_sample_data = induce_correlations(
        data_for_induce, correlation_matrix_arr
    )

    # Create new UncertainFunction objects from the correlated sample data
    correlated_distributions = []
    for i, original_uv in enumerate(uvs):
        new_tag = original_uv.tag
        if new_tag is not None:
            new_tag = f"{original_uv.tag}_corr"
        correlated_distributions.append(
            UncertainFunction(correlated_sample_data[:, i].copy(), tag=new_tag)
        )

    return correlated_distributions


def induce_correlations(data: NDArray, correlation_matrix: NDArray) -> NDArray:
    """
    Induce a set of correlations on a column-wise dataset, preserving marginals.

    Parameters
    ----------
    data : 2d-array (m_samples, n_variables)
        Input data. Each column is a variable.
    correlation_matrix : 2d-array (n_variables, n_variables)
        Target correlation matrix. Must be symmetric and positive-definite.

    Returns
    -------
    correlated_output_data : 2d-array
        Data with original marginals but new correlation structure.
    """
    num_samples, num_variables = data.shape

    # 1. Convert each column to ranks (1 to num_samples)
    # data is (m_samples, n_variables), we want to rank along axis 0 (down the columns)
    # 'average' is default, good for ties
    data_rank = rankdata(data, axis=0, method="average")

    # 2. Convert ranks to pseudo-uniform scores [0,1] interval
    # These approximate the empirical CDF values.
    # Using (rank - 0.5) / N for midpoint probability, common in stats
    uniform_scores = (data_rank - 0.5) / num_samples
    # Alternative: uniform_scores = data_rank / (num_samples + 1.0) # As in original mcerp

    # 3. Transform uniform scores to standard normal space using inverse normal CDF
    # normal_transformed_data = norm.ppf(uniform_scores)
    # Handle potential infs from ppf(0) or ppf(1) if ranks are exactly 0 or 1
    # This depends on the uniform_scores formula. For (rank-0.5)/N, ppf(0) and ppf(1) are less likely
    # unless N is very small. For rank/(N+1), they are also avoided.
    # If ranks can be 0 (not standard for rankdata), or if N is small, might need clipping/handling.
    # A robust way: clip scores slightly before ppf to avoid exact 0 or 1.
    epsilon = 1e-10  # Small epsilon
    clipped_uniform_scores = np.clip(uniform_scores, epsilon, 1 - epsilon)
    normal_transformed_data = norm.ppf(clipped_uniform_scores)

    # 4. Compute Cholesky decomposition of the target correlation matrix
    try:
        target_chol = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError as e:
        # Add context to the error
        raise ValueError(
            "Target correlation matrix may not be positive definite. "
            f"Cholesky decomposition failed: {e}"
        )

    # 5. Compute current correlation matrix of the normal-transformed data
    # Ensure there's variance before computing correlations
    std_devs = np.std(normal_transformed_data, axis=0)
    if np.any(std_devs < 1e-9):  # Check if any column is effectively constant
        problematic_vars = np.where(std_devs < 1e-9)[0]
        # Warning or error: correlation is ill-defined for constant variables
        # For now, we'll proceed, but np.corrcoef might produce NaNs or errors.
        print(
            f"Warning: Variables at column indices {problematic_vars} are nearly constant after normal transformation. Correlation results may be unstable."
        )

    current_norm_corr = np.corrcoef(normal_transformed_data, rowvar=False)

    # Check for NaNs in current_norm_corr (can happen if a column was constant)
    if np.any(np.isnan(current_norm_corr)):
        # Attempt to fix NaNs if they are due to perfect correlation (1) or no variance (0)
        # For identity parts where variance was zero, correlation should be 0 with others, 1 with self (or undefined)
        # This is a complex recovery. For now, raise an error as it indicates issues.
        raise ValueError(
            "NaNs found in the current correlation matrix of normal-transformed data. "
            "This often happens if one or more input variables (columns in 'data') "
            "are constant or become constant after rank transformation."
        )

    # 6. Compute Cholesky decomposition of the current normal correlation matrix
    try:
        # Add small jitter for numerical stability if it's very close to singular but theoretically PD.
        current_norm_chol = np.linalg.cholesky(
            current_norm_corr + np.eye(num_variables) * 1e-9
        )
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Current correlation matrix of normal-transformed data may not be positive definite. "
            f"Cholesky decomposition failed: {e}. This can happen if transformed data is rank-deficient."
        )

    # 7. Compute the re-correlation transformation matrix
    # s.T = inv(q).T @ p.T  =>  s = p @ inv(q)
    recorrelation_transform = target_chol @ np.linalg.inv(current_norm_chol)

    # 8. Apply transformation to get normal data with target correlation
    # X_new_normal = X_normal @ s.T
    correlated_normal_data = normal_transformed_data @ recorrelation_transform.T

    # 9. Get ranks of this new correlated normal data
    correlated_normal_ranks = rankdata(correlated_normal_data, axis=0, method="average")

    # 10. Transfer correlation structure back to original marginal distributions
    correlated_output_data = np.empty_like(data)
    for i in range(num_variables):
        # Sort original data values for the current variable
        sorted_original_values_for_var_i = np.sort(data[:, i])

        # Use ranks (1 to num_samples) from correlated_normal_data to pick from sorted_original_values
        # Ranks need to be converted to 0-indexed array indices
        order_indices = correlated_normal_ranks[:, i].astype(int) - 1
        correlated_output_data[:, i] = sorted_original_values_for_var_i[order_indices]

    return correlated_output_data
