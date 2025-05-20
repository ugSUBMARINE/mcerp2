from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def _generate_lhs_unit_hypercube(
    num_samples: int, num_dimensions: int, *, rng: np.random.Generator | None = None
) -> NDArray:
    """
    Generates a Latin Hypercube sample in the unit hypercube [0,1]^num_dimensions.
    Each column's samples are independently permuted.

    Parameters
    ----------
    num_samples : int
        Number of sample points to generate (rows).
    num_dimensions : int
        Number of variables/dimensions (columns).
    rng : np.random.Generator, optional
        NumPy random number generator instance for reproducibility.
        If None, uses np.random.default_rng().

    Returns
    -------
    np.ndarray
        An array of shape (num_samples, num_dimensions) with LHS samples in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize the hypercube
    lhs_samples = np.empty((num_samples, num_dimensions))

    # Generate stratified samples for each dimension (column)
    for j in range(num_dimensions):
        # Create N random points, one in each stratum [i/N, (i+1)/N)
        # stratified_points_in_unit_interval = (rng.uniform(size=num_samples) + np.arange(num_samples)) / num_samples
        # The above is a common way. Original mcerp's logic was:
        # segmentMin = i * segmentSize
        # point = segmentMin + (rng.random() * segmentSize)
        # This is equivalent to:
        stratum_indices = np.arange(num_samples)  # 0, 1, ..., N-1
        random_offsets_in_strata = rng.uniform(low=0.0, high=1.0, size=num_samples)

        # Points in [0,1) ensuring one per stratum, then shuffle
        # Points are (idx + offset)/N
        points_in_unit_interval = (
            stratum_indices + random_offsets_in_strata
        ) / num_samples

        # Shuffle these points for this dimension independently
        lhs_samples[:, j] = rng.permutation(points_in_unit_interval)

    return lhs_samples


def _apply_ppf_to_lhs(
    lhs_unit_samples: NDArray, distributions: list | tuple, single_dist_dims: int = 1
) -> NDArray:
    """
    Applies the Percent Point Function (PPF) of given distributions
    to LHS samples from the unit hypercube.

    Parameters
    ----------
    lhs_unit_samples : np.ndarray
        LHS samples in the unit hypercube [0,1]^D, shape (num_samples, D).
    distributions : list or tuple of scipy.stats.rv_frozen or a single rv_frozen
        Distribution object(s) with a .ppf method.
    single_dist_dims : int
        If 'distributions' is a single rv_frozen object, this specifies how many
        columns of lhs_unit_samples should be transformed using this distribution.
        Ignored if 'distributions' is a list/tuple.

    Returns
    -------
    np.ndarray
        Transformed LHS samples, shape (num_samples, D).
    """
    num_samples, num_dimensions = lhs_unit_samples.shape
    transformed_samples = np.empty_like(lhs_unit_samples)

    if hasattr(distributions, "__getitem__"):  # List or tuple of distributions
        if len(distributions) != num_dimensions:
            raise ValueError(
                "Number of distributions must match number of dimensions in lhs_unit_samples."
            )
        for i in range(num_dimensions):
            dist_obj = distributions[i]
            if not hasattr(dist_obj, "ppf"):
                raise TypeError(f"Distribution at index {i} must have a 'ppf' method.")
            transformed_samples[:, i] = dist_obj.ppf(lhs_unit_samples[:, i])
    else:  # Single distribution object
        dist_obj = distributions
        if not hasattr(dist_obj, "ppf"):
            raise TypeError("Distribution object must have a 'ppf' method.")
        if num_dimensions != single_dist_dims:
            # Check if lhs_unit_samples has expected columns
            raise ValueError(
                f"LHS unit samples have {num_dimensions} dimensions, but single_dist_dims is {single_dist_dims}."
            )
        # num_dimensions here should be single_dist_dims
        for i in range(num_dimensions):
            transformed_samples[:, i] = dist_obj.ppf(lhs_unit_samples[:, i])

    return transformed_samples


def lhd(
    dist,
    size: int,
    dims: int = 1,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """
    Create a Latin Hypercube Sample design.

    Parameters
    ----------
    dist : scipy.stats.rv_frozen or list/tuple of rv_frozen
        A single frozen scipy.stats distribution object, or a sequence of them.
        Each object must have a `ppf` (Percent Point Function) method.
    size : int
        Number of sample points to generate for each variable.
    dims : int, optional
        If `dist` is a single distribution object, `dims` specifies how many
        variables (columns) should be generated using this distribution.
        Defaults to 1. Ignored if `dist` is a sequence.
    rng : np.random.Generator, optional
        NumPy random number generator instance for reproducibility.
        If None, uses np.random.default_rng().

    Returns
    -------
    np.ndarray
        A 2D array of shape (size, num_variables) containing the LHS samples.
        `num_variables` is `len(dist)` if `dist` is a sequence, otherwise it's `dims`.

    Comments: 'form', 'iterations' are currently non-functional as per original, so omitted for now
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("'size' must be a positive integer.")
    if not isinstance(dims, int) or dims <= 0:
        raise ValueError("'dims' must be a positive integer.")

    if hasattr(dist, "__getitem__"):  # Sequence of distributions
        num_vars = len(dist)
        if num_vars == 0:
            raise ValueError("'dist' must have at least one element.")
    else:  # Single distribution
        num_vars = dims

    # 1. Generate LHS samples in the unit hypercube [0,1]^num_vars
    unit_lhs_samples = _generate_lhs_unit_hypercube(
        num_samples=size, num_dimensions=num_vars, rng=rng
    )

    # 2. Transform these unit samples using the PPF of the given distribution(s)
    # The _apply_ppf_to_lhs function needs to know if 'dist' was single or a list.
    # We pass 'dims' to it as single_dist_dims for that distinction.
    final_lhs_samples = _apply_ppf_to_lhs(
        unit_lhs_samples, distributions=dist, single_dist_dims=dims
    )

    return final_lhs_samples
