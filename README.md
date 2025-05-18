# mcerp2: Monte Carlo Error Propagation (ndarray-based)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Or your chosen license -->

**Code:** [https://github.com/ugSUBMARINE/mcerp2](https://github.com/ugSUBMARINE/mcerp2)
**Original `mcerp` package:** [https://github.com/tisimst/mcerp](https://github.com/tisimst/mcerp)

## Overview

`mcerp2` is a Python package for performing Monte Carlo-based error propagation (also known as uncertainty analysis). It uses Latin Hypercube Sampling (LHS) to efficiently sample from input variable distributions and transparently propagate their uncertainties through mathematical calculations.

This package is a **reimplementation and modernization** of the original `mcerp` library created by Abraham Lee. The primary goal of `mcerp2` is to leverage a more direct integration with NumPy by subclassing `numpy.ndarray` for its core uncertain number objects. This approach aims to:

*   Simplify the internal codebase.
*   Provide more seamless compatibility with the broader NumPy ecosystem and its ufuncs.
*   Offer a potentially more performant and maintainable foundation.

With `mcerp2`, you can define variables with uncertainties using various statistical distributions and then perform calculations with them as if they were regular numbers. The library automatically tracks and quantifies the uncertainty in the results.

## Main Features (and Current Status)

*   **Transparent Uncertainty Propagation:** Perform calculations with uncertain variables directly.
    *   *Status:* Core arithmetic operations are functional.
*   **NumPy Integration:** Uncertain variables are `numpy.ndarray` subclasses, enabling direct use of many NumPy universal functions (ufuncs like `np.sqrt`, `np.exp`, `np.log`).
    *   *Status:* Implemented via `__array_ufunc__`. The `mcerp.umath` module from the original is no longer necessary; use `numpy` functions directly.
*   **Statistical Distribution Constructors:** Easily define uncertain variables from `scipy.stats` distributions (e.g., Normal, Uniform, Exponential).
    *   *Status:* A selection of common constructors (e.g., `N`, `U`, `Exp`, `Beta`, `Pois`) are available. More can be added.
*   **Latin Hypercube Sampling:** Efficiently samples input distributions for Monte Carlo simulation.
    *   *Status:* Randomized LHS is implemented and used by default.
*   **Correlation Enforcement:** Impose a specified correlation structure on a set of uncertain variables.
    *   *Status:* A `correlate` function is available.
*   **Descriptive Statistics:** Easily obtain mean, variance, standard deviation, skewness, and kurtosis of uncertain results.
    *   *Status:* Implemented as properties/methods on uncertain objects.
*   **Probabilistic Comparisons (vs. Scalars):** Determine probabilities like `P(X <= value)` using explicit methods (e.g., `x.cdf(value)`, `x.sf(value)`).
    *   *Status:* Implemented. Note that comparison operators (`<`, `==`, etc.) with scalars will behave like NumPy element-wise operations, returning a boolean `UncertainFunction`.

## Current Limitations & Differences from Original `mcerp`

`mcerp2` is an ongoing development effort. While core functionality is taking shape, there are notable differences and limitations compared to the original `mcerp` package:

*   **Plotting Functionality:** The convenient `.plot()` methods for visualizing distributions are **not yet implemented** in this version.
*   **`umath` Module Removed:** The `mcerp.umath` submodule is no longer present. Users should use NumPy's mathematical functions directly (e.g., `numpy.log(x)` instead of `mcerp.umath.log(x)`).
*   **Distribution Constructors:** Not all distribution constructors from the original `mcerp` may be implemented yet. The most common ones are prioritized.
*   **Comparison Operators:**
    *   Comparison operators (`<`, `<=`, `==`, etc.) between an `UncertainFunction` and a scalar or array will perform element-wise NumPy operations, resulting in a boolean `UncertainFunction`. To get a probability (e.g., `P(X < 5)`), you can use `(X < 5).mean()` or explicit methods like `X.cdf(5)`.
    *   Comparisons between two `UncertainFunction` objects (e.g., `X < Y`) are also handled via element-wise NumPy operations. The corresponding dunder methods have not been overridden.
*   **Advanced LHS & Statistical Wrappers:** Space-filling/orthogonal LHS and wrappers for `scipy.stats` functions (the `mcerp.stats` module) are not yet part of `mcerp2`.
*   **Correlation Induction:** The `correlate` function is implemented, but returns new `UnvertainFunction` objects. 
*   **API Stability:** As a new implementation, some API details might evolve.

## Installation

`mcerp2` works with Python 3.8+ and requires NumPy, and SciPy.

Currently, `mcerp2` is not yet on PyPI. To install it from this repository for development:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ugSUBMARINE/mcerp2.git
    cd mcerp2
    ```
2.  Create a virtual environment (recommended) and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .\.venv\Scripts\activate  # On Windows
    ```
3.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```
    Alternatively, if you are using `uv`:
    ```bash
    uv venv
    source .venv/bin/activate # or equivalent
    uv pip install -e .
    ```

## Quick Example

```python
import mcerp2
import numpy as np

# Set the number of Monte Carlo samples (optional)
mcerp2.set_npts(5000)

# Define uncertain variables
length = mcerp2.N(10.0, 0.1, tag='length_cm')  # Normal(mean=10, std=0.1)
width  = mcerp2.U(4.5, 5.5, tag='width_cm')    # Uniform between 4.5 and 5.5

# Perform calculations
area = length * width
area.tag = 'area_cm2'

# Get statistics
print(f"Area: {area.mean():.2f} +/- {area.std():.2f} cm^2")
area.describe()

# Probabilistic query
prob_area_gt_50 = (area > 50.0).mean() # P(Area > 50) using element-wise comparison
# or using explicit method:
# prob_area_gt_50_alt = area.prob_gt(50.0)
print(f"Probability Area > 50 cm^2: {prob_area_gt_50:.3f}")

# Correlation example (simplified)
import numpy as np
from mcerp2 import correlate
temp = mcerp2.N(25, 1, tag='temp_C')
press = mcerp2.N(101, 0.5, tag='pressure_kPa')
print(np.corrcoef(temp, press))  # Check correlation, should be close to 0.

corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
temp_corr, press_corr = correlate([temp, press], corr_matrix)
print(np.corrcoef(temp_corr, press_corr))  # Check correlation, should be close to 0.7
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details. <!-- Or your chosen license -->

## Acknowledgements

*   This package is heavily inspired by and aims to modernize the original `mcerp` package by Abraham Lee.
*   The core Latin Hypercube Sampling and correlation induction algorithms are adapted from established statistical methods.
