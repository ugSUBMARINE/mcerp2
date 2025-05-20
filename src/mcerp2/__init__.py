# mcerp2/__init__.py
from __future__ import annotations

from .core import (
    UncertainFunction,
    UncertainVariable,
    config as mcerp_config,  # Expose config object
)
from .correlate import correlate
from .distributions import *

__version__ = "0.1.0"
__author__ = "Karl Gruber"

# The primary way to create an uncertain variable
uv = UncertainVariable


# Expose npts directly and provide a setter function
def get_npts() -> int:
    return mcerp_config.npts


def set_npts(new_npts: int) -> None:
    if not isinstance(new_npts, int) or new_npts <= 0:
        raise ValueError("npts must be a positive integer.")
    mcerp_config.npts = new_npts
    # Optionally, you might want to warn users if variables have already been created
    # with the old npts, as they won't be updated.
    print(
        f"mcerp2.npts set to {new_npts}. Note: Existing UncertainVariables will not be resampled."
    )
