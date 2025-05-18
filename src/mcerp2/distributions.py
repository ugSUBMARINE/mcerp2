# mcerp2/constructors.py
from __future__ import annotations
from scipy import stats
from .core import UncertainVariable, config  # Import uv (UncertainVariable) and config


# --- CONTINUOUS DISTRIBUTIONS ---
def Normal(mu: float, sigma: float, tag: str | None = None) -> UncertainVariable:
    """A Normal (Gaussian) random variate."""
    if not sigma > 0:
        raise ValueError('Normal "sigma" must be greater than zero')
    if tag is None:
        tag = f"Normal(mu={mu}, sigma={sigma})"
    return UncertainVariable(stats.norm(loc=mu, scale=sigma), tag=tag)


N = Normal


def Uniform(low: float, high: float, tag: str | None = None) -> UncertainVariable:
    """A Uniform random variate."""
    if not low < high:
        raise ValueError('Uniform "low" must be less than "high"')
    if tag is None:
        tag = f"Uniform(low={low}, high={high})"
    return UncertainVariable(stats.uniform(loc=low, scale=high - low), tag=tag)


U = Uniform


def Exponential(lamda: float, tag: str | None = None) -> UncertainVariable:
    """An Exponential random variate. lamda is the rate parameter (1/scale)."""
    if not lamda > 0:
        raise ValueError('Exponential "lamda" (rate) must be greater than zero')
    if tag is None:
        tag = f"Exponential(lamda={lamda})"
    return UncertainVariable(stats.expon(scale=1.0 / lamda), tag=tag)


Exp = Exponential


def Beta(
    alpha: float, beta: float, low: float = 0, high: float = 1, tag: str | None = None
) -> UncertainVariable:
    """A Beta random variate."""
    if not (alpha > 0 and beta > 0):
        raise ValueError(
            'Beta "alpha" and "beta_param" parameters must be greater than zero'
        )
    if not low < high:
        raise ValueError('Beta "low" must be less than "high"')
    if tag is None:
        tag = f"Beta(alpha={alpha}, beta={beta}, low={low}, high={high})"
    return UncertainVariable(
        stats.beta(alpha, beta, loc=low, scale=high - low), tag=tag
    )


# TODO: Add more continuous distributions as needed


# --- DISCRETE DISTRIBUTIONS ---
def Bernoulli(p: float, tag: str | None = None) -> UncertainVariable:
    """A Bernoulli random variate."""
    if not (0 <= p <= 1):
        raise ValueError(
            'Bernoulli probability "p" must be between zero and one, inclusive'
        )
    if tag is None:
        tag = f"Bernoulli(p={p})"
    return UncertainVariable(stats.bernoulli(p), tag=tag)


Bern = Bernoulli


def Poisson(lamda: float, tag: str | None = None) -> UncertainVariable:
    """A Poisson random variate."""
    if not lamda >= 0:
        raise ValueError('Poisson "lamda" must be non-negative.')
    if tag is None:
        tag = f"Poisson(lamda={lamda})"
    return UncertainVariable(stats.poisson(mu=lamda), tag=tag)


Pois = Poisson

# TODO: Add more discrete distributions as needed
