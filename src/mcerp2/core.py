# mcerp2/core.py
from __future__ import annotations

import numpy as np
from .lhd import lhd


# --- Configuration ---
class MCERPConfig:
    npts = 10000  # Default number of Monte Carlo points


config = MCERPConfig()  # Global config object


# --- UncertainFunction: The ndarray Subclass ---
class UncertainFunction(np.ndarray):
    __array_priority__ = 15.0

    def __new__(cls, input_array, tag: str | None = None, dtype=None, order=None):
        # Ensure input_array is array-like, then view as this class
        obj = np.asarray(input_array, dtype=dtype, order=order).view(cls)
        obj.tag = tag
        return obj

    def __array_finalize__(self, parent_obj):
        if parent_obj is None:
            return
        # Propagate tag from parent object (e.g., from slicing or view)
        self.tag = getattr(parent_obj, "tag", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        # input_tags = []
        for x_in in inputs:
            if isinstance(x_in, UncertainFunction):
                args.append(x_in.view(np.ndarray))
                # input_tags.append(x_in.tag)
            else:
                args.append(x_in)

        outputs = kwargs.pop("out", None)
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, UncertainFunction):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == "at":  # In-place operations
            return None

        # The tag of the result is set to None
        result_tag = None

        if ufunc.nout == 1:
            if isinstance(results, np.ndarray):
                results = results.view(UncertainFunction)
                results.tag = result_tag
        else:
            results = tuple(
                (r.view(UncertainFunction) if isinstance(r, np.ndarray) else r)
                for r in results
            )
            for res_item in results:
                if isinstance(res_item, UncertainFunction):
                    res_item.tag = result_tag
        return results

    # --- Properties ---
    @property
    def skew(self):
        mn = self.mean()
        sd = self.std()
        if np.abs(sd) <= 1e-15:
            return 0.0  # Avoid division by zero for constant arrays
        return np.mean((self - mn) ** 3) / sd**3

    @property
    def kurt(self):  # Pearson's kurtosis
        mn = self.mean()
        sd = self.std()
        if np.abs(sd) <= 1e-15:
            return 0.0  # For constant arrays, kurtosis can be ill-defined or 0
        return np.mean((self - mn) ** 4) / sd**4

    @property
    def stats(self):
        return self.mean(), self.var(), self.skew, self.kurt

    # --- Methods ---
    def cdf(self, val):
        """
        Empirical cumulative distribution function (CDF) for the uncertain function
        evaluated at val.
        val ... can be a scalar or an array-like object.
        """
        val = np.atleast_1d(val)
        cdf = np.mean(self <= val[:, np.newaxis], axis=1)
        return np.asarray(cdf) if cdf.size > 1 else cdf[0]

    def sf(self, val):
        """
        Empirical survival function (SF) for the uncertain function
        evaluated at val.
        val ... can be a scalar or an array-like object.
        """
        return 1.0 - self.cdf(val)

    def ppf(self, q):
        """
        Empirical percent-point function (PPF) for the uncertain function
        evaluated at q.
        q ... can be a scalar or an array-like object.
        """
        q = np.clip(np.asarray(q), 0.0, 1.0)
        ppf = np.quantile(self, q, method="linear")
        return np.asarray(ppf) if ppf.size > 1 else ppf

    def isf(self, q):
        """
        Empirical inverse survival function (ISF) for the uncertain function
        evaluated at q.
        q ... can be a scalar or an array-like object.
        """
        q = np.clip(np.asarray(q), 0.0, 1.0)
        return self.ppf(1.0 - q)

    def interval(self, confidence=0.95):
        """
        Empirical confidence interval for the uncertain function
        evaluated at confidence level.
        confidence ... float between 0 and 1.
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("Confidence level must be between 0 and 1.")
        alpha = (1.0 - confidence) / 2.0
        lower_bound = self.ppf(alpha)
        upper_bound = self.ppf(1.0 - alpha)
        return lower_bound, upper_bound

    def describe(self, name=None):
        mn, vr, sk, kt = self.stats
        header_name = name if name is not None else self.tag
        s = (
            f"MCERP Uncertain Value ({header_name}):\n"
            if header_name
            else "MCERP Uncertain Value:\n"
        )
        s += f" > Mean................... {mn: }\n"
        s += f" > Variance............... {vr: }\n"
        s += f" > Skewness Coefficient... {sk: }\n"
        s += f" > Kurtosis Coefficient... {kt: }\n"
        print(s)

    # --- String Representation ---
    def __repr__(self):
        # Use ndarray's repr and append tag info
        ndarray_repr = super().__repr__()
        # Remove the closing parenthesis of ndarray_repr, add tag, then add new parenthesis
        # e.g., array([1, 2]) tag=None) -> array([1, 2], tag=None)
        if ndarray_repr.endswith(")"):
            return f"{ndarray_repr[:-1]}, tag={self.tag!r})"
        else:  # Should not happen for standard ndarray repr
            return f"{ndarray_repr}, tag={self.tag!r}"


# --- UncertainVariable: Subclass for Scipy Stats integration ---
class UncertainVariable(UncertainFunction):
    def __new__(cls, rv, tag=None, dtype=None, order=None):
        if not hasattr(rv, "dist") or not hasattr(rv, "ppf"):
            raise ValueError(
                "Input 'rv' must be a scipy.stats frozen distribution with a 'ppf' method."
            )

        # Generate samples using lhd
        mcpts_data = lhd(dist=rv, size=config.npts).flatten()

        # Create the object using UncertainFunction's __new__
        obj = UncertainFunction.__new__(
            cls, mcpts_data, tag=tag, dtype=dtype, order=order
        )
        obj.rv = rv  # Store the original scipy.stats distribution object
        return obj

    def __array_finalize__(self, parent_obj):
        super().__array_finalize__(parent_obj)
        # Propagate 'rv' if the parent had it (e.g., simple view or copy)
        # If parent_obj resulted from a ufunc, it might be UncertainFunction, not UncertainVariable
        self.rv = getattr(parent_obj, "rv", None)
