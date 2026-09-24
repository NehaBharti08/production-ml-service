"""FastNormalQuantileTransformer must be the parent's function, only faster.

It mirrors a private sklearn method, so these tests are the only thing standing
between it and a silent divergence after an sklearn upgrade. They require exact
equality, not closeness: the class exists on the promise that swapping it in
changes no score by any amount.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer

from mlservice.data.features import FastNormalQuantileTransformer


def _pair(**kwargs: object) -> tuple[QuantileTransformer, FastNormalQuantileTransformer]:
    params = {"n_quantiles": 1000, "output_distribution": "normal", "random_state": 42}
    params.update(kwargs)
    return QuantileTransformer(**params), FastNormalQuantileTransformer(**params)


@pytest.fixture
def credit_like(rng: np.random.Generator) -> np.ndarray:
    """Columns with the shapes that make quantile transforms awkward.

    Heavy right skew (income-like), a zero-dominated count (pub_rec is zero
    in 87.1% of loans), a bounded percentage, and missing values — so repeated
    quantiles, the clip bounds and the NaN path are all exercised.
    """
    n = 20_000
    income = rng.lognormal(11, 0.6, n)
    pub_rec = np.where(rng.random(n) < 0.871, 0.0, rng.integers(1, 6, n).astype(float))
    util = np.clip(rng.normal(50, 25, n), 0, 150)
    with_nan = rng.normal(0, 1, n)
    with_nan[rng.random(n) < 0.1] = np.nan
    return np.column_stack([income, pub_rec, util, with_nan])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260924)


class TestEquivalence:
    def test_forward_transform_is_bit_identical(self, credit_like: np.ndarray) -> None:
        slow, fast = _pair()
        slow.fit(credit_like)
        fast.fit(credit_like)
        np.testing.assert_array_equal(fast.transform(credit_like), slow.transform(credit_like))

    def test_out_of_range_values_hit_the_same_clip(self, credit_like: np.ndarray) -> None:
        """Values beyond the training range exercise the clip bounds, which
        this class precomputes rather than recomputing per column."""
        slow, fast = _pair()
        slow.fit(credit_like)
        fast.fit(credit_like)
        extreme = np.array([[1e12, 1e6, -1e6, 1e9], [-1e12, -1e6, 1e6, -1e9]])
        np.testing.assert_array_equal(fast.transform(extreme), slow.transform(extreme))

    def test_single_row_matches(self, credit_like: np.ndarray) -> None:
        """The serving shape. One row is where the overhead lived."""
        slow, fast = _pair()
        slow.fit(credit_like)
        fast.fit(credit_like)
        row = credit_like[:1].copy()
        np.testing.assert_array_equal(fast.transform(row), slow.transform(row))

    def test_inverse_is_untouched(self, credit_like: np.ndarray) -> None:
        slow, fast = _pair()
        slow.fit(credit_like)
        fast.fit(credit_like)
        z = slow.transform(credit_like)
        np.testing.assert_array_equal(fast.inverse_transform(z), slow.inverse_transform(z))

    def test_uniform_output_defers_to_the_parent(self, credit_like: np.ndarray) -> None:
        slow, fast = _pair(output_distribution="uniform")
        slow.fit(credit_like)
        fast.fit(credit_like)
        np.testing.assert_array_equal(fast.transform(credit_like), slow.transform(credit_like))


class TestItIsActuallyFaster:
    def test_single_row_is_materially_faster(self, credit_like: np.ndarray) -> None:
        """Positive control for the reason the class exists.

        Deliberately loose (2x, on a best-of-N timing) so it cannot flake on a
        loaded CI runner — the measured difference is ~5x. If this fails, the
        class is pure risk with no benefit and should be deleted.
        """
        import time

        wide = np.tile(credit_like, (1, 14))[:, :54]  # the champion's 54 numeric columns
        slow, fast = _pair()
        slow.fit(wide)
        fast.fit(wide)
        row = wide[:1].copy()

        def best(t: QuantileTransformer) -> float:
            for _ in range(5):
                t.transform(row)
            timings = []
            for _ in range(30):
                start = time.perf_counter()
                t.transform(row)
                timings.append(time.perf_counter() - start)
            return min(timings)

        assert best(slow) > 2 * best(fast)
