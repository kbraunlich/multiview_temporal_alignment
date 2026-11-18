"""Between-group disparity, the SRM shared space, and the cross-species filter."""

import itertools
from dataclasses import dataclass

import numpy as np

from mvta.srm import DetSRM


def disparity(left, right):
    """Sum of squared differences after centering and unit-energy scaling.

    The inputs are copied and are not modified.
    """
    return float(np.sum((_unit_energy(left) - _unit_energy(right)) ** 2))


def mean_cross_disparity(group_a, group_b):
    """Mean disparity over every pair of views drawn from the two groups."""
    scores = [disparity(a, b) for a, b in itertools.product(group_a, group_b)]
    if not scores:
        raise ValueError("Both groups need at least one view.")
    return float(np.mean(scores))


class SRMSpace:
    """Deterministic SRM with a time × voxels interface.

    ``fit`` and ``transform`` take a list of ``(n_time, n_voxels)`` arrays.
    Voxel counts may differ across views; the number of time points must match
    at fit time. Held-out scores are ``(n_time, n_components)``.
    """

    def __init__(self, n_components, n_iter=10, random_state=0):
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.random_state = random_state
        self._model = None

    def fit(self, views):
        data = [_as_voxels_by_time(view) for view in views]
        n_time = data[0].shape[1]
        min_voxels = min(view.shape[0] for view in data)
        n_components = min(self.n_components, n_time, min_voxels)
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self._model = DetSRM(
            n_iter=self.n_iter,
            features=n_components,
            rand_seed=self.random_state,
        )
        self._model.fit(data)
        return self

    def transform(self, views):
        if self._model is None:
            raise RuntimeError("Call fit before transform.")
        scores = self._model.transform([_as_voxels_by_time(view) for view in views])
        return [np.asarray(view, dtype=float).T for view in scores]


@dataclass(frozen=True)
class SharedFilter:
    """Cross-species rotation and weighting of one shared space.

    ``bold_directions`` and ``mion_directions`` rotate that group's scores
    into the singular vectors of the mean BOLD–MION correlation matrix.
    ``weights`` has one entry per direction and is zero for directions whose
    correlation does not exceed the time-shifted null.
    """

    bold_directions: np.ndarray
    mion_directions: np.ndarray
    weights: np.ndarray

    @property
    def n_identified(self):
        """Number of directions whose correlation exceeds the null."""
        return int(np.count_nonzero(self.weights > 0))


def fit_correlation_filter(bold_scores, mion_scores):
    """Rotate toward BOLD–MION correlation and weight by singular values.

    Same SVD as :func:`fit_shared_filter`, without the time-shift null. Kernel
    learning uses this inside BFGS, where a stochastic null would add noise.
    """
    bold_scores = [np.asarray(view, dtype=float) for view in bold_scores]
    mion_scores = [np.asarray(view, dtype=float) for view in mion_scores]
    observed = _mean_cross_correlation(bold_scores, mion_scores)
    bold_directions, observed_s, mion_directions_t = np.linalg.svd(
        observed, full_matrices=False
    )
    return SharedFilter(
        bold_directions=bold_directions,
        mion_directions=mion_directions_t.T,
        weights=np.asarray(observed_s, dtype=float),
    )


def fit_shared_filter(bold_scores, mion_scores, *, n_shifts=20, percentile=95.0, rng=0):
    """Fit the cross-species filter from shared-space scores.

    Each score array is ``(n_time, n_components)``. The scores should come
    from runs the shared space was not fit to; otherwise the fit itself
    inflates the cross-species correlations. The null circularly shifts every
    MION series by a random offset between a quarter and three quarters of
    the series, ``n_shifts`` times, and keeps the largest singular value of
    each shifted correlation matrix. A direction's weight is the square root
    of its squared singular value minus the ``percentile`` of those squared
    null maxima, floored at zero. Comparing every rank with the null maximum,
    not with the null value at the same rank, keeps noise directions from
    passing once the shared directions are used up.
    """
    bold_scores = [np.asarray(view, dtype=float) for view in bold_scores]
    mion_scores = [np.asarray(view, dtype=float) for view in mion_scores]
    n_time = mion_scores[0].shape[0]
    if n_time < 4:
        raise ValueError("The filter needs at least 4 time points.")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    observed = _mean_cross_correlation(bold_scores, mion_scores)
    bold_directions, observed_s, mion_directions_t = np.linalg.svd(
        observed, full_matrices=False
    )
    shifts = generator.integers(n_time // 4, 3 * n_time // 4 + 1, size=int(n_shifts))
    null_max = np.array(
        [
            np.linalg.svd(
                _mean_cross_correlation(
                    bold_scores, [np.roll(view, shift, axis=0) for view in mion_scores]
                ),
                compute_uv=False,
            )[0]
            for shift in shifts
        ]
    )
    threshold = np.percentile(null_max**2, percentile)
    weights = np.sqrt(np.maximum(observed_s**2 - threshold, 0.0))
    return SharedFilter(
        bold_directions=bold_directions,
        mion_directions=mion_directions_t.T,
        weights=weights,
    )


def project_shared_scores(scores, directions):
    """Z-score shared-space scores, then rotate them into filter directions."""
    return _zscore(np.asarray(scores, dtype=float)) @ np.asarray(directions, dtype=float)


def apply_filter_weights(scores, directions, weights):
    """Rotate scores and scale columns by filter weights, dropping zeros.

    Returns ``None`` when every weight is zero.
    """
    projected = project_shared_scores(scores, directions)
    weights = np.asarray(weights, dtype=float).ravel()
    keep = weights > 0
    if not np.any(keep):
        return None
    return projected[:, keep] * weights[keep]


def _mean_cross_correlation(group_a, group_b):
    """Mean component-by-component correlation matrix over every pair of views."""
    total = 0.0
    n_pairs = 0
    for left, right in itertools.product(group_a, group_b):
        total = total + _zscore(left).T @ _zscore(right) / max(left.shape[0] - 1, 1)
        n_pairs += 1
    if n_pairs == 0:
        raise ValueError("Both groups need at least one view.")
    return total / n_pairs


def _zscore(data):
    centered = data - data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=1, keepdims=True)
    return np.nan_to_num(centered / np.where(std > 0, std, 1.0))


def _as_voxels_by_time(view):
    array = np.asarray(view, dtype=float)
    if array.ndim != 2:
        raise ValueError("Each view must be a 2-D array of shape (n_time, n_voxels).")
    return array.T


def _unit_energy(data):
    scaled = np.array(data, dtype=float, copy=True)
    scaled -= scaled.mean(axis=0)
    norm = np.linalg.norm(scaled)
    if norm > 0:
        scaled /= norm
    return scaled
