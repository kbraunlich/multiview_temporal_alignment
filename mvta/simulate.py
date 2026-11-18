"""Simulated movie responses for two groups with different HRFs."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d


@dataclass(frozen=True)
class MovieSimulation:
    """Stacked run data for each subject in the two groups.

    ``bold`` and ``mion`` are lists of arrays shaped
    ``(n_runs * n_tr, n_voxels)``. ``run_index`` labels the run of each row.
    """

    bold: list
    mion: list
    neural: np.ndarray
    run_index: np.ndarray
    tr: float
    n_shared: int


def _as_generator(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _zscore_columns(data):
    centered = data - data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return np.nan_to_num(centered / std)


def simulate_movie(
    bold,
    mion,
    *,
    n_tr=300,
    tr=2.5,
    n_runs=5,
    n_subjects=2,
    n_voxels=500,
    n_shared=2,
    n_group=2,
    n_idio=1,
    n_signals_per_voxel=3,
    event_prob=None,
    rng=0,
):
    """Simulate voxel time series for a BOLD group and a MION group.

    Each run has shared signals, group signals, and a subject-specific signal.
    Every voxel is a random mixture of ``n_signals_per_voxel`` of those
    signals, and the mixture stays the same across runs.     The mixed signal is
    convolved with that group's HRF and z-scored within the run.

    Each latent is a normalized average spike rate: an independent Uniform(0, 1)
    draw at every TR. Set ``event_prob`` to draw sparse binary events instead,
    each TR active with that probability.
    """
    bold = np.asarray(bold, dtype=float).reshape(-1, 1)
    mion = np.asarray(mion, dtype=float).reshape(-1, 1)
    n_signals = n_shared + n_group + n_idio
    if n_signals_per_voxel > n_signals:
        raise ValueError("n_signals_per_voxel cannot exceed the number of signals.")
    if n_subjects < 1 or n_runs < 2 or n_tr < 2 or n_voxels < 1:
        raise ValueError("Simulation needs at least 2 runs and 2 TRs per subject.")

    rng = _as_generator(rng)

    def draw(shape):
        if event_prob is None:
            return rng.random(shape)
        return rng.binomial(1, event_prob, shape).astype(float)

    groups = {"bold": [[] for _ in range(n_subjects)], "mion": [[] for _ in range(n_subjects)]}
    kernels = {"bold": bold, "mion": mion}
    shared_runs = []

    # Each voxel keeps the same mixture in every run, as a real voxel keeps
    # its tuning, so a shared space fit on some runs applies to a held-out run.
    mixing = {}
    for group in ("bold", "mion"):
        for subject in range(n_subjects):
            matrix = np.zeros((n_signals, n_voxels))
            for voxel in range(n_voxels):
                weights = rng.random(n_signals_per_voxel)
                weights /= weights.sum()
                padded = np.concatenate([weights, np.zeros(n_signals - n_signals_per_voxel)])
                matrix[:, voxel] = rng.permutation(padded)
            mixing[group, subject] = matrix

    for _ in range(n_runs):
        shared = draw((n_tr, n_shared))
        shared_runs.append(shared)
        for group in ("bold", "mion"):
            group_signal = draw((n_tr, n_group))
            for subject in range(n_subjects):
                idiosyncratic = draw((n_tr, n_idio))
                signals = np.hstack([shared, group_signal, idiosyncratic])
                neural = signals @ mixing[group, subject]
                convolved = convolve2d(neural, kernels[group])
                groups[group][subject].append(_zscore_columns(convolved[:n_tr]))

    run_index = np.repeat(np.arange(n_runs), n_tr)
    return MovieSimulation(
        bold=[np.vstack(subject) for subject in groups["bold"]],
        mion=[np.vstack(subject) for subject in groups["mion"]],
        neural=np.vstack(shared_runs),
        run_index=run_index,
        tr=float(tr),
        n_shared=int(n_shared),
    )
