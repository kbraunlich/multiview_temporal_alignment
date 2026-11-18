"""BOLD and MION hemodynamic response functions."""

import numpy as np
from nilearn.glm.first_level import spm_hrf
from scipy.interpolate import PchipInterpolator


def hrf_times(tr=2.5, oversampling=1, time_length=32.0, onset=0.0):
    """Sample times, in seconds, shared by the BOLD and MION kernels.

    The grid matches ``nilearn.glm.first_level.spm_hrf``: ``time_length / dt``
    samples spaced from 0 to ``time_length``, then shifted by ``onset``.
    """
    dt = tr / oversampling
    n_samples = int(np.rint(float(time_length) / dt))
    return np.linspace(0, time_length, n_samples) - onset


def bold_hrf(tr=2.5, oversampling=1, time_length=32.0, onset=0.0):
    """SPM difference-of-gammas BOLD HRF at the requested TR.

    Parameters
    ----------
    tr : float
        Repetition time, in seconds.
    oversampling : int
        Temporal oversampling factor. 1 returns one sample per TR.
    time_length : float
        Kernel duration, in seconds.
    onset : float
        Onset time, in seconds.
    """
    return np.asarray(
        spm_hrf(
            tr,
            oversampling=oversampling,
            time_length=time_length,
            onset=onset,
        ),
        dtype=float,
    )


def mion_hrf(tr=2.5, oversampling=1, time_length=32.0, onset=0.0):
    """MION HRF from Leite et al., NeuroImage 16:283–294 (2002).

    Parameters match :func:`bold_hrf`. The kernel keeps its native negative
    polarity and is scaled so the absolute value of its samples sums to 1.
    """
    time_stamps = hrf_times(tr, oversampling, time_length, onset)
    hrf = -16.4486 * (
        -0.184 / 1.5 * np.exp(-time_stamps / 1.5)
        + 0.330 / 4.5 * np.exp(-time_stamps / 4.5)
        + 0.670 / 13.5 * np.exp(-time_stamps / 13.5)
    )
    hrf /= np.abs(hrf.sum())
    return hrf


def variable_bold_hrf(
    tr=1.0,
    *,
    time_to_peak=6.1,
    positive_fwhm=4.1,
    time_to_undershoot=10.8,
    undershoot_ratio=0.21,
    time_length=32.0,
):
    """Phenomenological positive BOLD HRF controlled by four shape descriptors.

    The curve is a monotone piecewise-cubic interpolation through baseline,
    the positive lobe's half-height crossings and peak, a zero crossing, and
    the undershoot minimum. Its positive peak is one, so ``undershoot_ratio``
    is the magnitude of the negative minimum relative to that peak.

    These parameters describe the sampled simulation HRF; they are not the
    internal parameters of the SPM difference-of-gammas model.
    """
    tr = float(tr)
    time_length = float(time_length)
    time_to_peak = float(time_to_peak)
    positive_fwhm = float(positive_fwhm)
    time_to_undershoot = float(time_to_undershoot)
    undershoot_ratio = float(undershoot_ratio)
    if tr <= 0 or time_length <= tr:
        raise ValueError("tr must be positive and time_length must exceed tr.")
    if positive_fwhm <= 0:
        raise ValueError("positive_fwhm must be positive.")
    if undershoot_ratio < 0:
        raise ValueError("undershoot_ratio must be nonnegative.")

    half_rise = time_to_peak - positive_fwhm / 2.0
    half_fall = time_to_peak + positive_fwhm / 2.0
    if half_rise <= 0:
        raise ValueError("time_to_peak must exceed half of positive_fwhm.")
    if time_to_undershoot <= half_fall:
        raise ValueError(
            "time_to_undershoot must follow the positive lobe's falling half-height."
        )

    n_samples = int(np.rint(time_length / tr))
    times = np.arange(n_samples, dtype=float) * tr
    if times.size < 2 or time_to_undershoot >= times[-1]:
        raise ValueError("time_length must extend beyond time_to_undershoot.")

    zero_crossing = half_fall + 0.5 * (time_to_undershoot - half_fall)
    anchors_t = [0.0, half_rise, time_to_peak, half_fall, zero_crossing]
    anchors_y = [0.0, 0.5, 1.0, 0.5, 0.0]
    if undershoot_ratio > 0:
        recovery_time = time_to_undershoot + max(6.0, 1.5 * positive_fwhm)
        recovery_time = min(recovery_time, times[-1])
        if recovery_time <= time_to_undershoot:
            raise ValueError("time_length is too short for the undershoot recovery.")
        anchors_t.extend([time_to_undershoot, recovery_time])
        anchors_y.extend([-undershoot_ratio, 0.0])
    if anchors_t[-1] < times[-1]:
        anchors_t.append(times[-1])
        anchors_y.append(0.0)

    return np.asarray(PchipInterpolator(anchors_t, anchors_y)(times), dtype=float)


def hrf_shape_metrics(hrf, tr=1.0):
    """Measure peak time, positive FWHM, undershoot time, and depth ratio."""
    values = np.asarray(hrf, dtype=float).ravel()
    tr = float(tr)
    if values.size < 3 or tr <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("hrf must contain at least three finite samples and tr must be positive.")

    times = np.arange(values.size, dtype=float) * tr
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    if peak <= 0:
        raise ValueError("hrf must have a positive peak.")
    half = peak / 2.0
    left = _half_height_crossing(
        times[: peak_index + 1], values[: peak_index + 1], half, rising=True
    )
    right = _half_height_crossing(
        times[peak_index:], values[peak_index:], half, rising=False
    )

    after_peak = values[peak_index + 1 :]
    if after_peak.size and float(after_peak.min()) < 0:
        undershoot_index = peak_index + 1 + int(np.argmin(after_peak))
        undershoot_time = float(times[undershoot_index])
        undershoot_ratio = float(-values[undershoot_index] / peak)
    else:
        undershoot_time = float("nan")
        undershoot_ratio = 0.0

    return {
        "time_to_peak": float(times[peak_index]),
        "positive_fwhm": float(right - left),
        "time_to_undershoot": undershoot_time,
        "undershoot_ratio": undershoot_ratio,
    }


def _half_height_crossing(times, values, half, *, rising):
    """Linearly interpolate one half-height crossing around a monotone lobe."""
    if rising:
        candidates = np.flatnonzero(values <= half)
        lower = int(candidates[-1]) if candidates.size else 0
        upper = min(lower + 1, values.size - 1)
    else:
        candidates = np.flatnonzero(values <= half)
        upper = int(candidates[0]) if candidates.size else values.size - 1
        lower = max(upper - 1, 0)
    if lower == upper or values[upper] == values[lower]:
        return float(times[lower])
    fraction = (half - values[lower]) / (values[upper] - values[lower])
    return float(times[lower] + fraction * (times[upper] - times[lower]))
