"""Cross-species stimulus epoch decoding in an SRM shared space."""

import numpy as np

from mvta.align import SRMSpace, fit_shared_filter, project_shared_scores
from mvta.fit import apply_kernel


def segment_matching(test, reference, win_size=10, column_weights=None):
    """Chance-corrected time-segment matching accuracy.

    ``test`` and ``reference`` are ``(n_time, n_components)`` series from the
    same movie. Every window of ``win_size`` TRs in ``test`` is matched to the
    window of ``reference`` with the highest correlation. A match is correct
    when it picks the same time. Reference windows that overlap the correct
    window are excluded, apart from the correct one itself. Chance is one over
    the number of candidate windows left for each test window.
    ``column_weights`` rescales components after each window column is
    standardized, so a zero weight drops that component.
    """
    return _match_accuracy(_window_correlation(test, reference, win_size, column_weights), win_size)


def decode_methods(
    bold_views,
    mion_views,
    run_index,
    kernels,
    *,
    bold_kernels=None,
    n_components,
    win_size=10,
    filter_shared=True,
    random_state=0,
    verbose=True,
):
    """Decode held-out runs in an SRM shared space for several temporal preparations.

    ``kernels`` maps a label to the kernel applied to every MION series before
    the shared space is fit. ``bold_kernels`` optionally maps method labels to
    kernels applied to every BOLD series for that method; labels omitted from
    it leave BOLD unchanged. This supports symmetric double convolution by
    applying the canonical BOLD HRF to MION and the canonical MION HRF to BOLD.
    A kernel is a 1-D array used for every run or a ``(n_runs, n_taps)`` array
    whose row ``i`` was learned with run ``i`` held out.

    For each held-out run, SRM with ``n_components`` is fit on the other runs.
    Decoding runs in both directions: each MION subject is decoded against the
    mean BOLD subject, and each BOLD subject against the mean MION subject.
    The score is the mean of the two directions, so neither species dominates
    when the groups differ in size. With
    ``filter_shared``, each training run in turn is a calibration run: SRM is
    fit on the remaining training runs, the cross-species filter is fit on the
    calibration run's scores, and the held-out run is rotated and weighted
    with it. The window correlation matrices are averaged over calibration
    runs before matching, because each calibration run has its own SRM basis.
    ``random_state`` seeds the filter's time-shift null.

    Returns a dict from label to an array of chance-corrected accuracy per run.
    """
    run_index = np.asarray(run_index)
    runs = np.unique(run_index)
    bold_kernels = {} if bold_kernels is None else dict(bold_kernels)
    unknown_labels = set(bold_kernels) - set(kernels)
    if unknown_labels:
        labels = ", ".join(sorted(unknown_labels))
        raise ValueError(f"bold_kernels contains unknown method labels: {labels}")
    if filter_shared and runs.size < 3:
        raise ValueError("filter_shared needs at least 3 runs: test, calibration, and fit.")
    requested_components = int(n_components)
    if requested_components < 1:
        raise ValueError("n_components must be at least 1.")
    max_components = min(
        np.asarray(view).shape[1] for view in list(bold_views) + list(mion_views)
    )
    n_components = min(requested_components, max_components)
    if n_components != requested_components:
        print(
            "SED shared-space dimensionality adjusted from "
            f"{requested_components} to {n_components} because the smallest "
            f"view has {max_components} features.",
            flush=True,
        )
    if verbose:
        filter_label = "with" if filter_shared else "without"
        print(
            f"SED spatial alignment: SRM, {n_components} components, "
            f"{filter_label} the cross-species filter",
            flush=True,
        )
    generator = np.random.default_rng(random_state)
    n_bold = len(bold_views)
    results = {label: [] for label in kernels}
    identified = {label: [] for label in kernels}
    for position, test_run in enumerate(runs):
        test = run_index == test_run
        for label, kernel in kernels.items():
            kernel = np.asarray(kernel, dtype=float)
            if kernel.ndim == 2:
                kernel = kernel[position]
            prepared_mion = [
                apply_kernel(view, kernel, run_index) for view in mion_views
            ]
            if label in bold_kernels:
                bold_kernel = np.asarray(bold_kernels[label], dtype=float)
                if bold_kernel.ndim == 2:
                    bold_kernel = bold_kernel[position]
                prepared_bold = [
                    apply_kernel(view, bold_kernel, run_index) for view in bold_views
                ]
            else:
                prepared_bold = bold_views
            views = list(prepared_bold) + list(prepared_mion)

            if not filter_shared:
                model = SRMSpace(n_components).fit([v[~test] for v in views])
                shared = model.transform([v[test] for v in views])
                correlations = _cross_species_correlations(
                    shared[:n_bold], shared[n_bold:], win_size
                )
                results[label].append(_two_way_accuracy(correlations, win_size))
                continue

            summed = None
            n_used = 0
            for calibration_run in runs[runs != test_run]:
                calibration = run_index == calibration_run
                fit_rows = ~test & ~calibration
                model = SRMSpace(n_components).fit([v[fit_rows] for v in views])
                calibration_scores = model.transform([v[calibration] for v in views])
                shared_filter = fit_shared_filter(
                    calibration_scores[:n_bold],
                    calibration_scores[n_bold:],
                    rng=generator,
                )
                identified[label].append(shared_filter.n_identified)
                if shared_filter.n_identified == 0:
                    continue
                shared = model.transform([v[test] for v in views])
                correlations = _cross_species_correlations(
                    [project_shared_scores(view, shared_filter.bold_directions) for view in shared[:n_bold]],
                    [project_shared_scores(view, shared_filter.mion_directions) for view in shared[n_bold:]],
                    win_size,
                    shared_filter.weights,
                )
                if summed is None:
                    summed = correlations
                else:
                    summed = [
                        [total + new for total, new in zip(totals, news)]
                        for totals, news in zip(summed, correlations)
                    ]
                n_used += 1
            if n_used == 0:
                results[label].append(0.0)
                continue
            averaged = [[total / n_used for total in totals] for totals in summed]
            results[label].append(_two_way_accuracy(averaged, win_size))

    if filter_shared and verbose:
        print("Directions above the time-shift null (mean over calibration runs)", flush=True)
        for label, counts in identified.items():
            print(f"  {label:20s} {np.mean(counts):.1f} of {n_components}", flush=True)
    return {label: np.asarray(values) for label, values in results.items()}


def _cross_species_correlations(bold_shared, mion_shared, win_size, column_weights=None):
    """Window correlations for both decoding directions.

    Returns ``[mion_to_bold, bold_to_mion]``: one matrix per MION subject
    against the mean BOLD subject, and one per BOLD subject against the mean
    MION subject.
    """
    bold_mean = np.mean(bold_shared, axis=0)
    mion_mean = np.mean(mion_shared, axis=0)
    return [
        [_window_correlation(view, bold_mean, win_size, column_weights) for view in mion_shared],
        [_window_correlation(view, mion_mean, win_size, column_weights) for view in bold_shared],
    ]


def _two_way_accuracy(correlations, win_size):
    """Mean over the two directions of the mean accuracy across subjects."""
    return float(
        np.mean(
            [
                np.mean([_match_accuracy(matrix, win_size) for matrix in direction])
                for direction in correlations
            ]
        )
    )


def _window_correlation(test, reference, win_size, column_weights=None):
    """Correlation between every test window and every reference window."""
    test = np.asarray(test, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n_segments = test.shape[0] - win_size + 1
    if n_segments < 2:
        raise ValueError("The series are too short for this window size.")
    test_windows = _windows(test, win_size, n_segments)
    reference_windows = _windows(reference, win_size, n_segments)
    if column_weights is not None:
        weights = np.asarray(column_weights, dtype=float).ravel()
        if weights.shape != (test.shape[1],):
            raise ValueError("column_weights must have one value per component.")
        lagged = np.tile(weights, win_size)
        keep = lagged > 0
        if not np.any(keep):
            raise ValueError("At least one column weight must be positive.")
        test_windows = test_windows[:, keep] * lagged[keep]
        reference_windows = reference_windows[:, keep] * lagged[keep]
    return np.corrcoef(test_windows, reference_windows)[:n_segments, n_segments:]


def _match_accuracy(correlation, win_size):
    n_segments = correlation.shape[0]
    index = np.arange(n_segments)
    overlap = np.abs(index[:, None] - index[None, :]) < win_size
    overlap &= ~np.eye(n_segments, dtype=bool)
    correlation = np.where(overlap, -np.inf, correlation)
    accuracy = np.mean(np.argmax(correlation, axis=1) == index)
    chance = np.mean(1.0 / (n_segments - overlap.sum(axis=1)))
    return float(accuracy - chance)


def _windows(data, win_size, n_segments):
    windows = np.hstack([data[offset : offset + n_segments] for offset in range(win_size)])
    centered = windows - windows.mean(axis=0)
    std = windows.std(axis=0)
    std[std == 0] = 1.0
    return centered / std
