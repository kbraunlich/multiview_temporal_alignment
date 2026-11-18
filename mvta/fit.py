"""Cross-validated convolutional-kernel objective."""

import numpy as np
from scipy import optimize
from scipy.signal import convolve2d

from mvta.align import (
    SRMSpace,
    apply_filter_weights,
    fit_correlation_filter,
    mean_cross_disparity,
)


def apply_kernel(data, kernel, run_index):
    """Convolve each run with ``kernel``, crop, and z-score columns.

    ``data`` is ``(n_time, n_voxels)`` and is not modified. Runs are convolved
    separately so the kernel does not cross run boundaries.
    """
    data = np.asarray(data, dtype=float)
    kernel = np.asarray(kernel, dtype=float).reshape(-1, 1)
    corrected = np.empty_like(data)
    for run in np.unique(run_index):
        selected = run_index == run
        block = data[selected]
        convolved = convolve2d(block, kernel)[: block.shape[0]]
        corrected[selected] = _zscore_columns(convolved)
    return corrected


def cross_species_loss(
    kernel,
    bold_views,
    mion_views,
    run_index,
    objective="multiview",
    n_components=None,
):
    """Mean held-out between-species disparity for one candidate kernel.

    ``objective`` is ``"multiview"``, ``"voxel_average"``, or ``"no_alignment"``.
    The multiview objective fits SRM on the inner training runs, rotates and
    weights the scores toward BOLD–MION correlation, and scores the inner test
    run. That kernel-learning space is separate from the SED shared space.
    """
    corrected = [apply_kernel(view, kernel, run_index) for view in mion_views]
    losses = []
    for test_run in np.unique(run_index):
        test = run_index == test_run
        train = ~test
        bold_test = [view[test] for view in bold_views]
        mion_test = [view[test] for view in corrected]
        if objective == "voxel_average":
            bold_test = [_column_mean(view) for view in bold_test]
            mion_test = [_column_mean(view) for view in mion_test]
            losses.append(mean_cross_disparity(bold_test, mion_test))
        elif objective == "no_alignment":
            losses.append(mean_cross_disparity(bold_test, mion_test))
        elif objective == "multiview":
            if n_components is None:
                raise ValueError(
                    "n_components is required for the multiview objective."
                )
            losses.append(
                _multiview_fold_loss(
                    [view[train] for view in bold_views],
                    [view[train] for view in corrected],
                    bold_test,
                    mion_test,
                    n_components,
                )
            )
        else:
            raise ValueError(
                "objective must be 'multiview', 'voxel_average', or 'no_alignment'."
            )
    return float(np.mean(losses))


def learn_kernels(
    bold_views,
    mion_views,
    run_index,
    *,
    kernel_length,
    objective="multiview",
    n_components=None,
    bold_hrf=None,
    mion_hrf=None,
    gtol=1e-5,
    maxiter=500,
    verbose=False,
):
    """Learn one convolutional kernel per held-out run.

    Returns an array shaped ``(n_runs, kernel_length)``, in ascending run
    order. Each row was fit with that run held out. BFGS starts from a flat
    kernel that sums to one. For the multiview objective, every objective
    evaluation refits SRM on every inner fold, so the full default settings
    are slow. ``n_components`` is the SRM rank used only for that objective.

    The shared-space loss is unchanged when the kernel is multiplied by -1,
    because the shared space absorbs that sign. When both HRFs are passed, each
    fitted kernel is given the sign that best matches the least-squares kernel
    mapping the MION HRF onto the BOLD HRF.
    """
    run_index = np.asarray(run_index)
    kernel_length = int(kernel_length)
    if kernel_length < 1:
        raise ValueError("kernel_length must be at least 1.")
    if objective == "multiview" and n_components is None:
        raise ValueError("n_components is required for the multiview objective.")
    if (bold_hrf is None) != (mion_hrf is None):
        raise ValueError("Pass both bold_hrf and mion_hrf to resolve the kernel sign.")
    reference = None
    if bold_hrf is not None:
        reference = _mion_to_bold_kernel(bold_hrf, mion_hrf, kernel_length)
    runs = np.unique(run_index)
    initial = np.ones(kernel_length) / kernel_length
    kernels = []
    for fold, test_run in enumerate(runs, start=1):
        if verbose:
            print(f"Fold {fold}/{len(runs)}: holding out run {test_run}", flush=True)
        train = run_index != test_run
        result = optimize.minimize(
            cross_species_loss,
            initial,
            args=(
                [view[train] for view in bold_views],
                [view[train] for view in mion_views],
                run_index[train],
                objective,
                n_components,
            ),
            method="BFGS",
            options={"gtol": gtol, "maxiter": maxiter, "disp": False},
        )
        kernel = np.asarray(result.x, dtype=float)
        if reference is not None:
            kernel = _match_kernel_sign(kernel, reference)
        kernels.append(kernel)
        if verbose and not result.success:
            print(f"  stopped after {result.nit} iterations: {result.message}", flush=True)
    return np.vstack(kernels)


def sse_loss(kernel, bold, mion, run_index):
    """Sum of squared errors between BOLD and MION convolved with ``kernel``.

    ``bold`` and ``mion`` are single time series of the same length, stacked
    over runs and labeled by ``run_index``. Each run is convolved separately.
    The first ``len(kernel) - 1`` samples of each run are left out, because
    predicting them needs MION samples from before the run began.
    """
    kernel = np.asarray(kernel, dtype=float).ravel()
    bold = np.asarray(bold, dtype=float).ravel()
    mion = np.asarray(mion, dtype=float).ravel()
    run_index = np.asarray(run_index)
    skip = kernel.size - 1
    sse = 0.0
    for run in np.unique(run_index):
        selected = run_index == run
        predicted = np.convolve(mion[selected], kernel)[: selected.sum()]
        sse += np.sum((bold[selected][skip:] - predicted[skip:]) ** 2)
    return float(sse)


def sse_kernel(bold, mion, run_index, *, kernel_length, gtol=1e-5, maxiter=500):
    """Learn the kernel minimizing :func:`sse_loss` with BFGS.

    The optimizer and its flat starting kernel, which sums to one, are the
    same as in :func:`learn_kernels`; only the loss differs. Returns the
    kernel and the ``scipy.optimize.OptimizeResult``.
    """
    bold = np.asarray(bold, dtype=float).ravel()
    mion = np.asarray(mion, dtype=float).ravel()
    run_index = np.asarray(run_index)
    length = int(kernel_length)
    if length < 1:
        raise ValueError("kernel_length must be at least 1.")
    if not bold.size == mion.size == run_index.size:
        raise ValueError("bold, mion, and run_index must have the same length.")
    result = optimize.minimize(
        sse_loss,
        np.ones(length) / length,
        args=(bold, mion, run_index),
        method="BFGS",
        options={"gtol": gtol, "maxiter": maxiter, "disp": False},
    )
    return np.asarray(result.x, dtype=float), result


_EMPTY_FILTER_LOSS = 1e3


def _multiview_fold_loss(bold_train, mion_train, bold_test, mion_test, n_components):
    """SRM on inner-train, correlation weights from train, disparity on test."""
    model = SRMSpace(n_components)
    train_views = list(bold_train) + list(mion_train)
    model.fit(train_views)
    train_shared = model.transform(train_views)
    n_bold = len(bold_train)
    shared_filter = fit_correlation_filter(
        train_shared[:n_bold], train_shared[n_bold:]
    )
    test_shared = model.transform(list(bold_test) + list(mion_test))
    bold_filtered = [
        apply_filter_weights(view, shared_filter.bold_directions, shared_filter.weights)
        for view in test_shared[:n_bold]
    ]
    mion_filtered = [
        apply_filter_weights(view, shared_filter.mion_directions, shared_filter.weights)
        for view in test_shared[n_bold:]
    ]
    if any(view is None for view in bold_filtered + mion_filtered):
        return _EMPTY_FILTER_LOSS
    return mean_cross_disparity(bold_filtered, mion_filtered)


def _mion_to_bold_kernel(bold, mion, kernel_length):
    """Least-squares kernel that maps the MION HRF onto the BOLD HRF."""
    bold = np.asarray(bold, dtype=float).ravel()
    mion = np.asarray(mion, dtype=float).ravel()
    n_time = min(bold.size, mion.size)
    length = int(kernel_length)
    design = np.zeros((n_time, length))
    for lag in range(length):
        design[lag:, lag] = mion[: n_time - lag]
    kernel, *_ = np.linalg.lstsq(design, bold[:n_time], rcond=None)
    return np.asarray(kernel, dtype=float)


def _match_kernel_sign(kernel, reference):
    """Return ``kernel`` or its negative, whichever matches ``reference``."""
    kernel = np.asarray(kernel, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n_tap = min(kernel.size, reference.size)
    learned = kernel[:n_tap] - kernel[:n_tap].mean()
    known = reference[:n_tap] - reference[:n_tap].mean()
    if np.dot(learned, known) < 0:
        return -kernel
    return kernel


def _zscore_columns(data):
    centered = data - data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return centered / std


def _column_mean(data):
    return np.asarray(data, dtype=float).mean(axis=1, keepdims=True)
