"""Four-panel figure for the MVTA demo."""

import matplotlib.pyplot as plt
import numpy as np

from mvta.hrf import hrf_shape_metrics

BOLD_COLOR = "#d62728"
MION_COLOR = "#1f77b4"
MVTA_COLOR = "black"


def plot_neural(neural, run_index, tr, plot_run=0, n_tr_plot=60):
    """Plot the shared neural movie signals for one run.

    These are the signals generated before mixing into voxels and convolution
    with an HRF. Columns are the shared components. The default window is the
    first 60 TRs, the same stretch shown for the BOLD and MION movie responses.
    """
    neural = np.asarray(neural, dtype=float)
    if neural.ndim == 1:
        neural = neural.reshape(-1, 1)
    segment = neural[np.asarray(run_index) == plot_run]
    n_show = min(int(n_tr_plot), segment.shape[0])
    segment = segment[:n_show]
    time = np.arange(n_show) * tr
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    _style(ax)
    colors = ["#222222", "#7f7f7f", "#4c78a8", "#54a24b"]
    for component in range(segment.shape[1]):
        label = "Shared" if segment.shape[1] == 1 else f"Shared {component + 1}"
        ax.plot(
            time,
            segment[:, component],
            color=colors[component % len(colors)],
            lw=1.6,
            label=label,
        )
    if segment.shape[1] > 1:
        ax.legend(
            frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized spike rate")
    ax.set_title(f"Simulated spike rate at TR={tr:g}")
    return fig


def plot_hrf_shapes(selected_bold, canonical_bold, mion, *, tr):
    """Plot the selected BOLD HRF beside canonical BOLD and fixed MION."""
    selected_bold = np.asarray(selected_bold, dtype=float).ravel()
    canonical_bold = np.asarray(canonical_bold, dtype=float).ravel()
    mion = np.asarray(mion, dtype=float).ravel()
    metrics = hrf_shape_metrics(selected_bold, tr)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    _style(ax)
    ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
    ax.plot(
        np.arange(selected_bold.size) * tr,
        _peak_norm(selected_bold),
        color=BOLD_COLOR,
        lw=2.2,
        label="Selected BOLD",
    )
    ax.plot(
        np.arange(canonical_bold.size) * tr,
        _peak_norm(canonical_bold),
        color=BOLD_COLOR,
        lw=1.5,
        ls="--",
        alpha=0.7,
        label="Canonical SPM BOLD",
    )
    ax.plot(
        np.arange(mion.size) * tr,
        _peak_norm(mion),
        color=MION_COLOR,
        lw=1.5,
        alpha=0.8,
        label="Fixed MION",
    )
    ax.axvline(
        metrics["time_to_peak"],
        color=BOLD_COLOR,
        lw=0.9,
        ls=":",
        label=f"TTP {metrics['time_to_peak']:g} s",
    )
    if np.isfinite(metrics["time_to_undershoot"]):
        ax.axvline(
            metrics["time_to_undershoot"],
            color="#9467bd",
            lw=0.9,
            ls=":",
            label=f"TTU {metrics['time_to_undershoot']:g} s",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Peak-normalized response")
    ax.set_title(
        "Selected HRF: "
        f"FWHM {metrics['positive_fwhm']:.2f} s, "
        f"undershoot ratio {metrics['undershoot_ratio']:.2f}"
    )
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        columnspacing=1.0,
        handlelength=2.2,
    )
    return fig


def plot_demo(
    bold_hrf,
    mion_hrf,
    *,
    tr,
    bold_views,
    mion_views,
    run_index,
    kernel,
    plot_run=0,
    n_tr_plot=60,
    corrected_window=None,
):
    """Draw the impulse responses, movie responses, kernel, and transformed movie.

    Panels B and D show the same run: the mean across voxels of the first
    subject in each group, z-scored over the run. Panel B is the first
    ``n_tr_plot`` TRs. Panel D convolves that run's MION series with
    ``kernel`` and plots the result unscaled, so it is the kernel's actual
    prediction of BOLD. ``corrected_window`` is ``(start, stop)`` in seconds
    for panel D; omitted, panel D uses the same samples as panel B.
    """
    bold_hrf = np.asarray(bold_hrf, dtype=float).ravel()
    mion_hrf = np.asarray(mion_hrf, dtype=float).ravel()
    kernel = np.asarray(kernel, dtype=float).ravel()
    run_index = np.asarray(run_index)

    bold_mean = _run_mean(bold_views, run_index, plot_run)
    mion_mean = _run_mean(mion_views, run_index, plot_run)
    bold_z = _zscore(bold_mean)
    mion_z = _zscore(mion_mean)
    corrected_z = np.convolve(mion_mean, kernel)[: bold_mean.size]
    n_show = min(int(n_tr_plot), bold_z.size)
    movie_time = np.arange(n_show) * tr

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), constrained_layout=True)
    if corrected_window is None:
        axes[1, 1].sharex(axes[0, 1])
        axes[0, 1].tick_params(labelbottom=False)
    else:
        axes[0, 1].set_xlabel("Time (s)")

    impulse_time = np.arange(bold_hrf.size) * tr
    _style(axes[0, 0])
    axes[0, 0].axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
    axes[0, 0].plot(
        impulse_time,
        _peak_norm(bold_hrf),
        color=BOLD_COLOR,
        lw=2,
        label="BOLD",
    )
    axes[0, 0].plot(
        impulse_time,
        _peak_norm(mion_hrf),
        color=MION_COLOR,
        lw=2,
        label="MION",
    )
    axes[0, 0].set_title("A    Impulse response", loc="left")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Normalized response")
    axes[0, 0].legend(frameon=False, loc="upper right")

    _style(axes[0, 1])
    axes[0, 1].plot(movie_time, bold_z[:n_show], color=BOLD_COLOR, lw=1.8, label="BOLD")
    axes[0, 1].plot(movie_time, mion_z[:n_show], color=MION_COLOR, lw=1.8, label="MION")
    axes[0, 1].set_title("B    Simulated movie", loc="left")
    axes[0, 1].set_ylabel("Response (z)")
    axes[0, 1].legend(frameon=False, loc="upper right")

    kernel_time = np.arange(kernel.size) * tr
    _style(axes[1, 0])
    axes[1, 0].axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
    axes[1, 0].plot(kernel_time, kernel, color=MVTA_COLOR, lw=2)
    axes[1, 0].set_title("C    Least-squares kernel", loc="left")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Weight")

    if corrected_window is None:
        corrected_time = movie_time
        corrected_bold = bold_z[:n_show]
        corrected_mion = corrected_z[:n_show]
    else:
        start_s, stop_s = corrected_window
        start = int(round(float(start_s) / tr))
        stop = int(round(float(stop_s) / tr))
        if not 0 <= start < stop <= bold_z.size:
            raise ValueError(
                f"corrected_window {corrected_window} is outside the run "
                f"(0 to {bold_z.size * tr:g} s)."
            )
        corrected_time = np.arange(start, stop) * tr
        corrected_bold = bold_z[start:stop]
        corrected_mion = corrected_z[start:stop]
    _style(axes[1, 1])
    axes[1, 1].plot(corrected_time, corrected_bold, color=BOLD_COLOR, lw=1.8, label="BOLD")
    axes[1, 1].plot(
        corrected_time,
        corrected_mion,
        color=MVTA_COLOR,
        lw=1.8,
        ls="--",
        label="Transformed MION",
    )
    axes[1, 1].set_title("D    Transformed movie", loc="left")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Response (z)")
    axes[1, 1].legend(frameon=False, loc="upper right")
    return fig


def _run_mean(views, run_index, run):
    return np.asarray(views[0][run_index == run], dtype=float).mean(axis=1)


def _peak_norm(values):
    peak = np.max(np.abs(values))
    if peak == 0:
        return values
    return values / peak


def _zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=0)
    if std == 0:
        return values - values.mean()
    return (values - values.mean()) / std


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.margins(x=0.03)


DECODING_COLORS = {
    "double convolution": "#ff7f0e",
    "canonical double convolution": "#ff7f0e",
    "oracle double convolution": "#9467bd",
    "MVTA": "black",
    "no multiview": "#7f7f7f",
    "no alignment": MION_COLOR,
}


def plot_decoding(results, title="Cross-species stimulus epoch decoding"):
    """Bar plot of chance-corrected decoding accuracy.

    ``results`` maps a method label to one accuracy per held-out run. Bars are
    the mean across runs and dots are the individual runs.
    """
    labels = list(results)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    _style(ax)
    ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
    for position, label in enumerate(labels):
        values = np.asarray(results[label], dtype=float)
        color = DECODING_COLORS.get(label, "#4c78a8")
        ax.bar(position, values.mean(), color=color, alpha=0.85, width=0.65)
        spread = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else [0.0]
        ax.scatter(position + np.asarray(spread), values, color="white", edgecolor="black", s=22, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy above chance")
    ax.set_title(title)
    return fig


def plot_sed_methods(results, title="SED robustness across methods"):
    """Grouped decoding plot across SED configurations.

    ``results`` maps each SED-setting label to a mapping from temporal
    preparation method to one mean accuracy per simulation scenario. Bars are
    means across scenarios; dots are the individual scenario means.
    """
    sed_labels = list(results)
    if not sed_labels:
        raise ValueError("results must contain at least one SED configuration.")
    method_labels = list(results[sed_labels[0]])
    if not method_labels:
        raise ValueError("Each SED configuration must contain at least one method.")

    fig_width = max(8.8, 1.55 * len(sed_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), constrained_layout=True)
    _style(ax)
    ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
    group_positions = np.arange(len(sed_labels), dtype=float)
    width = 0.8 / len(method_labels)

    for method_index, method in enumerate(method_labels):
        positions = (
            group_positions
            - 0.4
            + width / 2
            + method_index * width
        )
        scenario_values = [
            np.asarray(results[sed_label][method], dtype=float)
            for sed_label in sed_labels
        ]
        means = [values.mean() for values in scenario_values]
        color = DECODING_COLORS.get(method, "#4c78a8")
        ax.bar(
            positions,
            means,
            width=width * 0.92,
            color=color,
            alpha=0.85,
            label=method,
        )
        for position, values in zip(positions, scenario_values):
            spread = (
                np.linspace(-width * 0.24, width * 0.24, values.size)
                if values.size > 1
                else [0.0]
            )
            ax.scatter(
                position + np.asarray(spread),
                values,
                color="white",
                edgecolor="black",
                s=18,
                zorder=3,
            )

    ax.set_xticks(group_positions)
    ax.set_xticklabels(sed_labels, rotation=0, ha="center")
    ax.set_ylabel("Accuracy above chance")
    ax.set_title(title)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.45,
    )
    return fig
