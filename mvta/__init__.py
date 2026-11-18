"""Multiview temporal alignment of simulated MION and BOLD movie data."""

from mvta.align import SRMSpace, disparity, fit_shared_filter, mean_cross_disparity
from mvta.decode import decode_methods, segment_matching
from mvta.fit import apply_kernel, cross_species_loss, learn_kernels, sse_kernel, sse_loss
from mvta.hrf import bold_hrf, hrf_shape_metrics, mion_hrf, variable_bold_hrf
from mvta.plot import (
    plot_decoding,
    plot_demo,
    plot_hrf_shapes,
    plot_neural,
    plot_sed_methods,
)
from mvta.simulate import MovieSimulation, simulate_movie

__all__ = [
    "MovieSimulation",
    "SRMSpace",
    "apply_kernel",
    "bold_hrf",
    "cross_species_loss",
    "decode_methods",
    "disparity",
    "learn_kernels",
    "fit_shared_filter",
    "hrf_shape_metrics",
    "mean_cross_disparity",
    "mion_hrf",
    "plot_decoding",
    "plot_demo",
    "plot_hrf_shapes",
    "plot_neural",
    "plot_sed_methods",
    "segment_matching",
    "simulate_movie",
    "sse_kernel",
    "sse_loss",
    "variable_bold_hrf",
]
