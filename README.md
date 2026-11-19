# Multiview Temporal Alignment (MVTA)

Learn a convolutional kernel that temporally aligns MION time series with BOLD responses. The kernel is learned by minimizing cross-species differences in a shared representational space. That space is a deterministic shared response model (SRM; Chen et al. 2015), rotated and weighted toward BOLD–MION correlation. The notebooks in this repository demonstrate the method on simulated data.

## Requirements

- **Operating system.** Any OS that runs Python 3.11 or newer. Tested on Ubuntu 26.04.1 LTS.
- **Python.** 3.11 or newer. Tested on Python 3.14.7.
- **Python packages** (declared versions, then versions tested):
  - numpy >= 2 (tested 2.5.4)
  - scipy >= 1.11 (tested 1.18.1)
  - matplotlib >= 3.8 (tested 3.11.2)
  - nilearn >= 0.14 (tested 0.14.1)
  - jupyterlab >= 4 (tested 4.6.4)
  - ipykernel >= 6.29 (tested 7.4.0)
  - scikit-learn (pulled in by nilearn; tested 1.9.1)

Deterministic SRM is copied from nltools 0.5.1 (originally BrainIAK) into `mvta/srm.py` so this package does not depend on nltools or BrainIAK.

## Installation

Requires Python 3.11 or newer. Use a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Typical install time on a normal desktop computer: 2 minutes.

## Demo

All demo data are simulated; nothing is downloaded. Run the notebooks in this order:

1. `learn_hrf_transform_proof_of_principle.ipynb` shows that a MION-to-BOLD kernel exists: one noise-free neural signal is shared by both species. The kernel is learned by minimizing the squared error between BOLD and the convolved MION.
2. `learn_hrf_transform_mixed_signals.ipynb` adds species-specific and subject-specific signals. MVTA learns the kernel and is compared with double convolution, no multiview (temporal alignment on the mean timecourse across voxels), and no alignment (leave MION and BOLD unchanged). Each method is then evaluated with cross-species stimulus epoch decoding in a shared space.
3. `learn_hrf_transform_hrf_variability.ipynb` repeats the mixed-signals comparison while varying the BOLD HRF shape.

Example launch:

```bash
jupyter lab learn_hrf_transform_proof_of_principle.ipynb
```

Follow each notebook to simulate the data, learn the kernel, and generate the results.

Expected run time for the full demo on a normal desktop computer: about 5 minutes.

## Instructions for use

`mvta` takes already-extracted 2-D NumPy arrays shaped `(n_time, n_voxels)`, not NIfTI or CIFTI files. The number of time points must be the same across views; the number of voxels can differ. Both groups should have watched the same stimulus, sampled at the same TR. For each subject, runs are stacked vertically (along the time axis) in the same order, so `n_time` is the total number of time points across runs.

**1. Pass in the data.** Collect the BOLD subjects in one list and the MION subjects in another. `run_index` is a 1-D array of length `n_time` that labels the run of each row.

```python
from mvta import apply_kernel, learn_kernels

# bold_views: list of (n_time, n_voxels) arrays, one per BOLD subject
# mion_views: list of (n_time, n_voxels) arrays, one per MION subject
# run_index:  (n_time,) run labels shared by every view
```

Choose a kernel duration that is a multiple of the TR (the notebooks use 15 s). `n_components` is the SRM rank used while learning the kernel.

**2. Perform MVTA.** `learn_kernels` uses nested cross-validation. Each run is held out in turn (outer loop), and the kernel for that fold is scored with leave-one-run-out on the remaining runs (inner loop). On each inner fold it fits SRM, rotates and weights the scores toward BOLD–MION correlation, and measures between-species disparity. It returns one kernel per outer fold, shaped `(n_runs, kernel_length)`. Apply a kernel with `apply_kernel`, which convolves each run separately and z-scores the result:

```python
kernels = learn_kernels(
    bold_views,
    mion_views,
    run_index,
    kernel_length=kernel_length,
    objective="multiview",
    n_components=kernel_n_components,
)

# kernels[i] was learned with run i held out; apply one of those kernels
mion_aligned = [apply_kernel(view, kernels[0], run_index) for view in mion_views]
```

**3. Perform hyperalignment.** After the MION series have been temporally transformed, project the two groups into a shared space with a spatial hyperalignment method of your choice. Fit that model on training runs and apply it to held-out runs, using the same outer run split as the kernel. 

The package includes `decode_methods` for cross-species stimulus epoch decoding, which is how the notebooks evaluate temporal alignment after hyperalignment. That evaluation uses SRM plus the same correlation rotation, with a time-shift null to drop directions that are not shared above chance.

## Cite

If you use this code, please cite:

Braunlich, K., Duyck, M., Duffield, S., Behel, K., Baker, C. I., & Conway, B. R. (2026). *Translating functional brain activity between humans and monkeys*. bioRxiv. https://www.biorxiv.org/content/10.64898/2026.10.01.755966v1
