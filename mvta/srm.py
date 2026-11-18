"""Deterministic Shared Response Model (Chen et al. 2015).

Copied from nltools 0.5.1 (``nltools.external.srm.DetSRM``), which vendors
BrainIAK's implementation, so ``mvta`` does not depend on nltools or BrainIAK.

The algorithm is unchanged: random orthonormal maps, then block coordinate
descent with a Procrustes update for each subject's ``W_i``.

Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J., & Ramadge,
P. J. (2015). A reduced-dimension fMRI shared response model. In Advances
in Neural Information Processing Systems (pp. 460-468).

Copyright 2016 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np


def _init_w_transforms(data, features, random_states):
    """Initialize the mappings (Wi) with random orthogonal matrices.

    ``data`` is a list of arrays shaped ``(voxels_i, samples)``.
    """
    w = []
    voxels = np.empty(len(data), dtype=int)
    for subject in range(len(data)):
        voxels[subject] = data[subject].shape[0]
        rnd_matrix = random_states[subject].random_sample(
            (voxels[subject], features)
        )
        q, _r = np.linalg.qr(rnd_matrix)
        w.append(q)
    return w, voxels


class DetSRM:
    """Deterministic Shared Response Model.

    Factorizes multi-subject data as a shared response ``S`` and an orthogonal
    transform ``W_i`` per subject: ``X_i ≈ W_i S``.

    Inputs are voxels × time, matching nltools / BrainIAK.
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0):
        self.n_iter = int(n_iter)
        self.features = int(features)
        self.rand_seed = rand_seed

    def fit(self, X, y=None):
        if len(X) <= 1:
            raise ValueError(
                f"There are not enough subjects ({len(X)}) to train the model."
            )
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                f"{self.features} features."
            )
        number_trs = X[0].shape[1]
        for subject in range(len(X)):
            if not np.all(np.isfinite(X[subject])):
                raise ValueError("SRM inputs must be finite.")
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects.")
        self.w_, self.s_ = self._srm(X)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, "w_"):
            raise RuntimeError("Call fit before transform.")
        if len(X) != len(self.w_):
            raise ValueError(
                "The number of subjects does not match the one in the model."
            )
        return [self.w_[subject].T.dot(X[subject]) for subject in range(len(X))]

    def _compute_shared_response(self, data, w):
        shared = np.zeros((w[0].shape[1], data[0].shape[1]))
        for subject in range(len(w)):
            shared = shared + w[subject].T.dot(data[subject])
        shared /= len(w)
        return shared

    def _srm(self, data):
        subjects = len(data)
        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = [
            np.random.RandomState(
                self.random_state_.randint(2**32 - 1, dtype=np.int64)
            )
            for _ in range(len(data))
        ]
        w, _voxels = _init_w_transforms(data, self.features, random_states)
        shared_response = self._compute_shared_response(data, w)
        for _iteration in range(self.n_iter):
            for subject in range(subjects):
                a_subject = data[subject].dot(shared_response.T)
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.001)
                u_subject, _s, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False
                )
                w[subject] = u_subject.dot(v_subject)
            shared_response = self._compute_shared_response(data, w)
        return w, shared_response
