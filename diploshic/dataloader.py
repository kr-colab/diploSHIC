"""Data loading and domain adaptation batch generator for diploSHIC."""

import os

import numpy as np
import tensorflow as tf


FVEC_CLASSES = ["hard", "neut", "soft", "linkedSoft", "linkedHard"]
CLASS_INDICES = {name: i for i, name in enumerate(FVEC_CLASSES)}


def load_fvecs(directory, num_sub_wins=11):
    """Load the 5 fvec training/testing files from a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing hard.fvec, neut.fvec, soft.fvec,
        linkedSoft.fvec, linkedHard.fvec.
    num_sub_wins : int
        Number of sub-windows.

    Returns
    -------
    X : ndarray of shape (n_samples, nDims, num_sub_wins, 1)
    y : ndarray of shape (n_samples,) with integer class labels
    """
    if not directory.endswith("/"):
        directory += "/"

    arrays = []
    labels = []
    nDims = None
    for i, cls in enumerate(FVEC_CLASSES):
        data = np.loadtxt(directory + cls + ".fvec", skiprows=1)
        if nDims is None:
            nDims = int(data.shape[1] / num_sub_wins)
        reshaped = np.reshape(data, (data.shape[0], nDims, num_sub_wins))
        arrays.append(reshaped)
        labels.append(np.repeat(i, len(reshaped)))

    X = np.concatenate(arrays)
    y = np.concatenate(labels)
    X = X.reshape(X.shape[0], nDims, num_sub_wins, 1)
    return X, y


def load_empirical_fvec(filepath, num_sub_wins=11):
    """Load a single fvec file (empirical or simulated).

    Auto-detects VCF format (4 coordinate columns) vs sim format.

    Parameters
    ----------
    filepath : str
        Path to the .fvec file.
    num_sub_wins : int
        Number of sub-windows.

    Returns
    -------
    X : ndarray of shape (n_samples, nDims, num_sub_wins, 1)
    """
    import pandas as pd
    x_df = pd.read_table(filepath)
    # VCF format has chrom, classifiedWinStart, classifiedWinEnd, bigWinRange as first 4 cols
    if "chrom" in x_df.columns:
        data = x_df[list(x_df.columns)[4:]].to_numpy()
    else:
        data = x_df.to_numpy()
    nDims = int(data.shape[1] / num_sub_wins)
    X = data.reshape(data.shape[0], nDims, num_sub_wins, 1)
    return X


class DASequence(tf.keras.utils.Sequence):
    """Batch generator for domain adaptation training.

    Each batch contains:
    - batch_size source samples for the predictor (discriminator labels masked)
    - batch_size//2 source samples for the discriminator (predictor labels masked, domain=0)
    - batch_size//2 target samples for the discriminator (predictor labels masked, domain=1)

    Parameters
    ----------
    X_source : ndarray
        Source (simulated) feature data, shape (n, nDims, numSubWins, 1).
    Y_source : ndarray
        One-hot encoded source labels, shape (n, 5).
    X_target : ndarray
        Target (empirical) feature data, shape (m, nDims, numSubWins, 1).
    mean : ndarray
        Per-feature mean for normalization.
    std : ndarray
        Per-feature std for normalization.
    batch_size : int
        Number of source predictor samples per batch.
    shuffle : bool
        Whether to shuffle indices each epoch.
    """

    def __init__(self, X_source, Y_source, X_target, mean, std, batch_size=32, shuffle=True):
        self.X_source = X_source
        self.Y_source = Y_source
        self.X_target = X_target
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.src_indices = np.arange(len(X_source))
        self.tgt_indices = np.arange(len(X_target))
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.X_source) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.src_indices)
            np.random.shuffle(self.tgt_indices)

    def _normalize(self, X):
        std = self.std.copy()
        std[std == 0] = 1.0
        return (X - self.mean) / std

    def __getitem__(self, idx):
        bs = self.batch_size
        half_bs = max(1, bs // 2)

        # Source samples for predictor
        pred_start = (idx * bs) % len(self.src_indices)
        pred_idx = np.take(self.src_indices, np.arange(pred_start, pred_start + bs), mode="wrap")

        # Source samples for discriminator (domain=0)
        disc_src_start = ((idx * half_bs) + len(self.src_indices) // 2) % len(self.src_indices)
        disc_src_idx = np.take(self.src_indices, np.arange(disc_src_start, disc_src_start + half_bs), mode="wrap")

        # Target samples for discriminator (domain=1)
        disc_tgt_start = (idx * half_bs) % len(self.tgt_indices)
        disc_tgt_idx = np.take(self.tgt_indices, np.arange(disc_tgt_start, disc_tgt_start + half_bs), mode="wrap")

        # Build combined batch
        X_pred = self.X_source[pred_idx]
        X_disc_src = self.X_source[disc_src_idx]
        X_disc_tgt = self.X_target[disc_tgt_idx]
        X_batch = np.concatenate([X_pred, X_disc_src, X_disc_tgt], axis=0)
        X_batch = self._normalize(X_batch)

        n_pred = len(X_pred)
        n_disc_src = len(X_disc_src)
        n_disc_tgt = len(X_disc_tgt)
        total = n_pred + n_disc_src + n_disc_tgt

        # Predictor labels: real for pred samples, -1 for disc samples
        n_classes = self.Y_source.shape[1]
        Y_pred = np.full((total, n_classes), -1.0, dtype=np.float32)
        Y_pred[:n_pred] = self.Y_source[pred_idx]

        # Discriminator labels: -1 for pred samples, 0 for source disc, 1 for target disc
        Y_disc = np.full((total, 1), -1.0, dtype=np.float32)
        Y_disc[n_pred:n_pred + n_disc_src] = 0.0
        Y_disc[n_pred + n_disc_src:] = 1.0

        return X_batch.astype(np.float32), {"predictor": Y_pred, "discriminator": Y_disc}
