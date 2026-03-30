"""Model definitions for diploSHIC CNN architectures."""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    GlobalAveragePooling1D, BatchNormalization, Activation, Reshape,
    concatenate, Permute,
)


def build_baseline_model(n_stats=12, n_subwins=11):
    """Build the original diploSHIC 3-branch CNN.

    Architecture: three parallel Conv2D branches with different dilation rates,
    concatenated before dense layers. 5-class softmax output.

    Parameters
    ----------
    n_stats : int
        Number of summary statistics (rows in the feature image).
    n_subwins : int
        Number of sub-windows (columns in the feature image).

    Returns
    -------
    keras.Model
    """
    model_in = Input(shape=(n_stats, n_subwins, 1), name="stats_input")

    # Branch A: standard convolution
    h = Conv2D(128, 3, activation="relu", padding="same", name="conv1_1")(model_in)
    h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=3, padding="same", name="pool1")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flatten1")(h)

    # Branch B: dilated convolution (rate=3)
    dh = Conv2D(128, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_1")(model_in)
    dh = Conv2D(64, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_2")(dh)
    dh = MaxPooling2D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflatten1")(dh)

    # Branch C: dilated convolution (rate=4)
    dh1 = Conv2D(128, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_1")(model_in)
    dh1 = Conv2D(64, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_2")(dh1)
    dh1 = MaxPooling2D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flatten1")(dh1)

    # Merge and classify
    merged = concatenate([h, dh, dh1])
    merged = Dense(512, activation="relu", name="dense_512")(merged)
    merged = Dropout(0.2, name="drop7")(merged)
    merged = Dense(128, activation="relu", name="dense_128")(merged)
    merged = Dropout(0.1, name="drop8")(merged)
    output = Dense(5, activation="softmax", name="out_dense")(merged)

    return Model(inputs=[model_in], outputs=[output], name="diploSHIC_baseline")


def build_daf_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=10):
    """Build the extended diploSHIC CNN with DAF histogram and distance branches.

    The summary-stat branches are identical to the baseline model. Two additional
    branches process DAF histogram and inter-SNP distance features.

    Parameters
    ----------
    n_stats : int
        Number of summary statistics.
    n_subwins : int
        Number of sub-windows.
    n_daf_bins : int
        Number of DAF histogram bins per sub-window.
    n_dist_features : int
        Number of distance summary features per sub-window.

    Returns
    -------
    keras.Model
    """
    # --- Input 1: Summary statistics (same as baseline) ---
    stats_in = Input(shape=(n_stats, n_subwins, 1), name="stats_input")

    # Branch A: standard convolution
    h = Conv2D(128, 3, activation="relu", padding="same", name="conv1_1")(stats_in)
    h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=3, padding="same", name="pool1")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flatten1")(h)

    # Branch B: dilated convolution (rate=3)
    dh = Conv2D(128, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_1")(stats_in)
    dh = Conv2D(64, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_2")(dh)
    dh = MaxPooling2D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflatten1")(dh)

    # Branch C: dilated convolution (rate=4)
    dh1 = Conv2D(128, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_1")(stats_in)
    dh1 = Conv2D(64, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_2")(dh1)
    dh1 = MaxPooling2D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flatten1")(dh1)

    # --- Input 2: DAF histograms ---
    daf_in = Input(shape=(n_daf_bins, n_subwins, 1), name="daf_input")
    d = Conv2D(64, 3, activation="relu", padding="same", name="daf_conv1")(daf_in)
    d = Conv2D(32, 3, activation="relu", padding="same", name="daf_conv2")(d)
    d = MaxPooling2D(pool_size=2, name="daf_pool")(d)
    d = Dropout(0.15, name="daf_drop")(d)
    d = Flatten(name="daf_flatten")(d)

    # --- Input 3: Distance features ---
    dist_in = Input(shape=(n_dist_features, n_subwins, 1), name="dist_input")
    e = Conv2D(32, 2, activation="relu", padding="same", name="dist_conv1")(dist_in)
    e = Flatten(name="dist_flatten")(e)
    e = Dense(32, activation="relu", name="dist_dense")(e)

    # --- Merge all branches ---
    merged = concatenate([h, dh, dh1, d, e])
    merged = Dense(512, activation="relu", name="dense_512")(merged)
    merged = Dropout(0.2, name="drop7")(merged)
    merged = Dense(128, activation="relu", name="dense_128")(merged)
    merged = Dropout(0.1, name="drop8")(merged)
    output = Dense(5, activation="softmax", name="out_dense")(merged)

    return Model(inputs=[stats_in, daf_in, dist_in], outputs=[output], name="diploSHIC_daf")


def build_fused1d_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=10):
    """Build a 1D CNN with early fusion of all feature types.

    All features are stacked per sub-window into a single channel vector,
    then processed with 1D convolutions along the sub-window (genomic position)
    axis. This correctly treats sub-windows as a sequence with spatial locality
    and different feature types as channels at each position.

    Parameters
    ----------
    n_stats : int
        Number of summary statistics.
    n_subwins : int
        Number of sub-windows.
    n_daf_bins : int
        Number of DAF histogram bins per sub-window.
    n_dist_features : int
        Number of distance summary features per sub-window.

    Returns
    -------
    keras.Model
    """
    n_channels = n_stats + n_daf_bins + n_dist_features  # 36

    # Three inputs matching the existing data pipeline
    stats_in = Input(shape=(n_stats, n_subwins, 1), name="stats_input")
    daf_in = Input(shape=(n_daf_bins, n_subwins, 1), name="daf_input")
    dist_in = Input(shape=(n_dist_features, n_subwins, 1), name="dist_input")

    # Reshape each from (batch, features, subwins, 1) to (batch, subwins, features)
    # by stripping the channel dim then transposing
    stats_r = Reshape((n_stats, n_subwins), name="stats_squeeze")(stats_in)
    stats_r = Permute((2, 1), name="stats_transpose")(stats_r)  # (batch, 11, 12)

    daf_r = Reshape((n_daf_bins, n_subwins), name="daf_squeeze")(daf_in)
    daf_r = Permute((2, 1), name="daf_transpose")(daf_r)  # (batch, 11, 20)

    dist_r = Reshape((n_dist_features, n_subwins), name="dist_squeeze")(dist_in)
    dist_r = Permute((2, 1), name="dist_transpose")(dist_r)  # (batch, 11, 4)

    # Early fusion: concatenate all features at each sub-window position
    x = concatenate([stats_r, daf_r, dist_r], axis=-1, name="fuse")  # (batch, 11, 36)

    # Conv1D blocks along sub-window axis with batch normalization
    # Dilation schedule [1, 1, 3] gives receptive field = 11,
    # exactly covering the full sub-window sequence
    x = Conv1D(64, 3, padding="same", name="conv1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)

    x = Conv1D(64, 3, padding="same", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)

    x = Conv1D(128, 3, padding="same", dilation_rate=3, name="conv3_dil3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Activation("relu", name="relu3")(x)

    # Global average pooling — collapses sub-window axis without parameter explosion
    x = GlobalAveragePooling1D(name="gap")(x)

    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="bn_dense")(x)
    x = Activation("relu", name="relu_dense")(x)
    x = Dropout(0.3, name="drop1")(x)

    output = Dense(5, activation="softmax", name="out_dense")(x)

    return Model(inputs=[stats_in, daf_in, dist_in], outputs=[output], name="diploSHIC_fused1d")


def _conv_bn_relu(x, filters, kernel_size, dilation_rate=1, name_prefix=""):
    """Conv1D → BatchNorm → ReLU block."""
    x = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               name=f"{name_prefix}_conv")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def _fuse_inputs(stats_in, daf_in, dist_in, n_stats, n_daf_bins, n_dist_features, n_subwins):
    """Reshape and concatenate the three inputs into (batch, n_subwins, n_channels)."""
    stats_r = Permute((2, 1))(Reshape((n_stats, n_subwins))(stats_in))
    daf_r = Permute((2, 1))(Reshape((n_daf_bins, n_subwins))(daf_in))
    dist_r = Permute((2, 1))(Reshape((n_dist_features, n_subwins))(dist_in))
    return concatenate([stats_r, daf_r, dist_r], axis=-1, name="fuse")


def build_multiscale1d_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=10):
    """Build a multi-scale 1D CNN with parallel branches at different receptive fields.

    Three parallel branches process the fused feature sequence with different
    dilation schedules, each capturing patterns at a different spatial scale:
      - Local branch:  [1, 1]     RF=5   (adjacent sub-window patterns)
      - Medium branch: [1, 3]     RF=7   (center-vs-near-flank contrast)
      - Full branch:   [1, 1, 3]  RF=11  (full sequence context)

    Branches are concatenated after GlobalAveragePooling, giving the dense
    head multiple views of the same data at different resolutions.

    Parameters
    ----------
    n_stats : int
        Number of summary statistics.
    n_subwins : int
        Number of sub-windows.
    n_daf_bins : int
        Number of DAF histogram bins per sub-window.
    n_dist_features : int
        Number of distance summary features per sub-window.

    Returns
    -------
    keras.Model
    """
    stats_in = Input(shape=(n_stats, n_subwins, 1), name="stats_input")
    daf_in = Input(shape=(n_daf_bins, n_subwins, 1), name="daf_input")
    dist_in = Input(shape=(n_dist_features, n_subwins, 1), name="dist_input")

    x = _fuse_inputs(stats_in, daf_in, dist_in, n_stats, n_daf_bins, n_dist_features, n_subwins)

    # Branch A: local context (RF=5)
    a = _conv_bn_relu(x, 48, 3, dilation_rate=1, name_prefix="a1")
    a = _conv_bn_relu(a, 48, 3, dilation_rate=1, name_prefix="a2")
    a = GlobalAveragePooling1D(name="gap_a")(a)

    # Branch B: medium context (RF=7)
    b = _conv_bn_relu(x, 48, 3, dilation_rate=1, name_prefix="b1")
    b = _conv_bn_relu(b, 48, 3, dilation_rate=3, name_prefix="b2")
    b = GlobalAveragePooling1D(name="gap_b")(b)

    # Branch C: full context (RF=11)
    c = _conv_bn_relu(x, 48, 3, dilation_rate=1, name_prefix="c1")
    c = _conv_bn_relu(c, 48, 3, dilation_rate=1, name_prefix="c2")
    c = _conv_bn_relu(c, 48, 3, dilation_rate=3, name_prefix="c3")
    c = GlobalAveragePooling1D(name="gap_c")(c)

    merged = concatenate([a, b, c], name="merge_scales")

    merged = Dense(128, name="dense_128")(merged)
    merged = BatchNormalization(name="bn_dense")(merged)
    merged = Activation("relu", name="relu_dense")(merged)
    merged = Dropout(0.3, name="drop1")(merged)

    output = Dense(5, activation="softmax", name="out_dense")(merged)

    return Model(inputs=[stats_in, daf_in, dist_in], outputs=[output], name="diploSHIC_multiscale1d")


def _make_norm_layer(name, use_bn=True, groups=16):
    """Create a normalization layer based on the use_bn setting.

    use_bn=True:  BatchNormalization (stores running stats — good ID, bad OOD)
    use_bn=False: no normalization
    use_bn='group': GroupNormalization (per-sample, no stored stats — good for OOD)
    """
    if use_bn is True:
        return BatchNormalization(name=name)
    elif use_bn == "group":
        from keras.layers import GroupNormalization
        return GroupNormalization(groups=groups, name=name)
    else:
        return Activation("linear", name=name)  # identity passthrough


def _res_conv_block(x, filters, kernel_size, dilation_rate=1, name_prefix="", use_bn=True):
    """Residual Conv1D block: Conv→[Norm]→ReLU + skip connection.

    If input channels differ from filters, a 1x1 projection is applied
    to the skip path.
    """
    from keras.layers import Add
    shortcut = x
    h = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               name=f"{name_prefix}_conv")(x)
    h = _make_norm_layer(f"{name_prefix}_bn", use_bn)(h)

    # Project shortcut if channel dimensions don't match
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same", name=f"{name_prefix}_proj")(shortcut)
        shortcut = _make_norm_layer(f"{name_prefix}_proj_bn", use_bn)(shortcut)

    h = Add(name=f"{name_prefix}_add")([h, shortcut])
    h = Activation("relu", name=f"{name_prefix}_relu")(h)
    return h


class _SliceMean(tf.keras.layers.Layer):
    """Extract and average specific positions along axis 1."""
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, x):
        gathered = tf.gather(x, self.indices, axis=1)
        return tf.reduce_mean(gathered, axis=1)

    def get_config(self):
        return {**super().get_config(), "indices": self.indices}


class _SliceIndex(tf.keras.layers.Layer):
    """Extract a single position along axis 1."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index, :]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


class _InstanceNorm1D(tf.keras.layers.Layer):
    """Instance normalization along the spatial (sub-window) axis.

    For each sample and each feature channel independently, normalizes
    across the sub-window positions to zero mean and unit variance.
    This strips demography-dependent absolute feature magnitudes while
    preserving the spatial contrast patterns that indicate selection.
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):
        # x shape: (batch, n_subwins, n_channels)
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.epsilon)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


def _zone_pool(h, center_idx, near_indices, far_indices, name_prefix=""):
    """Extract concentric zone features from a sequence tensor.

    Returns center, near-flank mean, far-flank mean, and global max.
    """
    from keras.layers import GlobalMaxPooling1D

    center = _SliceIndex(center_idx, name=f"{name_prefix}_center")(h)
    near = _SliceMean(near_indices, name=f"{name_prefix}_near")(h)
    far = _SliceMean(far_indices, name=f"{name_prefix}_far")(h)
    gmp = GlobalMaxPooling1D(name=f"{name_prefix}_gmp")(h)
    return center, near, far, gmp


def build_multiscale_res1d_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=10, use_bn=True):
    """Build a multi-scale residual 1D CNN with concentric zone pooling.

    Sequential residual blocks with increasing dilation rates, tapped
    at each depth. At each tap, four concentric zone pools extract
    the spatial structure of the sweep signal:

      - Center:    sub-window 5 (the classified window)
      - Near-flank: mean of sub-windows 3,4,6,7 (immediate neighbors)
      - Far-flank:  mean of sub-windows 0,1,2,8,9,10 (distant regions)
      - GMP:       global max across all sub-windows (peak signal)

    The zone design encodes the radial symmetry of sweep effects:
    sweeps peak at center, close-linked peaks in near-flank,
    far-linked peaks in far-flank, neutral is uniform. The symmetric
    pooling (left and right combined per zone) is compatible with
    sub-window reversal augmentation.

    Parameters
    ----------
    n_stats : int
        Number of summary statistics.
    n_subwins : int
        Number of sub-windows.
    n_daf_bins : int
        Number of DAF histogram bins per sub-window.
    n_dist_features : int
        Number of distance summary features per sub-window.

    Returns
    -------
    keras.Model
    """
    center_idx = n_subwins // 2  # 5 for n_subwins=11
    # Near: 2 positions on each side of center
    near_indices = [center_idx - 2, center_idx - 1, center_idx + 1, center_idx + 2]
    # Far: everything else (3 positions on each side)
    far_indices = [i for i in range(n_subwins) if i != center_idx and i not in near_indices]

    stats_in = Input(shape=(n_stats, n_subwins, 1), name="stats_input")
    daf_in = Input(shape=(n_daf_bins, n_subwins, 1), name="daf_input")
    dist_in = Input(shape=(n_dist_features, n_subwins, 1), name="dist_input")

    x = _fuse_inputs(stats_in, daf_in, dist_in, n_stats, n_daf_bins, n_dist_features, n_subwins)

    # Instance normalization: per-sample, per-channel normalization across sub-windows.
    # Strips absolute demographic signal (e.g., "Anopheles has high diversity")
    # while preserving spatial contrast patterns (e.g., "center has lower diversity
    # than flanks") that are consistent across demographic models.
    x = _InstanceNorm1D(name="instance_norm")(x)

    # Block 1: local (RF=3), with projection 36→64
    h = _res_conv_block(x, 64, 3, dilation_rate=1, name_prefix="res1", use_bn=use_bn)
    ctr1, near1, far1, gmp1 = _zone_pool(h, center_idx, near_indices, far_indices, "t1")

    # Block 2: local (RF=5), residual within 64→64
    h = _res_conv_block(h, 64, 3, dilation_rate=1, name_prefix="res2", use_bn=use_bn)
    ctr2, near2, far2, gmp2 = _zone_pool(h, center_idx, near_indices, far_indices, "t2")

    # Block 3: full context (RF=11), residual within 64→64
    h = _res_conv_block(h, 64, 3, dilation_rate=3, name_prefix="res3", use_bn=use_bn)
    ctr3, near3, far3, gmp3 = _zone_pool(h, center_idx, near_indices, far_indices, "t3")

    # 3 taps × 4 zone pools × 64 channels = 768 features
    merged = concatenate([
        ctr1, near1, far1, gmp1,
        ctr2, near2, far2, gmp2,
        ctr3, near3, far3, gmp3,
    ], name="merge_zones")

    merged = Dense(256, name="dense_256")(merged)
    merged = _make_norm_layer("bn_dense1", use_bn)(merged)
    merged = Activation("relu", name="relu_dense1")(merged)
    merged = Dropout(0.3, name="drop1")(merged)

    merged = Dense(128, name="dense_128")(merged)
    merged = _make_norm_layer("bn_dense2", use_bn)(merged)
    merged = Activation("relu", name="relu_dense2")(merged)
    merged = Dropout(0.2, name="drop2")(merged)

    output = Dense(5, activation="softmax", name="out_dense")(merged)

    return Model(inputs=[stats_in, daf_in, dist_in], outputs=[output], name="diploSHIC_multiscale_res1d")
