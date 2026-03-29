"""Model definitions for diploSHIC CNN architectures."""

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


def build_daf_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=4):
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


def build_fused1d_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=4):
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


def build_multiscale1d_model(n_stats=12, n_subwins=11, n_daf_bins=20, n_dist_features=4):
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
