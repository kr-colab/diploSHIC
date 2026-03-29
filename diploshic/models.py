"""Model definitions for diploSHIC CNN architectures."""

from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, concatenate,
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
