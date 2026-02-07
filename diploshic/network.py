"""Model construction and domain adaptation components for diploSHIC."""

import tensorflow as tf
from keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, Layer, MaxPooling2D, concatenate
)
from keras.models import Model


class GradReverse(Layer):
    """Gradient reversal layer.

    Forward pass: identity. Backward pass: negate gradients.
    Used to train domain-invariant features (Ganin et al., 2016).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -dy
            return x, grad
        return _reverse(x)

    def get_config(self):
        return super().get_config()


def masked_categorical_crossentropy(y_true, y_pred):
    """Categorical crossentropy that skips samples where all labels are -1."""
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.not_equal(y_true[:, 0], -1)
    mask = tf.cast(mask, y_pred.dtype)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    masked_loss = loss * mask
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_loss) / n_valid


def masked_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy that skips samples where label is -1."""
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.not_equal(y_true[:, 0], -1)
    mask = tf.cast(mask, y_pred.dtype)
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    masked_loss = loss * mask
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_loss) / n_valid


def masked_categorical_accuracy(y_true, y_pred):
    """Categorical accuracy that skips samples where all labels are -1."""
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.not_equal(y_true[:, 0], -1)
    mask = tf.cast(mask, y_pred.dtype)
    correct = tf.cast(
        tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)),
        y_pred.dtype,
    )
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(correct * mask) / n_valid


def masked_binary_accuracy(y_true, y_pred):
    """Binary accuracy that skips samples where label is -1."""
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.not_equal(y_true[:, 0], -1)
    mask = tf.cast(mask, y_pred.dtype)
    pred_labels = tf.cast(y_pred > 0.5, y_pred.dtype)
    correct = tf.cast(tf.equal(y_true, pred_labels), y_pred.dtype)
    correct = tf.reduce_mean(correct, axis=-1)
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(correct * mask) / n_valid


def _build_conv_trunk(model_in):
    """Build the shared 3-branch convolutional trunk."""
    # Branch 1: standard conv
    h = Conv2D(128, 3, activation="relu", padding="same", name="conv1_1")(model_in)
    h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=3, name="pool1", padding="same")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flaten1")(h)

    # Branch 2: dilated conv (rate 3)
    dh = Conv2D(128, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_1")(model_in)
    dh = Conv2D(64, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_2")(dh)
    dh = MaxPooling2D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflaten1")(dh)

    # Branch 3: dilated conv (rate 4)
    dh1 = Conv2D(128, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_1")(model_in)
    dh1 = Conv2D(64, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_2")(dh1)
    dh1 = MaxPooling2D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flaten1")(dh1)

    return concatenate([h, dh, dh1])


def construct_model(input_shape, domain_adaptation=False, da_weight=1.0, pred_weight=1.0):
    """Construct the diploSHIC CNN model.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (nDims, numSubWins, 1).
    domain_adaptation : bool
        If False, builds the standard single-output model.
        If True, adds a discriminator branch with gradient reversal.
    da_weight : float
        Loss weight for the discriminator branch (DA mode only).
    pred_weight : float
        Loss weight for the predictor branch (DA mode only).

    Returns
    -------
    keras.Model
        Compiled model.
    """
    model_in = Input(input_shape)
    trunk = _build_conv_trunk(model_in)

    if not domain_adaptation:
        # Standard model — exact same architecture as original inline code
        h = Dense(512, name="512dense", activation="relu")(trunk)
        h = Dropout(0.2, name="drop7")(h)
        h = Dense(128, name="last_dense", activation="relu")(h)
        h = Dropout(0.1, name="drop8")(h)
        output = Dense(5, name="out_dense", activation="softmax")(h)
        model = Model(inputs=[model_in], outputs=[output])
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    # Domain adaptation model — predictor + discriminator branches
    # Predictor branch
    p = Dense(512, name="pred_dense512", activation="relu")(trunk)
    p = Dropout(0.2, name="pred_drop1")(p)
    p = Dense(128, name="pred_dense128", activation="relu")(p)
    p = Dropout(0.1, name="pred_drop2")(p)
    predictor = Dense(5, name="predictor", activation="softmax")(p)

    # Discriminator branch (with gradient reversal)
    d = GradReverse(name="grad_reverse")(trunk)
    d = Dense(512, name="disc_dense512", activation="relu")(d)
    d = Dropout(0.2, name="disc_drop1")(d)
    d = Dense(128, name="disc_dense128", activation="relu")(d)
    d = Dropout(0.1, name="disc_drop2")(d)
    discriminator = Dense(1, name="discriminator", activation="sigmoid")(d)

    model = Model(inputs=[model_in], outputs=[predictor, discriminator])
    model.compile(
        loss={
            "predictor": masked_categorical_crossentropy,
            "discriminator": masked_binary_crossentropy,
        },
        loss_weights={
            "predictor": pred_weight,
            "discriminator": da_weight,
        },
        optimizer="adam",
        metrics={
            "predictor": [masked_categorical_accuracy],
            "discriminator": [masked_binary_accuracy],
        },
    )
    return model


def get_custom_objects():
    """Return dict of custom objects needed for loading DA models."""
    return {
        "GradReverse": GradReverse,
        "masked_categorical_crossentropy": masked_categorical_crossentropy,
        "masked_binary_crossentropy": masked_binary_crossentropy,
        "masked_categorical_accuracy": masked_categorical_accuracy,
        "masked_binary_accuracy": masked_binary_accuracy,
    }
