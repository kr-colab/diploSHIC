"""Tests for domain adaptation components."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from diploshic.network import (
    GradReverse,
    construct_model,
    get_custom_objects,
    masked_binary_accuracy,
    masked_binary_crossentropy,
    masked_categorical_accuracy,
    masked_categorical_crossentropy,
)
from diploshic.dataloader import DASequence, load_empirical_fvec, load_fvecs

TRAINING_DIR = str(Path(__file__).parent.parent / "diploshic" / "training")
TESTING_DIR = str(Path(__file__).parent.parent / "diploshic" / "testing")
EMPIRICAL_FVEC = str(Path(__file__).parent.parent / "testEmpirical.fvec")


# ── GradReverse ──────────────────────────────────────────────────────────


class TestGradReverse:
    def test_forward_is_identity(self):
        layer = GradReverse()
        x = tf.constant([1.0, 2.0, 3.0])
        out = layer(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_gradient_is_negated(self):
        layer = GradReverse()
        x = tf.Variable([1.0, 2.0, 3.0])
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(y)
        grad = tape.gradient(loss, x)
        # Forward: identity, so dloss/dy = 1 for each element
        # GRL negates gradients, so dloss/dx = -1 for each element
        np.testing.assert_array_equal(grad.numpy(), [-1.0, -1.0, -1.0])

    def test_serialization(self):
        layer = GradReverse(name="test_grl")
        config = layer.get_config()
        assert config["name"] == "test_grl"
        restored = GradReverse.from_config(config)
        assert restored.name == "test_grl"


# ── Masked losses ────────────────────────────────────────────────────────


class TestMaskedLosses:
    def test_masked_categorical_crossentropy_excludes_masked(self):
        y_true = tf.constant([[1, 0, 0], [0, 1, 0], [-1, -1, -1]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]], dtype=tf.float32)
        masked_loss = masked_categorical_crossentropy(y_true, y_pred).numpy()
        # Compute expected: standard loss on first 2 only
        std_loss = tf.keras.losses.categorical_crossentropy(y_true[:2], y_pred[:2]).numpy()
        expected = np.sum(std_loss) / 2.0
        np.testing.assert_allclose(masked_loss, expected, rtol=1e-5)

    def test_masked_categorical_crossentropy_no_mask(self):
        y_true = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype=tf.float32)
        masked = masked_categorical_crossentropy(y_true, y_pred).numpy()
        std = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred)).numpy()
        np.testing.assert_allclose(masked, std, rtol=1e-5)

    def test_masked_binary_crossentropy_excludes_masked(self):
        y_true = tf.constant([[0.0], [1.0], [-1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9], [0.5]], dtype=tf.float32)
        masked_loss = masked_binary_crossentropy(y_true, y_pred).numpy()
        std_loss = tf.keras.losses.binary_crossentropy(y_true[:2], y_pred[:2]).numpy()
        expected = np.sum(std_loss) / 2.0
        np.testing.assert_allclose(masked_loss, expected, rtol=1e-5)

    def test_masked_categorical_accuracy_excludes_masked(self):
        y_true = tf.constant([[1, 0, 0], [0, 1, 0], [-1, -1, -1]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]], dtype=tf.float32)
        acc = masked_categorical_accuracy(y_true, y_pred).numpy()
        assert acc == 1.0  # both unmasked predictions are correct

    def test_masked_binary_accuracy_excludes_masked(self):
        y_true = tf.constant([[0.0], [1.0], [-1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9], [0.5]], dtype=tf.float32)
        acc = masked_binary_accuracy(y_true, y_pred).numpy()
        assert acc == 1.0  # both unmasked predictions are correct


# ── construct_model ──────────────────────────────────────────────────────


class TestConstructModel:
    def test_standard_model_single_output(self):
        model = construct_model((12, 11, 1), domain_adaptation=False)
        assert len(model.outputs) == 1
        assert model.output_shape == (None, 5)

    def test_standard_model_layer_names(self):
        model = construct_model((12, 11, 1), domain_adaptation=False)
        names = [l.name for l in model.layers]
        for expected in ["conv1_1", "conv1_2", "dconv1_1", "dconv1_2", "dconv4_1", "dconv4_2",
                         "512dense", "last_dense", "out_dense"]:
            assert expected in names, f"Missing layer: {expected}"

    def test_da_model_two_outputs(self):
        model = construct_model((12, 11, 1), domain_adaptation=True)
        assert len(model.outputs) == 2

    def test_da_model_output_names(self):
        model = construct_model((12, 11, 1), domain_adaptation=True)
        output_names = list(model.output_names)
        assert "predictor" in output_names
        assert "discriminator" in output_names

    def test_da_model_output_shapes(self):
        model = construct_model((12, 11, 1), domain_adaptation=True)
        # predictor: (None, 5), discriminator: (None, 1)
        shapes = [o.shape for o in model.outputs]
        assert shapes[0][-1] == 5
        assert shapes[1][-1] == 1

    def test_da_model_has_grad_reverse(self):
        model = construct_model((12, 11, 1), domain_adaptation=True)
        grl_layers = [l for l in model.layers if isinstance(l, GradReverse)]
        assert len(grl_layers) == 1


# ── load_fvecs ───────────────────────────────────────────────────────────


class TestLoadFvecs:
    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(TRAINING_DIR, "hard.fvec")),
        reason="Training data not available",
    )
    def test_load_training_data(self):
        X, y = load_fvecs(TRAINING_DIR)
        assert X.ndim == 4
        assert X.shape[1] > 0  # nDims
        assert X.shape[2] == 11  # numSubWins
        assert X.shape[3] == 1  # channel
        assert len(np.unique(y)) == 5
        assert len(X) == len(y)

    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(TESTING_DIR, "hard.fvec")),
        reason="Testing data not available",
    )
    def test_load_testing_data(self):
        X, y = load_fvecs(TESTING_DIR)
        assert X.ndim == 4
        assert len(X) == len(y)


# ── load_empirical_fvec ─────────────────────────────────────────────────


class TestLoadEmpiricalFvec:
    @pytest.mark.skipif(
        not os.path.isfile(EMPIRICAL_FVEC),
        reason="testEmpirical.fvec not available",
    )
    def test_load_vcf_format(self):
        X = load_empirical_fvec(EMPIRICAL_FVEC)
        assert X.ndim == 4
        assert X.shape[2] == 11
        assert X.shape[3] == 1


# ── DASequence ───────────────────────────────────────────────────────────


class TestDASequence:
    def setup_method(self):
        np.random.seed(42)
        self.n_src = 100
        self.n_tgt = 50
        self.nDims = 12
        self.numSubWins = 11
        self.X_src = np.random.randn(self.n_src, self.nDims, self.numSubWins, 1).astype(np.float32)
        self.Y_src = tf.keras.utils.to_categorical(np.random.randint(0, 5, self.n_src), 5)
        self.X_tgt = np.random.randn(self.n_tgt, self.nDims, self.numSubWins, 1).astype(np.float32)
        self.mean = np.zeros((self.nDims, self.numSubWins, 1))
        self.std = np.ones((self.nDims, self.numSubWins, 1))

    def test_batch_shapes(self):
        seq = DASequence(self.X_src, self.Y_src, self.X_tgt, self.mean, self.std, batch_size=32)
        X_batch, Y_dict = seq[0]
        # batch_size + batch_size//2 + batch_size//2 = 32 + 16 + 16 = 64
        assert X_batch.shape[0] == 64
        assert Y_dict["predictor"].shape == (64, 5)
        assert Y_dict["discriminator"].shape == (64, 1)

    def test_masking_pattern(self):
        seq = DASequence(self.X_src, self.Y_src, self.X_tgt, self.mean, self.std, batch_size=32)
        X_batch, Y_dict = seq[0]
        bs = 32
        half_bs = 16

        # First bs samples: predictor labels are real (not -1), discriminator masked
        assert np.all(Y_dict["predictor"][:bs, 0] != -1)
        assert np.all(Y_dict["discriminator"][:bs] == -1)

        # Next half_bs: predictor masked, discriminator = 0 (source)
        assert np.all(Y_dict["predictor"][bs:bs + half_bs, 0] == -1)
        assert np.all(Y_dict["discriminator"][bs:bs + half_bs] == 0)

        # Last half_bs: predictor masked, discriminator = 1 (target)
        assert np.all(Y_dict["predictor"][bs + half_bs:, 0] == -1)
        assert np.all(Y_dict["discriminator"][bs + half_bs:] == 1)

    def test_len(self):
        seq = DASequence(self.X_src, self.Y_src, self.X_tgt, self.mean, self.std, batch_size=32)
        assert len(seq) == 100 // 32  # 3


# ── Model save/load round-trip ──────────────────────────────────────────


class TestModelSaveLoad:
    def test_standard_model_round_trip(self):
        from keras.models import model_from_json
        model = construct_model((12, 11, 1), domain_adaptation=False)
        model_json = model.to_json()
        loaded = model_from_json(model_json)
        assert len(loaded.outputs) == 1

    def test_da_model_round_trip(self):
        from keras.models import model_from_json
        model = construct_model((12, 11, 1), domain_adaptation=True)
        model_json = model.to_json()
        loaded = model_from_json(model_json, custom_objects=get_custom_objects())
        assert len(loaded.outputs) == 2

    def test_da_model_weights_round_trip(self):
        from keras.models import model_from_json
        model = construct_model((12, 11, 1), domain_adaptation=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "model.json")
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            with open(json_path, "w") as f:
                f.write(model.to_json())
            model.save_weights(weights_path)
            with open(json_path) as f:
                loaded = model_from_json(f.read(), custom_objects=get_custom_objects())
            loaded.load_weights(weights_path)
            # Verify predictions match
            x = np.random.randn(2, 12, 11, 1).astype(np.float32)
            orig = model.predict(x, verbose=0)
            rest = loaded.predict(x, verbose=0)
            np.testing.assert_allclose(orig[0], rest[0], rtol=1e-5)
            np.testing.assert_allclose(orig[1], rest[1], rtol=1e-5)


# ── Integration test ────────────────────────────────────────────────────


class TestDAIntegration:
    @pytest.mark.skipif(
        not (os.path.isfile(os.path.join(TRAINING_DIR, "hard.fvec"))
             and os.path.isfile(EMPIRICAL_FVEC)),
        reason="Training or empirical data not available",
    )
    def test_train_and_predict(self):
        from keras.models import model_from_json

        X_src, y_src = load_fvecs(TRAINING_DIR)
        Y_src = tf.keras.utils.to_categorical(y_src, 5)
        X_tgt = load_empirical_fvec(EMPIRICAL_FVEC)

        # Compute normalization stats
        X_all = np.concatenate([X_src, X_tgt], axis=0)
        mean = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)

        model = construct_model(X_src.shape[1:], domain_adaptation=True)
        train_seq = DASequence(X_src, Y_src, X_tgt, mean, std, batch_size=32)

        model.fit(train_seq, epochs=1, verbose=0)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "model.json")
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            with open(json_path, "w") as f:
                f.write(model.to_json())
            model.save_weights(weights_path)

            with open(json_path) as f:
                loaded = model_from_json(f.read(), custom_objects=get_custom_objects())
            loaded.load_weights(weights_path)

        # Predict with loaded model
        std_safe = std.copy()
        std_safe[std_safe == 0] = 1.0
        X_normed = (X_tgt - mean) / std_safe
        preds = loaded.predict(X_normed, verbose=0)
        assert isinstance(preds, list)
        assert len(preds) == 2
        pred_classes = preds[0]
        assert pred_classes.shape == (len(X_tgt), 5)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(pred_classes.sum(axis=1), 1.0, atol=1e-5)
