#!/usr/bin/env python
"""Train a diploSHIC model and benchmark its performance.

Usage:
    python scripts/benchmark.py                          # run baseline
    python scripts/benchmark.py --model daf --compare    # run DAF model and compare
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

CLASS_NAMES = ["hard", "neutral", "soft", "linkedSoft", "linkedHard"]
FVEC_CLASSES = [("hard", 0), ("neut", 1), ("soft", 2), ("linkedSoft", 3), ("linkedHard", 4)]
SEED = 42

N_DAF_BINS = 20
N_DIST_FEATURES = 4


def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_fvec_data(data_dir, n_subwins=11):
    """Load .fvec files from a directory, return X array and y labels."""
    arrays = []
    labels = []
    for class_name, class_id in FVEC_CLASSES:
        fpath = os.path.join(data_dir, f"{class_name}.fvec")
        data = np.loadtxt(fpath, skiprows=1)
        n_dims = data.shape[1] // n_subwins
        data = data.reshape(data.shape[0], n_dims, n_subwins, 1)
        arrays.append(data)
        labels.append(np.full(data.shape[0], class_id))
    return np.concatenate(arrays), np.concatenate(labels)


def load_daf_fvec_data(data_dir, n_subwins=11):
    """Load .daf.fvec files, split into DAF histogram and distance arrays."""
    daf_arrays = []
    dist_arrays = []
    labels = []
    for class_name, class_id in FVEC_CLASSES:
        fpath = os.path.join(data_dir, f"{class_name}.daf.fvec")
        data = np.loadtxt(fpath, skiprows=1)
        n_total = data.shape[1]
        n_daf_cols = N_DAF_BINS * n_subwins
        n_dist_cols = N_DIST_FEATURES * n_subwins
        assert n_total == n_daf_cols + n_dist_cols, f"Expected {n_daf_cols + n_dist_cols} cols, got {n_total}"

        daf_flat = data[:, :n_daf_cols]
        dist_flat = data[:, n_daf_cols:]

        daf_arrays.append(daf_flat.reshape(data.shape[0], N_DAF_BINS, n_subwins, 1))
        dist_arrays.append(dist_flat.reshape(data.shape[0], N_DIST_FEATURES, n_subwins, 1))
        labels.append(np.full(data.shape[0], class_id))

    return (np.concatenate(daf_arrays), np.concatenate(dist_arrays), np.concatenate(labels))


def build_model(model_name, n_stats, n_subwins):
    """Build and compile a model by name."""
    from diploshic.models import build_baseline_model, build_daf_model

    if model_name == "baseline":
        model = build_baseline_model(n_stats, n_subwins)
    elif model_name == "daf":
        model = build_daf_model(n_stats, n_subwins)
    else:
        sys.exit(f"Unknown model: {model_name}")

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def _normalize(X_train, X_val, X_test):
    """Featurewise center and std normalization. Returns normalized arrays and params."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def train_and_evaluate(model, train_inputs, Y_train, val_inputs, Y_val,
                       test_inputs, Y_test, output_dir, epochs=100):
    """Train model with early stopping, evaluate, and save results.

    train_inputs/val_inputs/test_inputs can be a single array (baseline)
    or a list of arrays (multi-input models like daf).
    """
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "model.weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=5, verbose=1, mode="auto"),
        ModelCheckpoint(weights_path, monitor="val_accuracy", verbose=1,
                        save_best_only=True, save_weights_only=True, mode="auto"),
    ]

    # Normalize each input independently
    if isinstance(train_inputs, list):
        norm_train, norm_val, norm_test = [], [], []
        for tr, va, te in zip(train_inputs, val_inputs, test_inputs):
            tr_n, va_n, te_n, _, _ = _normalize(tr, va, te)
            norm_train.append(tr_n)
            norm_val.append(va_n)
            norm_test.append(te_n)
    else:
        tr_n, va_n, te_n, mean, std = _normalize(train_inputs, val_inputs, test_inputs)
        norm_train, norm_val, norm_test = tr_n, va_n, te_n
        np.save(os.path.join(output_dir, "train_mean.npy"), mean)
        np.save(os.path.join(output_dir, "train_std.npy"), std)

    start = time.time()
    history = model.fit(
        norm_train, Y_train,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(norm_val, Y_val),
    )
    train_time = time.time() - start

    # Load best weights
    model.load_weights(weights_path)

    # Evaluate
    score = model.evaluate(norm_test, Y_test, verbose=0)
    y_pred = np.argmax(model.predict(norm_test, verbose=0), axis=1)
    y_true = np.argmax(Y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        per_class_acc[name] = float((y_pred[mask] == i).mean()) if mask.sum() > 0 else 0.0

    results = {
        "model_name": model.name,
        "overall_loss": float(score[0]),
        "overall_accuracy": float(score[1]),
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "train_time_seconds": train_time,
        "epochs_run": len(history.history["loss"]),
        "seed": SEED,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    with open(os.path.join(output_dir, "model.json"), "w") as f:
        f.write(model.to_json())

    print(f"\n{'=' * 60}")
    print(f"Model: {model.name}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print(f"Overall loss:     {results['overall_loss']:.4f}")
    print(f"Training time:    {train_time:.1f}s ({results['epochs_run']} epochs)")
    print(f"\nPer-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name:12s}: {acc:.4f}")
    print(f"\nConfusion matrix:")
    print(f"  {'':12s} " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i, j]:8d}" for j in range(5))
        print(f"  {name:12s} {row}")
    print(f"\nResults saved to {output_dir}")

    return results


def compare_to_baseline(results, baseline_dir):
    """Print comparison of current results to saved baseline."""
    baseline_path = os.path.join(baseline_dir, "results.json")
    if not os.path.exists(baseline_path):
        print("\nNo baseline results found for comparison.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'=' * 60}")
    print("Comparison to baseline:")
    print(f"  Overall accuracy: {results['overall_accuracy']:.4f} vs {baseline['overall_accuracy']:.4f} "
          f"(delta: {results['overall_accuracy'] - baseline['overall_accuracy']:+.4f})")
    print(f"\n  Per-class accuracy deltas:")
    for name in CLASS_NAMES:
        curr = results["per_class_accuracy"].get(name, 0)
        base = baseline["per_class_accuracy"].get(name, 0)
        print(f"    {name:12s}: {curr:.4f} vs {base:.4f} (delta: {curr - base:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a diploSHIC model")
    parser.add_argument("--model", default="baseline", help="Model name (default: baseline)")
    parser.add_argument("--train-dir", default=None, help="Training fvec directory")
    parser.add_argument("--test-dir", default=None, help="Testing fvec directory")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--compare", action="store_true", help="Compare results to saved baseline")
    parser.add_argument("--n-subwins", type=int, default=11, help="Number of sub-windows")
    args = parser.parse_args()

    set_seeds()

    if args.train_dir is None:
        args.train_dir = os.path.join(REPO_ROOT, "diploshic", "training")
    if args.test_dir is None:
        args.test_dir = os.path.join(REPO_ROOT, "diploshic", "testing")

    benchmarks_dir = os.path.join(REPO_ROOT, "benchmarks")
    output_dir = os.path.join(benchmarks_dir, args.model)

    if args.model == "baseline" and os.path.exists(os.path.join(output_dir, "results.json")):
        print("Baseline results already exist. To re-run, delete benchmarks/baseline/ first.")
        return

    # Load summary stats (used by all models)
    print(f"Loading training stats from {args.train_dir}")
    X_stats_train, y_train = load_fvec_data(args.train_dir, args.n_subwins)
    print(f"Loading test stats from {args.test_dir}")
    X_stats_test, y_test = load_fvec_data(args.test_dir, args.n_subwins)

    n_stats = X_stats_train.shape[1]

    # Split test into validation and test (50/50), same split for all inputs
    indices = np.arange(len(y_test))
    idx_val, idx_test = train_test_split(indices, test_size=0.5, random_state=SEED)

    Y_train = tf.keras.utils.to_categorical(y_train, 5)
    Y_val = tf.keras.utils.to_categorical(y_test[idx_val], 5)
    Y_test = tf.keras.utils.to_categorical(y_test[idx_test], 5)

    if args.model == "baseline":
        train_inputs = X_stats_train
        val_inputs = X_stats_test[idx_val]
        test_inputs = X_stats_test[idx_test]
    elif args.model == "daf":
        print(f"Loading DAF features from {args.train_dir}")
        daf_train, dist_train, _ = load_daf_fvec_data(args.train_dir, args.n_subwins)
        print(f"Loading DAF features from {args.test_dir}")
        daf_test, dist_test, _ = load_daf_fvec_data(args.test_dir, args.n_subwins)

        train_inputs = [X_stats_train, daf_train, dist_train]
        val_inputs = [X_stats_test[idx_val], daf_test[idx_val], dist_test[idx_val]]
        test_inputs = [X_stats_test[idx_test], daf_test[idx_test], dist_test[idx_test]]
    else:
        sys.exit(f"Unknown model: {args.model}")

    n_samples = X_stats_train.shape[0]
    print(f"\nDataset sizes: train={n_samples}, val={len(idx_val)}, test={len(idx_test)}")

    model = build_model(args.model, n_stats, args.n_subwins)
    model.summary()

    results = train_and_evaluate(
        model, train_inputs, Y_train, val_inputs, Y_val, test_inputs, Y_test,
        output_dir, epochs=args.epochs
    )

    if args.compare:
        compare_to_baseline(results, os.path.join(benchmarks_dir, "baseline"))


if __name__ == "__main__":
    main()
