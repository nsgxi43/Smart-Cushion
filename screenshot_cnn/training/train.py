#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 - train the lightweight 2D CNN.

Architecture (project brief section 5):

    Input (72, 64, 1)
      -> Conv2D(8, 3x3, ReLU)  -> MaxPool2D
      -> Conv2D(16, 3x3, ReLU) -> MaxPool2D
      -> Conv2D(32, 3x3, ReLU)
      -> GlobalAveragePooling2D
      -> Dense(32, ReLU) -> Dropout
      -> Dense(7, Softmax)

Two variants are available. ``base`` is exactly the brief's stack. ``bn`` adds
BatchNormalization after each convolution - it folds into the preceding conv at
inference time, so it costs nothing in the deployed model. Both are trained and
the one with the lower validation loss is kept, so the choice is made on
evidence rather than assertion.

Augmentation is applied to the training split only: small translations, slight
intensity scaling and mild sensor-style noise. Left/Right are never mirrored
horizontally and Forward/Backward are never mirrored vertically. Rotation is
omitted deliberately - a cushion mat has a fixed orientation under the sitter,
so rotating the pressure field is not a physically plausible variation.

Usage:
    python training/train.py                 # trains both variants, keeps the better
    python training/train.py --arch base     # trains one variant only
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import CLASS_NAMES, INPUT_SHAPE, NUM_CLASSES, ROOT, set_seeds

PROCESSED = os.path.join(ROOT, "processed_dataset")
MODELS = os.path.join(ROOT, "models")
EVAL = os.path.join(ROOT, "evaluation")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_splits():
    X = np.load(os.path.join(PROCESSED, "X.npy"))
    y = np.load(os.path.join(PROCESSED, "y.npy"))
    s = np.load(os.path.join(PROCESSED, "split.npy"))
    out = {}
    for name in ("train", "val", "test"):
        m = s == name
        out[name] = (X[m], y[m])
    return out


def build_model(arch: str = "bn", hidden: int = 32, dropout: float = 0.3):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    use_bn = arch == "bn"
    inp = keras.Input(shape=INPUT_SHAPE, name="pressure_map")
    x = inp
    for i, f in enumerate((8, 16, 32)):
        x = layers.Conv2D(f, 3, padding="same", use_bias=not use_bn,
                          name=f"conv{i+1}")(x)
        if use_bn:
            x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.ReLU(name=f"relu{i+1}")(x)
        if i < 2:
            x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(hidden, activation="relu", name="dense_hidden")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="posture")(x)
    return keras.Model(inp, out, name=f"smart_cushion_cnn_{arch}")


def make_train_ds(X, y, batch_size: int, seed: int, shift: int = 4,
                  intensity: float = 0.10, noise: float = 0.02):
    """Training pipeline with augmentation. No flips - see module docstring."""
    import tensorflow as tf

    H, W, C = INPUT_SHAPE

    def augment(img, lab):
        # small translation: pad with the edge value, then random crop back
        p = tf.pad(img, [[shift, shift], [shift, shift], [0, 0]], mode="SYMMETRIC")
        img = tf.image.random_crop(p, (H, W, C))
        # slight intensity scaling
        img = img * tf.random.uniform([], 1.0 - intensity, 1.0 + intensity)
        # mild sensor-style noise
        img = img + tf.random.normal(tf.shape(img), 0.0, noise)
        return tf.clip_by_value(img, 0.0, 1.0), lab

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    # Sequential, deterministic map: the augmentation ops draw from the global
    # TF RNG, so a parallel map would make runs irreproducible. The dataset is
    # small enough that this costs nothing measurable.
    ds = ds.map(augment, num_parallel_calls=None, deterministic=True)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).with_options(opts)


def train_one(arch, data, args, verbose=0):
    import tensorflow as tf
    from tensorflow import keras

    set_seeds(args.seed)
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]

    model = build_model(arch, hidden=args.hidden, dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = os.path.join(MODELS, f"_ckpt_{arch}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True, verbose=verbose),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=args.lr_patience,
                                          min_lr=1e-5, verbose=0),
    ]

    # counts differ slightly between classes after the session-aware split
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float64)
    class_weight = {i: float(len(ytr) / (NUM_CLASSES * c)) if c else 0.0
                    for i, c in enumerate(counts)}

    ds = make_train_ds(Xtr, ytr, args.batch_size, args.seed,
                       shift=args.shift, intensity=args.intensity, noise=args.noise)
    hist = model.fit(ds, validation_data=(Xva, yva), epochs=args.epochs,
                     callbacks=callbacks, class_weight=class_weight, verbose=verbose)

    val_loss, val_acc = model.evaluate(Xva, yva, verbose=0)
    return model, hist.history, val_loss, val_acc


def plot_history(histories: dict, best_arch: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 8), squeeze=False)
    for col, (arch, h) in enumerate(sorted(histories.items())):
        tag = f"{arch}  (selected)" if arch == best_arch else arch
        ep = range(1, len(h["loss"]) + 1)
        ax = axes[0][col]
        ax.plot(ep, h["accuracy"], label="train", lw=1.6)
        ax.plot(ep, h["val_accuracy"], label="validation", lw=1.6)
        ax.set_title(f"{tag} - accuracy"); ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3); ax.legend(loc="lower right")
        ax = axes[1][col]
        ax.plot(ep, h["loss"], label="train", lw=1.6)
        ax.plot(ep, h["val_loss"], label="validation", lw=1.6)
        best = int(np.argmin(h["val_loss"]))
        ax.axvline(best + 1, color="k", ls="--", lw=1,
                   label=f"best val_loss (ep {best+1})")
        ax.set_title(f"{tag} - loss"); ax.set_xlabel("epoch")
        ax.set_ylabel("loss"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Smart Cushion CNN - training history", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["base", "bn", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--lr-patience", type=int, default=20)
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--intensity", type=float, default=0.03,
                    help="intensity-scaling range; 0.03 selected by a validation-loss "
                         "sweep over {0.00, 0.03, 0.06, 0.10} - see README")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(EVAL, exist_ok=True)
    set_seeds(args.seed)

    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")

    data = load_splits()
    for k, (X, y) in data.items():
        print(f"  {k:<6} {X.shape[0]:3d} images  per class "
              f"{np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    archs = ["base", "bn"] if args.arch == "both" else [args.arch]
    results, histories, models = {}, {}, {}
    for arch in archs:
        print(f"\n=== training variant '{arch}' ===")
        model, hist, vl, va = train_one(arch, data, args, verbose=args.verbose)
        epochs_run = len(hist["loss"])
        print(f"  epochs run {epochs_run}  best val_loss {vl:.4f}  val_acc {va:.4f}")
        results[arch] = dict(val_loss=float(vl), val_accuracy=float(va),
                             epochs_run=epochs_run,
                             params=int(model.count_params()))
        histories[arch] = hist
        models[arch] = model

    best_arch = min(results, key=lambda a: results[a]["val_loss"])
    best = models[best_arch]
    print(f"\nselected variant: '{best_arch}' (lowest validation loss)")
    for a, r in results.items():
        print(f"  {a:<5} val_loss {r['val_loss']:.4f}  val_acc {r['val_accuracy']:.4f} "
              f"  params {r['params']:,}")

    best.summary()
    out = os.path.join(MODELS, "best_model.keras")
    best.save(out)
    size_kb = os.path.getsize(out) / 1024
    print(f"\nsaved {out}  ({size_kb:.1f} KB on disk, "
          f"{best.count_params():,} parameters, "
          f"{best.count_params()*4/1024:.1f} KB as float32 weights)")

    plot_history(histories, best_arch, os.path.join(EVAL, "training_history.png"))
    print(f"saved {os.path.join(EVAL, 'training_history.png')}")

    with open(os.path.join(EVAL, "training_history.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(selected=best_arch, results=results,
                       args=vars(args), histories=histories), fh, indent=2)

    for arch in archs:
        p = os.path.join(MODELS, f"_ckpt_{arch}.keras")
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
