"""
YAMNet Feature Extraction Pipeline
====================================
Full end-to-end demo covering:
  - Synthetic audio generation (4 sound classes)
  - Feature extraction via frozen YAMNet
  - Path A: Custom classifier (Keras dense head)
  - Path B: Embedding analysis (UMAP + cosine similarity)

Requirements:
    pip install tensorflow tensorflow-hub numpy matplotlib scikit-learn umap-learn tqdm

Run:
    python yamnet_pipeline.py
"""

# ─────────────────────────────────────────────
# 0. Imports & Config
# ─────────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# Try importing umap; fall back to t-SNE if unavailable
try:
    import umap
    REDUCER = "umap"
except ImportError:
    from sklearn.manifold import TSNE
    REDUCER = "tsne"
    print("umap-learn not found — using t-SNE instead. Install with: pip install umap-learn")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SR = 16_000          # YAMNet required sample rate
CLIP_DURATION = 2    # seconds per clip
N_PER_CLASS = 80     # synthetic clips per class (increase for better results)
EPOCHS = 30
BATCH_SIZE = 32

CLASSES = ["sine_tone", "drum_hit", "white_noise", "chirp"]
COLORS  = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5"]   # purple, teal, coral, blue


# ─────────────────────────────────────────────
# 1. Synthetic Audio Generator
# ─────────────────────────────────────────────
def make_sine(duration: float, sr: int) -> np.ndarray:
    """Pure sine wave at a random musical frequency (A3–A5)."""
    freq = np.random.uniform(220, 880)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    return (wave * 0.8).astype(np.float32)


def make_drum(duration: float, sr: int) -> np.ndarray:
    """Short exponential-decay noise burst — approximates a kick/snare."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    hit_pos = np.random.randint(0, n // 4)
    noise = np.random.randn(n).astype(np.float32)
    envelope = np.exp(-15 * (t - t[hit_pos]).clip(min=0))
    return (noise * envelope * 0.9).astype(np.float32)


def make_noise(duration: float, sr: int) -> np.ndarray:
    """Band-limited white noise."""
    n = int(sr * duration)
    noise = np.random.randn(n).astype(np.float32)
    return (noise * 0.5).astype(np.float32)


def make_chirp(duration: float, sr: int) -> np.ndarray:
    """Linear frequency sweep from f0 → f1."""
    n = int(sr * duration)
    f0 = np.random.uniform(200, 600)
    f1 = np.random.uniform(1000, 4000)
    t = np.linspace(0, duration, n, endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    return (np.sin(phase) * 0.8).astype(np.float32)


GENERATORS = {
    "sine_tone":  make_sine,
    "drum_hit":   make_drum,
    "white_noise": make_noise,
    "chirp":      make_chirp,
}


def build_dataset(n_per_class: int, sr: int, duration: float):
    waveforms, labels = [], []
    for cls in CLASSES:
        gen = GENERATORS[cls]
        for _ in range(n_per_class):
            waveforms.append(gen(duration, sr))
            labels.append(cls)
    return waveforms, labels


# ─────────────────────────────────────────────
# 2. Load YAMNet & Extract Embeddings
# ─────────────────────────────────────────────
def load_yamnet():
    print("Loading YAMNet from TF Hub (downloads ~10 MB on first run)…")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet loaded ✓")
    return model


def extract_embedding(yamnet_model, waveform: np.ndarray) -> np.ndarray:
    """
    Run waveform through YAMNet and return the mean-pooled 1024-dim embedding.
    YAMNet processes ~0.96 s windows; for a 2-s clip we get ~2 frames and average.
    """
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform_tf)
    # embeddings: (N_frames, 1024) → pool to (1024,)
    return tf.reduce_mean(embeddings, axis=0).numpy()


def extract_all_embeddings(yamnet_model, waveforms, labels):
    print(f"Extracting embeddings for {len(waveforms)} clips…")
    X, y = [], []
    for wf, lbl in tqdm(zip(waveforms, labels), total=len(waveforms)):
        emb = extract_embedding(yamnet_model, wf)
        X.append(emb)
        y.append(lbl)
    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────────────────────
# 3A. Path A — Custom Classifier
# ─────────────────────────────────────────────
def build_classifier(n_classes: int) -> tf.keras.Model:
    """
    Tiny dense head on top of the 1024-dim YAMNet embedding.
    Two hidden layers with dropout — plenty for a 4-class problem.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="yamnet_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(X, y_str):
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = build_classifier(len(le.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
    ]

    print("\nTraining classifier…")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n── Classification Report ──────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, history, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# 3B. Path B — Embedding Analysis
# ─────────────────────────────────────────────
def reduce_embeddings(X: np.ndarray) -> np.ndarray:
    if REDUCER == "umap":
        print("Projecting with UMAP…")
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    else:
        print("Projecting with t-SNE…")
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=30)
    return reducer.fit_transform(X)


def cosine_similarity_matrix(X: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute mean per-class embedding and return the NxN cosine similarity matrix.
    """
    class_means = {}
    for cls in CLASSES:
        mask = labels == cls
        class_means[cls] = X[mask].mean(axis=0)

    n = len(CLASSES)
    sim = np.zeros((n, n))
    for i, a in enumerate(CLASSES):
        for j, b in enumerate(CLASSES):
            va, vb = class_means[a], class_means[b]
            sim[i, j] = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return sim


# ─────────────────────────────────────────────
# 4. Plotting
# ─────────────────────────────────────────────
def plot_all(history, X_test, y_test, y_pred, le, X, y_str):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Training curves ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history["accuracy"],     label="train acc", color=COLORS[0])
    ax1.plot(history.history["val_accuracy"], label="val acc",   color=COLORS[1], linestyle="--")
    ax1.plot(history.history["loss"],         label="train loss", color=COLORS[2], alpha=0.7)
    ax1.plot(history.history["val_loss"],     label="val loss",   color=COLORS[3], linestyle="--", alpha=0.7)
    ax1.set_title("Training curves", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Confusion matrix ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Confusion matrix (test set)", fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # ── Panel 3: Per-class accuracy bar ───────
    ax3 = fig.add_subplot(gs[0, 2])
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = ax3.bar(le.classes_, per_class_acc, color=COLORS[:len(le.classes_)], width=0.6)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Per-class accuracy", fontsize=12)
    ax3.set_ylabel("Accuracy")
    for bar, val in zip(bars, per_class_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # ── Panel 4: UMAP / t-SNE scatter ─────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    proj = reduce_embeddings(X)
    for cls, color in zip(CLASSES, COLORS):
        mask = y_str == cls
        ax4.scatter(proj[mask, 0], proj[mask, 1], label=cls, color=color,
                    alpha=0.7, s=30, linewidths=0)
    ax4.set_title(f"Embedding space ({REDUCER.upper()})", fontsize=12)
    ax4.legend(fontsize=9, markerscale=1.5)
    ax4.set_xlabel("Dim 1"); ax4.set_ylabel("Dim 2")
    ax4.grid(True, alpha=0.2)

    # ── Panel 5: Cosine similarity heatmap ────
    ax5 = fig.add_subplot(gs[1, 2])
    sim = cosine_similarity_matrix(X, y_str)
    im = ax5.imshow(sim, vmin=0.5, vmax=1.0, cmap="Blues")
    ax5.set_xticks(range(len(CLASSES))); ax5.set_yticks(range(len(CLASSES)))
    ax5.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax5.set_yticklabels(CLASSES, fontsize=9)
    ax5.set_title("Cosine similarity\n(mean class embeddings)", fontsize=12)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax5.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="navy" if sim[i,j] > 0.75 else "gray")

    fig.suptitle("YAMNet Feature Extraction Pipeline — Full Results", fontsize=14, y=1.01)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamnet_results.png")

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()


# ─────────────────────────────────────────────
# 5. Inference helper (use after training)
# ─────────────────────────────────────────────
def predict_new_sound(yamnet_model, classifier, le, waveform: np.ndarray):
    """
    Given a raw waveform (mono float32 @16kHz), return the predicted class
    and probability distribution.

    Example usage:
        wf = make_chirp(2.0, SR)
        label, probs = predict_new_sound(yamnet, clf, le, wf)
        print(label, probs)
    """
    emb = extract_embedding(yamnet_model, waveform)
    probs = classifier.predict(emb[np.newaxis], verbose=0)[0]
    pred_idx = np.argmax(probs)
    return le.classes_[pred_idx], dict(zip(le.classes_, probs.tolist()))


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # --- Build synthetic dataset ---
    print("Generating synthetic audio clips…")
    waveforms, labels = build_dataset(N_PER_CLASS, SR, CLIP_DURATION)
    print(f"  {len(waveforms)} clips across {len(CLASSES)} classes\n")

    # --- Load YAMNet ---
    yamnet = load_yamnet()

    # --- Extract embeddings (one-time pass through frozen YAMNet) ---
    X, y_str = extract_all_embeddings(yamnet, waveforms, labels)
    print(f"\nEmbedding matrix shape: {X.shape}")   # (320, 1024)

    # --- Path A: train classifier ---
    clf, le, history, X_test, y_test, y_pred = train_classifier(X, y_str)

    # --- Path B + full plot ---
    plot_all(history, X_test, y_test, y_pred, le, X, y_str)

    # --- Demo: predict one new clip ---
    print("\n── Inference demo ─────────────────────────")
    test_wf = make_chirp(CLIP_DURATION, SR)
    pred_label, pred_probs = predict_new_sound(yamnet, clf, le, test_wf)
    print(f"  New chirp → predicted: '{pred_label}'")
    for cls, p in sorted(pred_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:<12} {bar:<30} {p:.3f}")

    print("\nDone! ✓  See yamnet_results.png for the full visualization.")