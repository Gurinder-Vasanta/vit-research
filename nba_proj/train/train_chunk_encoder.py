import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras

from models.chunk_encoder import ChunkEncoder

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CACHE_DIR = "./frame_cache_vit"
STORE_NAME = "train_val_frames"

CHUNK_SIZE = 12
EMB_DIM = 768
TRAIN_BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4

SEED = 42


# --------------------------------------------------
# LOAD CACHED DATA
# --------------------------------------------------
def load_frame_store(store_name=STORE_NAME):
    emb_path = os.path.join(CACHE_DIR, f"{store_name}_emb.dat")
    meta_path = os.path.join(CACHE_DIR, f"{store_name}_meta.npz")

    meta = np.load(meta_path)
    n_frames = int(meta["n_frames"])
    emb_dim = int(meta["emb_dim"])

    frame_emb_mm = np.memmap(
        emb_path,
        dtype="float32",
        mode="r",
        shape=(n_frames, emb_dim),
    )

    return frame_emb_mm


def load_chunk_index_arrays(store_name=STORE_NAME):
    path = os.path.join(CACHE_DIR, f"{store_name}_chunk_indices.npz")
    data = np.load(path)

    train_chunk_indices = data["train_chunk_indices"]
    train_labels = data["train_labels"].astype(np.float32)
    val_chunk_indices = data["val_chunk_indices"]
    val_labels = data["val_labels"].astype(np.float32)

    return train_chunk_indices, train_labels, val_chunk_indices, val_labels


# --------------------------------------------------
# DATA SEQUENCE
# --------------------------------------------------
class ChunkEmbeddingSequence(tf.keras.utils.Sequence):
    def __init__(self, frame_emb_mm, chunk_indices, labels, batch_size=64, shuffle=True):
        self.frame_emb_mm = frame_emb_mm
        self.chunk_indices = chunk_indices
        self.labels = labels.astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.labels))
        batch_ids = self.order[start:end]

        batch_chunk_indices = self.chunk_indices[batch_ids]      # (B, 12)
        batch_embs = self.frame_emb_mm[batch_chunk_indices]      # (B, 12, 768)
        batch_labels = self.labels[batch_ids]                    # (B,)

        return batch_embs.astype(np.float32), batch_labels.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


# --------------------------------------------------
# TRAINING UTILS
# --------------------------------------------------
def compute_accuracy(labels, logits):
    probs = tf.nn.sigmoid(logits)
    preds = tf.cast(probs >= 0.5, tf.float32)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


@tf.function
def train_step(chunk_encoder, optimizer, loss_fn, frame_embs, labels):
    with tf.GradientTape() as tape:
        _, class_logits = chunk_encoder(frame_embs, training=True)
        labels_f = tf.cast(labels, tf.float32)
        labels_f = tf.reshape(labels_f, (-1, 1))

        loss = loss_fn(labels_f, class_logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, chunk_encoder.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, chunk_encoder.trainable_variables))

    acc = compute_accuracy(labels_f, class_logits)
    return loss, acc


@tf.function
def val_step(chunk_encoder, loss_fn, frame_embs, labels):
    _, class_logits = chunk_encoder(frame_embs, training=False)
    labels_f = tf.cast(labels, tf.float32)
    labels_f = tf.reshape(labels_f, (-1, 1))

    loss = loss_fn(labels_f, class_logits)
    loss = tf.reduce_mean(loss)
    acc = compute_accuracy(labels_f, class_logits)
    return loss, acc


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    frame_emb_mm = load_frame_store(STORE_NAME)
    train_chunk_indices, train_labels, val_chunk_indices, val_labels = load_chunk_index_arrays(STORE_NAME)

    print("frame_emb_mm shape:", frame_emb_mm.shape)
    print("train_chunk_indices:", train_chunk_indices.shape)
    print("train_labels:", train_labels.shape)
    print("val_chunk_indices:", val_chunk_indices.shape)
    print("val_labels:", val_labels.shape)

    train_seq = ChunkEmbeddingSequence(
        frame_emb_mm,
        train_chunk_indices,
        train_labels,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_seq = ChunkEmbeddingSequence(
        frame_emb_mm,
        val_chunk_indices,
        val_labels,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    chunk_encoder = ChunkEncoder(
        hidden_size=EMB_DIM,
        num_layers=3,
        num_heads=8,
        max_frames=CHUNK_SIZE
    )

    dummy = tf.zeros((2, CHUNK_SIZE, EMB_DIM), dtype=tf.float32)
    _ = chunk_encoder(dummy, training=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR,
                                         weight_decay=1e-2)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    best_val_loss = float("inf")
    save_dir = "./chunk_encoder_ckpts_cached"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

        # ---------------- TRAIN ----------------
        train_losses = []
        train_accs = []
        t0 = time.perf_counter()

        for step in range(len(train_seq)):
            frame_embs_batch, labels_batch = train_seq[step]

            loss, acc = train_step(
                chunk_encoder,
                optimizer,
                loss_fn,
                tf.convert_to_tensor(frame_embs_batch),
                tf.convert_to_tensor(labels_batch),
            )

            train_losses.append(float(loss.numpy()))
            train_accs.append(float(acc.numpy()))

            print(
                f"[train] step={step} "
                f"loss={train_losses[-1]:.4f} "
                f"acc={train_accs[-1]:.4f}"
            )

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        train_time = time.perf_counter() - t0

        # ---------------- VAL ----------------
        val_losses = []
        val_accs = []

        for step in range(len(val_seq)):
            frame_embs_batch, labels_batch = val_seq[step]

            loss, acc = val_step(
                chunk_encoder,
                loss_fn,
                tf.convert_to_tensor(frame_embs_batch),
                tf.convert_to_tensor(labels_batch),
            )

            val_losses.append(float(loss.numpy()))
            val_accs.append(float(acc.numpy()))

        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        print(
            f"[epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={train_time:.2f}s"
        )

        latest_path = os.path.join(save_dir, "chunk_encoder_latest.weights.h5")
        chunk_encoder.save_weights(latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "chunk_encoder_best.weights.h5")
            chunk_encoder.save_weights(best_path)
            print(f"saved best weights to {best_path}")


if __name__ == "__main__":
    main()
