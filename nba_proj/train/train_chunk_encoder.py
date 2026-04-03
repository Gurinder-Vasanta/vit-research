import os
import pickle
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
STORE_NAME = "train_val_frames_chunk8_stride2"

CHUNK_SIZE = 8
CHUNK_STRIDE = 2   # informational here; actual overlap is determined upstream
EMB_DIM = 768

TRAIN_BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0

SEED = 42

# SAVE_DIR = "./chunk_encoder_ckpts_cached_overlap_s8"
SAVE_DIR = "./chunk_encoder_ckpts_chunk8_stride2"


# --------------------------------------------------
# LOAD CACHED DATA
# --------------------------------------------------
def load_frame_store(store_name=STORE_NAME):
    emb_path = os.path.join(CACHE_DIR, f"{store_name}_emb.dat")
    meta_path = os.path.join(CACHE_DIR, f"{store_name}_meta.npz")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing frame embedding memmap: {emb_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing frame metadata file: {meta_path}")

    meta = np.load(meta_path)
    n_frames = int(meta["n_frames"])
    emb_dim = int(meta["emb_dim"])

    if emb_dim != EMB_DIM:
        raise ValueError(f"Expected EMB_DIM={EMB_DIM}, but cached frame store has emb_dim={emb_dim}")

    frame_emb_mm = np.memmap(
        emb_path,
        dtype="float32",
        mode="r",
        shape=(n_frames, emb_dim),
    )
    return frame_emb_mm

def compute_conditioned_separation(
    chunk_encoder,
    frame_emb_mm,
    chunk_indices,
    labels,
    meta,
    max_samples=2000,
    max_time_gap=0.08,
    require_diff_video=True,
):
    # sample subset
    idx = np.random.choice(len(labels), size=min(max_samples, len(labels)), replace=False)

    X = frame_emb_mm[chunk_indices[idx]].astype(np.float32)
    y = labels[idx]
    side = meta["side"][idx]
    vid = meta["vid"][idx]
    t = meta["t_center"][idx]

    # encode
    Z, _ = chunk_encoder(tf.convert_to_tensor(X), training=False)
    Z = Z.numpy()

    # normalize for cosine
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    pos_sims = []
    neg_sims = []

    n = len(Z)
    for i in range(n):
        for j in range(i + 1, n):

            # same side only
            if side[i] != side[j]:
                continue

            # temporal locality
            if abs(t[i] - t[j]) > max_time_gap:
                continue

            # avoid trivial same-video matches
            if require_diff_video and vid[i] == vid[j]:
                continue

            sim = float(np.dot(Z[i], Z[j]))

            if y[i] == y[j]:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        print("[cond sep] not enough valid pairs")
        return None

    pos_sim = np.mean(pos_sims)
    neg_sim = np.mean(neg_sims)
    gap = pos_sim - neg_sim

    print(f"[cond sep] pos={pos_sim} neg={neg_sim} gap={gap} "
          f"(pairs: {len(pos_sims)} pos / {len(neg_sims)} neg)")

    return gap

def load_chunk_metadata_arrays(store_name=STORE_NAME):
    # path = os.path.join(CACHE_DIR, f"{store_name}_chunk_meta.npz")
    path = './frame_cache_vit/train_val_frames_chunk8_stride2_chunk_meta.npz'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunk metadata file: {path}")

    data = np.load(path, allow_pickle=True)

    train_meta = {
        "vid": data["train_vid"],
        "clip": data["train_clip"],
        "side": data["train_side"],
        "t_center": data["train_t_center"].astype(np.float32),
        "t_width": data["train_t_width"].astype(np.float32),
    }

    val_meta = {
        "vid": data["val_vid"],
        "clip": data["val_clip"],
        "side": data["val_side"],
        "t_center": data["val_t_center"].astype(np.float32),
        "t_width": data["val_t_width"].astype(np.float32),
    }

    return train_meta, val_meta

def load_chunk_index_arrays(store_name=STORE_NAME):
    path = os.path.join(CACHE_DIR, f"{store_name}_chunk_indices.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunk index file: {path}")

    data = np.load(path)

    train_chunk_indices = data["train_chunk_indices"]
    train_labels = data["train_labels"].astype(np.float32)
    val_chunk_indices = data["val_chunk_indices"]
    val_labels = data["val_labels"].astype(np.float32)

    return train_chunk_indices, train_labels, val_chunk_indices, val_labels


def validate_chunk_arrays(train_chunk_indices, val_chunk_indices):
    if train_chunk_indices.ndim != 2:
        raise ValueError(f"train_chunk_indices must be 2D, got shape {train_chunk_indices.shape}")
    if val_chunk_indices.ndim != 2:
        raise ValueError(f"val_chunk_indices must be 2D, got shape {val_chunk_indices.shape}")

    train_chunk_len = train_chunk_indices.shape[1]
    val_chunk_len = val_chunk_indices.shape[1]

    if train_chunk_len != CHUNK_SIZE:
        raise ValueError(
            f"train_chunk_indices chunk length = {train_chunk_len}, expected CHUNK_SIZE = {CHUNK_SIZE}"
        )
    if val_chunk_len != CHUNK_SIZE:
        raise ValueError(
            f"val_chunk_indices chunk length = {val_chunk_len}, expected CHUNK_SIZE = {CHUNK_SIZE}"
        )

def collect_val_stats(chunk_encoder, val_seq):
    all_probs = []
    all_labels = []

    for step in range(len(val_seq)):
        frame_embs_batch, labels_batch = val_seq[step]
        _, logits = chunk_encoder(
            tf.convert_to_tensor(frame_embs_batch),
            training=False
        )
        probs = tf.nn.sigmoid(logits).numpy().reshape(-1)
        labels = labels_batch.reshape(-1)

        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds = (all_probs >= 0.5).astype(np.float32)

    tp = np.sum((preds == 1) & (all_labels == 1))
    tn = np.sum((preds == 0) & (all_labels == 0))
    fp = np.sum((preds == 1) & (all_labels == 0))
    fn = np.sum((preds == 0) & (all_labels == 1))

    print(f"[val stats] mean_prob={all_probs.mean():.4f}")
    print(f"[val stats] pred_pos_rate={preds.mean():.4f}")
    print(f"[val stats] true_pos_rate={all_labels.mean():.4f}")
    print(f"[val stats] TP={tp} TN={tn} FP={fp} FN={fn}")

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

        batch_chunk_indices = self.chunk_indices[batch_ids]   # (B, CHUNK_SIZE)
        batch_embs = self.frame_emb_mm[batch_chunk_indices]   # (B, CHUNK_SIZE, EMB_DIM)
        batch_labels = self.labels[batch_ids]                 # (B,)

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
        # labels_f = tf.cast(labels, tf.float32)
        # labels_f = tf.reshape(labels_f, (-1, 1))
        labels_f = tf.cast(labels, tf.float32)
        labels_f = tf.reshape(labels_f, (-1, 1))
        # labels_f = labels_f * 0.9 + 0.05
        labels_smooth = labels_f * 0.9 + 0.05

        # loss = loss_fn(labels_smooth, class_logits)
        loss = 0.5 * loss_fn(labels_smooth, class_logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, chunk_encoder.trainable_variables)
    grads = [
        tf.clip_by_norm(g, GRAD_CLIP_NORM) if g is not None else None
        for g in grads
    ]
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

def compute_train_separation(chunk_encoder, train_seq, max_batches=50):
    all_embs = []
    all_labels = []

    for step in range(min(len(train_seq), max_batches)):
        frame_embs_batch, labels_batch = train_seq[step]

        chunk_embs, _ = chunk_encoder(
            tf.convert_to_tensor(frame_embs_batch),
            training=False
        )

        all_embs.append(chunk_embs.numpy())
        all_labels.append(labels_batch)

    Z = np.concatenate(all_embs)
    y = np.concatenate(all_labels)

    emb_norm = np.mean(np.linalg.norm(Z, axis=1))
    print(f"[train emb] avg_norm={emb_norm:.4f}")

    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    pos_sims = []
    neg_sims = []

    for i in range(len(Z)):
        for j in range(i+1, len(Z)):
            sim = np.dot(Z[i], Z[j])
            if y[i] == y[j]:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)
    
    return np.mean(pos_sims), np.mean(neg_sims)

def compute_train_separation_conditioned(
    chunk_encoder,
    frame_emb_mm,
    chunk_indices,
    labels,
    sides,
    vids,
    t_centers,
    max_samples=2000,
    max_time_gap=0.08,
    require_diff_video=True,
):
    idx = np.random.choice(len(labels), size=min(max_samples, len(labels)), replace=False)

    X = frame_emb_mm[chunk_indices[idx]].astype(np.float32)
    y = labels[idx]
    s = sides[idx]
    v = vids[idx]
    t = t_centers[idx]

    Z, _ = chunk_encoder(tf.convert_to_tensor(X), training=False)
    Z = Z.numpy()

    raw_norm = np.mean(np.linalg.norm(Z, axis=1))
    print(f"[train emb] avg_norm={raw_norm:.4f}")

    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    pos_sims = []
    neg_sims = []

    n = len(Z)
    for i in range(n):
        for j in range(i + 1, n):
            if s[i] != s[j]:
                continue
            if abs(t[i] - t[j]) > max_time_gap:
                continue
            if require_diff_video and v[i] == v[j]:
                continue

            sim = float(np.dot(Z[i], Z[j]))
            if y[i] == y[j]:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        print("[train sep] not enough valid conditioned pairs")
        return None, None

    return np.mean(pos_sims), np.mean(neg_sims)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("======================================")
    print("Stage 1 Chunk Encoder Training")
    print(f"STORE_NAME     : {STORE_NAME}")
    print(f"CHUNK_SIZE     : {CHUNK_SIZE}")
    print(f"CHUNK_STRIDE   : {CHUNK_STRIDE}")
    print(f"EMB_DIM        : {EMB_DIM}")
    print(f"BATCH_SIZE     : {TRAIN_BATCH_SIZE}")
    print(f"EPOCHS         : {EPOCHS}")
    print(f"LR             : {LR}")
    print(f"WEIGHT_DECAY   : {WEIGHT_DECAY}")
    print("======================================")

    frame_emb_mm = load_frame_store(STORE_NAME)
    train_chunk_indices, train_labels, val_chunk_indices, val_labels = load_chunk_index_arrays(STORE_NAME)
    train_meta, val_meta = load_chunk_metadata_arrays(STORE_NAME)
    validate_chunk_arrays(train_chunk_indices, val_chunk_indices)

    print("frame_emb_mm shape   :", frame_emb_mm.shape)
    print("train_chunk_indices  :", train_chunk_indices.shape)
    print("train_labels         :", train_labels.shape)
    print("val_chunk_indices    :", val_chunk_indices.shape)
    print("val_labels           :", val_labels.shape)

    train_seq = ChunkEmbeddingSequence(
        frame_emb_mm=frame_emb_mm,
        chunk_indices=train_chunk_indices,
        labels=train_labels,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_seq = ChunkEmbeddingSequence(
        frame_emb_mm=frame_emb_mm,
        chunk_indices=val_chunk_indices,
        labels=val_labels,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    chunk_encoder = ChunkEncoder(
        hidden_size=EMB_DIM,
        num_layers=4,
        num_heads=8,
        max_frames=CHUNK_SIZE,
    )

    dummy = tf.zeros((2, CHUNK_SIZE, EMB_DIM), dtype=tf.float32)
    _ = chunk_encoder(dummy, training=False)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    best_val_loss = float("inf")
    best_val_acc = -1
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_sep = -10
    for epoch in range(EPOCHS):
        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

        # ---------------- TRAIN ----------------
        train_losses = []
        train_accs = []
        t0 = time.perf_counter()

        for step in range(len(train_seq)):
            frame_embs_batch, labels_batch = train_seq[step]

            loss, acc = train_step(
                chunk_encoder=chunk_encoder,
                optimizer=optimizer,
                loss_fn=loss_fn,
                frame_embs=tf.convert_to_tensor(frame_embs_batch),
                labels=tf.convert_to_tensor(labels_batch),
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
                chunk_encoder=chunk_encoder,
                loss_fn=loss_fn,
                frame_embs=tf.convert_to_tensor(frame_embs_batch),
                labels=tf.convert_to_tensor(labels_batch),
            )

            
            val_losses.append(float(loss.numpy()))
            val_accs.append(float(acc.numpy()))
        # collect_val_stats(chunk_encoder,val_seq)
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        print(
            f"[epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={train_time:.2f}s"
        )

        sep_score = compute_conditioned_separation(
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            chunk_indices=train_chunk_indices,
            labels=train_labels,
            meta=train_meta,
            max_samples=2000,        # start with this
            max_time_gap=0.08,
            require_diff_video=True,
        )
        # pos_sim, neg_sim = compute_train_separation(chunk_encoder, train_seq)
        # sep_score = pos_sim - neg_sim
        # print(f"[train sep] pos_sim={pos_sim:.4f} neg_sim={neg_sim:.4f} sep = {sep_score}")


        
# ./frame_cache_vit/train_val_frames_chunk12_stride4_chunk_meta.npz
        latest_path = os.path.join(SAVE_DIR, "chunk_encoder_latest.weights.h5")
        chunk_encoder.save_weights(latest_path)
        os.makedirs("stage1_block_weights", exist_ok=True)

        # for i in range(chunk_encoder.num_layers):   # or chunk_encoder.num_layers
        #     block = getattr(chunk_encoder, f"transformer_block_{i}")
        #     with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "wb") as f:
        #         pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_path = os.path.join(SAVE_DIR, "chunk_encoder_best.weights.h5")
        #     chunk_encoder.save_weights(best_path)
        #     print(f"saved best weights to {best_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        #     best_path = os.path.join(SAVE_DIR, "chunk_encoder_best_v3.weights.h5")
        #     chunk_encoder.save_weights(best_path)
        #     print(f"saved best weights to {best_path}")

        # if sep_score > best_sep:
            # best_sep = sep_score
            best_path = os.path.join(SAVE_DIR, "chunk_encoder_best_v3.weights.h5")
            chunk_encoder.save_weights(best_path)
            os.makedirs("stage1_block_weights", exist_ok=True)

            for i in range(chunk_encoder.num_layers):   # or chunk_encoder.num_layers
                block = getattr(chunk_encoder, f"transformer_block_{i}")
                with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "wb") as f:
                    pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved best weights to {best_path}")


if __name__ == "__main__":
    main()