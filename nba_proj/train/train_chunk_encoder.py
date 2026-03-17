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

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
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





# import os
# import time
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as tf_keras

# import dataset
# import config_chunks_cached as config
# from models.chunk_encoder import ChunkEncoder

# # NEW: PyTorch + transformers imports
# import torch
# from transformers import ViTModel, ViTImageProcessor

# # you already have this somewhere
# # from your current code
# # def hf_vit_embed_batch(frames_np): ...

# AUTOTUNE = tf.data.AUTOTUNE

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ------------------------------
# # LOAD PRETRAINED GOOGLE VIT
# # ------------------------------
# # Automatically handles resizing + center cropping.
# processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
# vit_model.eval()

# def hf_vit_embed_batch(frames_np):
#     """
#     frames_np: (N, 432, 768, 3) uint8 or float32
#     Returns (N, 768) numpy embeddings (L2 normalized)
#     """
#     frames_np = frames_np.astype(np.float32)
#     frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
#     with torch.no_grad():
#         inputs = processor(images=frames_list, return_tensors="pt",do_rescale=False).to(device)
#         out = vit_model(**inputs)
#         cls = out.last_hidden_state[:, 0, :]  # (N,768)
#         cls = cls.cpu().numpy()
#         cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
#         return cls.astype(np.float32)

# def compute_accuracy(labels, logits):
#     probs = tf.nn.sigmoid(logits)
#     preds = tf.cast(probs >= 0.5, tf.float32)
#     labels = tf.cast(labels, tf.float32)
#     labels = tf.reshape(labels, (-1, 1))
#     return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

# def train_step(chunk_encoder, optimizer, loss_fn, frame_embs, labels):
#     with tf.GradientTape() as tape:
#         chunk_embs, class_logits = chunk_encoder(frame_embs, training=True)
#         labels_f = tf.cast(labels, tf.float32)
#         labels_f = tf.reshape(labels_f, (-1, 1))

#         loss = loss_fn(labels_f, class_logits)
#         loss = tf.reduce_mean(loss)

#     grads = tape.gradient(loss, chunk_encoder.trainable_variables)
#     grads = [
#         tf.clip_by_norm(g, 1.0) if g is not None else None
#         for g in grads
#     ]
#     optimizer.apply_gradients(zip(grads, chunk_encoder.trainable_variables))

#     acc = compute_accuracy(labels_f, class_logits)
#     return loss, acc

# def val_step(chunk_encoder, loss_fn, frame_embs, labels):
#     chunk_embs, class_logits = chunk_encoder(frame_embs, training=False)
#     labels_f = tf.cast(labels, tf.float32)
#     labels_f = tf.reshape(labels_f, (-1, 1))

#     loss = loss_fn(labels_f, class_logits)
#     loss = tf.reduce_mean(loss)
#     acc = compute_accuracy(labels_f, class_logits)
#     return loss, acc


# def build_frame_embeddings(frames_batch):
#     """
#     frames_batch: (B, T, H, W, 3)
#     returns: (B, T, 768)
#     """
#     B = tf.shape(frames_batch)[0]
#     T = tf.shape(frames_batch)[1]

#     frames_np = tf.numpy_function(
#         hf_vit_embed_batch,
#         [tf.reshape(frames_batch, (-1, 432, 768, 3))],
#         tf.float32
#     )
#     frame_embs = tf.reshape(frames_np, (B, T, 768))
#     return frame_embs


# def main():
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     vids = config.VIDS_TO_USE
#     samples = dataset.load_samples(
#         vids,
#         stride=1,
#         max_clips=config.NUM_CLIPS_PER_VID
#     )

#     chunk_size = 12
#     chunk_samples = dataset.build_chunks(samples, chunk_size=chunk_size)

#     # split however you want
#     # if you already have train/val vid split, use that instead
#     train_chunks = [c for c in chunk_samples if f"vid{c['vid']}" in config.TRAIN_VIDS]
#     val_chunks   = [c for c in chunk_samples if f"vid{c['vid']}" in config.TEST_VIDS]

#     # train_chunks = train_chunks[0:100]
#     # val_chunks = val_chunks[0:100]
#     print("train chunks:", len(train_chunks))
#     print("val chunks:", len(val_chunks))

#     train_ds = dataset.build_tf_dataset_chunks(train_chunks, batch_size=chunk_size)
#     val_ds   = dataset.build_tf_dataset_chunks(val_chunks, batch_size=chunk_size)
    
#     chunk_encoder = ChunkEncoder(
#         hidden_size=768,
#         num_layers=3,
#         num_heads=8,
#         max_frames=chunk_size
#     )

#     # force build
#     dummy_frame_embs = tf.zeros((2, chunk_size, 768), dtype=tf.float32)
#     _ = chunk_encoder(dummy_frame_embs, training=False)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
#     loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#     best_val_loss = float("inf")
#     save_dir = "./chunk_encoder_ckpts"
#     os.makedirs(save_dir, exist_ok=True)

#     epochs = 100

#     for epoch in range(epochs):
#         print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

#         # -------- train --------
#         train_losses = []
#         train_accs = []
#         t0 = time.perf_counter()

#         for step, (frames_batch, metadata_batch, labels_batch) in enumerate(train_ds):
#             t2 = time.perf_counter()
#             frame_embs = build_frame_embeddings(frames_batch)
#             # frame_embs = tf.numpy_function(
#             #     hf_vit_embed_batch,
#             #     [tf.reshape(frames_batch, (-1, 432, 768, 3))],
#             #     tf.float32
#             # )
#             t3 = time.perf_counter()
#             print(t3 - t2)
#             # input('stop')
#             t4 = time.perf_counter()
#             loss, acc = train_step(
#                 chunk_encoder,
#                 optimizer,
#                 loss_fn,
#                 frame_embs,
#                 labels_batch
#             )
#             t5 = time.perf_counter()
#             print(t5-t4)
#             train_losses.append(float(loss.numpy()))
#             train_accs.append(float(acc.numpy()))

#             if step % 1 == 0:
#                 print(
#                     f"[train] step={step} "
#                     f"loss={train_losses[-1]:.4f} "
#                     f"acc={train_accs[-1]:.4f}"
#                 )

#         train_loss = float(np.mean(train_losses))
#         train_acc = float(np.mean(train_accs))
#         train_time = time.perf_counter() - t0

#         # -------- val --------
#         val_losses = []
#         val_accs = []
#         for step, (frames_batch, metadata_batch, labels_batch) in enumerate(val_ds):
#             frame_embs = build_frame_embeddings(frames_batch)

#             loss, acc = val_step(
#                 chunk_encoder,
#                 loss_fn,
#                 frame_embs,
#                 labels_batch
#             )

#             val_losses.append(float(loss.numpy()))
#             val_accs.append(float(acc.numpy()))

#         val_loss = float(np.mean(val_losses))
#         val_acc = float(np.mean(val_accs))

#         print(
#             f"[epoch {epoch + 1}] "
#             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
#             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
#             f"time={train_time:.2f}s"
#         )

#         latest_path = os.path.join(save_dir, "chunk_encoder_latest.weights.h5")
#         chunk_encoder.save_weights(latest_path)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_path = os.path.join(save_dir, "chunk_encoder_best.weights.h5")
#             chunk_encoder.save_weights(best_path)
#             print(f"saved best weights to {best_path}")


# if __name__ == "__main__":
#     main()







# import os
# import time
# import json
# import hashlib
# import gc
# import numpy as np
# import tensorflow as tf
# import torch
# from transformers import ViTModel, ViTImageProcessor

# import dataset
# import config_chunks_cached as config
# from models.chunk_encoder import ChunkEncoder

# AUTOTUNE = tf.data.AUTOTUNE
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHUNK_SIZE = 12
# TRAIN_BATCH_SIZE = 16
# EMBED_BATCH_SIZE = 32
# WRITE_CHUNK_GROUP = 512   # number of chunks to process/write at a time

# EPOCHS = 100
# LR = 5e-4

# CACHE_DIR = "./stage1_faster_embed_cache"
# CKPT_DIR = "./chunk_faster_encoder_ckpts"
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(CKPT_DIR, exist_ok=True)

# processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
# vit_model.eval()


# def hf_vit_embed_batch(frames_np):
#     """
#     frames_np: (N, H, W, 3)
#     returns: (N, 768)
#     NO NORMALIZATION
#     """
#     frames_np = frames_np.astype(np.float32)
#     frames_list = [frames_np[i] for i in range(frames_np.shape[0])]

#     with torch.no_grad():
#         inputs = processor(
#             images=frames_list,
#             return_tensors="pt",
#             do_rescale=False
#         ).to(device)

#         out = vit_model(**inputs)
#         cls = out.last_hidden_state[:, 0, :]
#         cls = cls.cpu().numpy().astype(np.float32)

#     del inputs, out
#     if device.type == "cuda":
#         torch.cuda.empty_cache()
#     return cls


# def make_cache_key(train_vids, test_vids, vids_to_use, num_clips_per_vid, chunk_size):
#     payload = {
#         "train_vids": sorted(list(train_vids)),
#         "test_vids": sorted(list(test_vids)),
#         "vids_to_use": list(vids_to_use),
#         "num_clips_per_vid": num_clips_per_vid,
#         "chunk_size": chunk_size,
#         "model": "google/vit-base-patch16-224",
#         "normalized": False,
#     }
#     s = json.dumps(payload, sort_keys=True)
#     return hashlib.md5(s.encode()).hexdigest()


# def get_split_paths(cache_prefix, split_name):
#     base = os.path.join(CACHE_DIR, f"{cache_prefix}_{split_name}")
#     return {
#         "emb": base + "_frame_embs.dat",
#         "labels": base + "_labels.dat",
#         "meta": base + "_meta.npz",
#     }


# def cache_exists(paths):
#     return (
#         os.path.exists(paths["emb"]) and
#         os.path.exists(paths["labels"]) and
#         os.path.exists(paths["meta"])
#     )


# def build_memmap_cache(chunk_list, split_name, cache_prefix):
#     """
#     Streams chunks -> embeddings -> disk-backed memmap.
#     Never holds the whole dataset or whole embedding matrix in RAM.
#     """
#     paths = get_split_paths(cache_prefix, split_name)

#     N = len(chunk_list)
#     print(f"\n[cache build] split={split_name}, chunks={N}")

#     emb_mm = np.memmap(
#         paths["emb"],
#         dtype="float32",
#         mode="w+",
#         shape=(N, CHUNK_SIZE, 768),
#     )
#     labels_mm = np.memmap(
#         paths["labels"],
#         dtype="float32",
#         mode="w+",
#         shape=(N,),
#     )

#     # Build dataset one chunk at a time so indexing stays simple.
#     raw_ds = dataset.build_tf_dataset_chunks(chunk_list, batch_size=32)

#     chunk_frames_buffer = []
#     chunk_labels_buffer = []
#     write_start_idx = 0

#     t0 = time.perf_counter()

#     def flush_buffer():
#         nonlocal write_start_idx, chunk_frames_buffer, chunk_labels_buffer

#         if len(chunk_frames_buffer) == 0:
#             return

#         frames_np = np.stack(chunk_frames_buffer, axis=0)   # (B, T, H, W, 3)
#         labels_np = np.asarray(chunk_labels_buffer, dtype=np.float32)  # (B,)

#         B, T, H, W, C = frames_np.shape
#         flat_frames = frames_np.reshape(B * T, H, W, C)

#         flat_embs_parts = []
#         for start in range(0, flat_frames.shape[0], EMBED_BATCH_SIZE):
#             end = min(start + EMBED_BATCH_SIZE, flat_frames.shape[0])
#             batch_embs = hf_vit_embed_batch(flat_frames[start:end])
#             flat_embs_parts.append(batch_embs)

#         flat_embs = np.concatenate(flat_embs_parts, axis=0)   # (B*T, 768)
#         chunk_embs = flat_embs.reshape(B, T, 768)

#         write_end_idx = write_start_idx + B
#         emb_mm[write_start_idx:write_end_idx] = chunk_embs
#         labels_mm[write_start_idx:write_end_idx] = labels_np

#         emb_mm.flush()
#         labels_mm.flush()

#         print(
#             f"[cache build] wrote chunks {write_start_idx}:{write_end_idx}/{N}"
#         )

#         write_start_idx = write_end_idx

#         # aggressively free memory
#         del frames_np, labels_np, flat_frames, flat_embs_parts, flat_embs, chunk_embs
#         chunk_frames_buffer.clear()
#         chunk_labels_buffer.clear()
#         gc.collect()
#         if device.type == "cuda":
#             torch.cuda.empty_cache()

#     for i, (frames_batch, metadata_batch, labels_batch) in enumerate(raw_ds):
#         chunk_frames_buffer.append(frames_batch.numpy()[0])   # (T, H, W, 3)
#         chunk_labels_buffer.append(labels_batch.numpy()[0])

#         if len(chunk_frames_buffer) >= WRITE_CHUNK_GROUP:
#             flush_buffer()

#     flush_buffer()

#     np.savez_compressed(
#         paths["meta"],
#         n_chunks=N,
#         chunk_size=CHUNK_SIZE,
#         emb_dim=768,
#     )

#     total_time = time.perf_counter() - t0
#     print(f"[cache build] done split={split_name} in {total_time:.2f}s")
#     return paths


# def load_memmap_cache(split_name, cache_prefix):
#     paths = get_split_paths(cache_prefix, split_name)
#     meta = np.load(paths["meta"])

#     N = int(meta["n_chunks"])
#     T = int(meta["chunk_size"])
#     D = int(meta["emb_dim"])

#     frame_embs = np.memmap(
#         paths["emb"],
#         dtype="float32",
#         mode="r",
#         shape=(N, T, D),
#     )
#     labels = np.memmap(
#         paths["labels"],
#         dtype="float32",
#         mode="r",
#         shape=(N,),
#     )

#     print(f"[cache] loaded split={split_name} frame_embs={(N, T, D)} labels={(N,)}")
#     return frame_embs, labels


# def get_or_build_memmap_cache(chunk_list, split_name, cache_prefix):
#     paths = get_split_paths(cache_prefix, split_name)
#     if cache_exists(paths):
#         return load_memmap_cache(split_name, cache_prefix)

#     build_memmap_cache(chunk_list, split_name, cache_prefix)
#     return load_memmap_cache(split_name, cache_prefix)


# def build_embedding_dataset(frame_embs, labels, batch_size, training):
#     ds = tf.data.Dataset.from_tensor_slices((frame_embs, labels))
#     if training:
#         ds = ds.shuffle(min(len(labels), 10000), reshuffle_each_iteration=True)
#     ds = ds.batch(batch_size).prefetch(AUTOTUNE)
#     return ds


# def compute_accuracy(labels, logits):
#     probs = tf.nn.sigmoid(logits)
#     preds = tf.cast(probs >= 0.5, tf.float32)
#     labels = tf.cast(labels, tf.float32)
#     labels = tf.reshape(labels, (-1, 1))
#     return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


# @tf.function
# def train_step(chunk_encoder, optimizer, loss_fn, frame_embs, labels):
#     with tf.GradientTape() as tape:
#         _, class_logits = chunk_encoder(frame_embs, training=True)
#         labels_f = tf.cast(labels, tf.float32)
#         labels_f = tf.reshape(labels_f, (-1, 1))

#         loss = loss_fn(labels_f, class_logits)
#         loss = tf.reduce_mean(loss)

#     grads = tape.gradient(loss, chunk_encoder.trainable_variables)
#     grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
#     optimizer.apply_gradients(zip(grads, chunk_encoder.trainable_variables))

#     acc = compute_accuracy(labels_f, class_logits)
#     return loss, acc


# @tf.function
# def val_step(chunk_encoder, loss_fn, frame_embs, labels):
#     _, class_logits = chunk_encoder(frame_embs, training=False)
#     labels_f = tf.cast(labels, tf.float32)
#     labels_f = tf.reshape(labels_f, (-1, 1))

#     loss = loss_fn(labels_f, class_logits)
#     loss = tf.reduce_mean(loss)
#     acc = compute_accuracy(labels_f, class_logits)
#     return loss, acc


# def main():
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     vids = config.VIDS_TO_USE
#     samples = dataset.load_samples(
#         vids,
#         stride=1,
#         max_clips=config.NUM_CLIPS_PER_VID
#     )

#     chunk_samples = dataset.build_chunks(samples, chunk_size=CHUNK_SIZE)

#     train_chunks = [c for c in chunk_samples if f"vid{c['vid']}" in config.TRAIN_VIDS]
#     val_chunks   = [c for c in chunk_samples if f"vid{c['vid']}" in config.TEST_VIDS]

#     print("train chunks:", len(train_chunks))
#     print("val chunks:", len(val_chunks))

#     cache_prefix = make_cache_key(
#         train_vids=config.TRAIN_VIDS,
#         test_vids=config.TEST_VIDS,
#         vids_to_use=config.VIDS_TO_USE,
#         num_clips_per_vid=config.NUM_CLIPS_PER_VID,
#         chunk_size=CHUNK_SIZE,
#     )

#     train_frame_embs, train_labels = get_or_build_memmap_cache(
#         train_chunks, "train", cache_prefix
#     )
#     val_frame_embs, val_labels = get_or_build_memmap_cache(
#         val_chunks, "val", cache_prefix
#     )

#     train_ds = build_embedding_dataset(
#         train_frame_embs, train_labels,
#         batch_size=TRAIN_BATCH_SIZE,
#         training=True
#     )
#     val_ds = build_embedding_dataset(
#         val_frame_embs, val_labels,
#         batch_size=TRAIN_BATCH_SIZE,
#         training=False
#     )

#     chunk_encoder = ChunkEncoder(
#         hidden_size=768,
#         num_layers=3,
#         num_heads=8,
#         max_frames=CHUNK_SIZE
#     )

#     dummy_frame_embs = tf.zeros((2, CHUNK_SIZE, 768), dtype=tf.float32)
#     _ = chunk_encoder(dummy_frame_embs, training=False)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
#     loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#     best_val_loss = float("inf")

#     for epoch in range(EPOCHS):
#         print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

#         train_losses = []
#         train_accs = []
#         t0 = time.perf_counter()

#         for step, (frame_embs_batch, labels_batch) in enumerate(train_ds):
#             loss, acc = train_step(
#                 chunk_encoder, optimizer, loss_fn, frame_embs_batch, labels_batch
#             )
#             train_losses.append(float(loss.numpy()))
#             train_accs.append(float(acc.numpy()))
#             print(f"[train] step={step} loss={train_losses[-1]:.4f} acc={train_accs[-1]:.4f}")

#         train_loss = float(np.mean(train_losses))
#         train_acc = float(np.mean(train_accs))
#         train_time = time.perf_counter() - t0

#         val_losses = []
#         val_accs = []
#         for step, (frame_embs_batch, labels_batch) in enumerate(val_ds):
#             loss, acc = val_step(chunk_encoder, loss_fn, frame_embs_batch, labels_batch)
#             val_losses.append(float(loss.numpy()))
#             val_accs.append(float(acc.numpy()))

#         val_loss = float(np.mean(val_losses))
#         val_acc = float(np.mean(val_accs))

#         print(
#             f"[epoch {epoch + 1}] "
#             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
#             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
#             f"time={train_time:.2f}s"
#         )

#         latest_path = os.path.join(CKPT_DIR, "chunk_encoder_latest.weights.h5")
#         chunk_encoder.save_weights(latest_path)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_path = os.path.join(CKPT_DIR, "chunk_encoder_best.weights.h5")
#             chunk_encoder.save_weights(best_path)
#             print(f"saved best weights to {best_path}")


# if __name__ == "__main__":
#     main()
