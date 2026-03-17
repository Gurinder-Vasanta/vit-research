import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras

import dataset
import config_chunks_cached as config
from models.chunk_encoder import ChunkEncoder

# NEW: PyTorch + transformers imports
import torch
from transformers import ViTModel, ViTImageProcessor

# you already have this somewhere
# from your current code
# def hf_vit_embed_batch(frames_np): ...

AUTOTUNE = tf.data.AUTOTUNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# LOAD PRETRAINED GOOGLE VIT
# ------------------------------
# Automatically handles resizing + center cropping.
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()

def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, 432, 768, 3) uint8 or float32
    Returns (N, 768) numpy embeddings (L2 normalized)
    """
    frames_np = frames_np.astype(np.float32)
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = processor(images=frames_list, return_tensors="pt",do_rescale=False).to(device)
        out = vit_model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (N,768)
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)

def compute_accuracy(labels, logits):
    probs = tf.nn.sigmoid(logits)
    preds = tf.cast(probs >= 0.5, tf.float32)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

def train_step(chunk_encoder, optimizer, loss_fn, frame_embs, labels):
    with tf.GradientTape() as tape:
        chunk_embs, class_logits = chunk_encoder(frame_embs, training=True)
        labels_f = tf.cast(labels, tf.float32)
        labels_f = tf.reshape(labels_f, (-1, 1))

        loss = loss_fn(labels_f, class_logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, chunk_encoder.trainable_variables)
    grads = [
        tf.clip_by_norm(g, 1.0) if g is not None else None
        for g in grads
    ]
    optimizer.apply_gradients(zip(grads, chunk_encoder.trainable_variables))

    acc = compute_accuracy(labels_f, class_logits)
    return loss, acc

def val_step(chunk_encoder, loss_fn, frame_embs, labels):
    chunk_embs, class_logits = chunk_encoder(frame_embs, training=False)
    labels_f = tf.cast(labels, tf.float32)
    labels_f = tf.reshape(labels_f, (-1, 1))

    loss = loss_fn(labels_f, class_logits)
    loss = tf.reduce_mean(loss)
    acc = compute_accuracy(labels_f, class_logits)
    return loss, acc


def build_frame_embeddings(frames_batch):
    """
    frames_batch: (B, T, H, W, 3)
    returns: (B, T, 768)
    """
    B = tf.shape(frames_batch)[0]
    T = tf.shape(frames_batch)[1]

    frames_np = tf.numpy_function(
        hf_vit_embed_batch,
        [tf.reshape(frames_batch, (-1, 432, 768, 3))],
        tf.float32
    )
    frame_embs = tf.reshape(frames_np, (B, T, 768))
    return frame_embs


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    vids = config.VIDS_TO_USE
    samples = dataset.load_samples(
        vids,
        stride=1,
        max_clips=config.NUM_CLIPS_PER_VID
    )

    chunk_size = 12
    chunk_samples = dataset.build_chunks(samples, chunk_size=chunk_size)

    # split however you want
    # if you already have train/val vid split, use that instead
    train_chunks = [c for c in chunk_samples if f"vid{c['vid']}" in config.TRAIN_VIDS]
    val_chunks   = [c for c in chunk_samples if f"vid{c['vid']}" in config.TEST_VIDS]

    # train_chunks = train_chunks[0:100]
    # val_chunks = val_chunks[0:100]
    print("train chunks:", len(train_chunks))
    print("val chunks:", len(val_chunks))

    train_ds = dataset.build_tf_dataset_chunks(train_chunks, batch_size=chunk_size)
    val_ds   = dataset.build_tf_dataset_chunks(val_chunks, batch_size=chunk_size)
    
    chunk_encoder = ChunkEncoder(
        hidden_size=768,
        num_layers=3,
        num_heads=8,
        max_frames=chunk_size
    )

    # force build
    dummy_frame_embs = tf.zeros((2, chunk_size, 768), dtype=tf.float32)
    _ = chunk_encoder(dummy_frame_embs, training=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    best_val_loss = float("inf")
    save_dir = "./chunk_encoder_ckpts"
    os.makedirs(save_dir, exist_ok=True)

    epochs = 10

    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

        # -------- train --------
        train_losses = []
        train_accs = []
        t0 = time.perf_counter()

        for step, (frames_batch, metadata_batch, labels_batch) in enumerate(train_ds):
            frame_embs = build_frame_embeddings(frames_batch)

            loss, acc = train_step(
                chunk_encoder,
                optimizer,
                loss_fn,
                frame_embs,
                labels_batch
            )

            train_losses.append(float(loss.numpy()))
            train_accs.append(float(acc.numpy()))

            if step % 1 == 0:
                print(
                    f"[train] step={step} "
                    f"loss={train_losses[-1]:.4f} "
                    f"acc={train_accs[-1]:.4f}"
                )

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        train_time = time.perf_counter() - t0

        # -------- val --------
        val_losses = []
        val_accs = []
        for step, (frames_batch, metadata_batch, labels_batch) in enumerate(val_ds):
            frame_embs = build_frame_embeddings(frames_batch)

            loss, acc = val_step(
                chunk_encoder,
                loss_fn,
                frame_embs,
                labels_batch
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
