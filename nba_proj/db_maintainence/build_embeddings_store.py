import os
import gc
import time
import random
import pprint
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor

import config_chunks_cached as config_chunks
from dataset import load_samples, build_chunks


# ============================================================
# CONFIG
# ============================================================
FRAME_BATCH_SIZE = 1024          # number of raw frames per ViT pass
NUM_LOAD_WORKERS = 16            # tune: 8 / 16 / 24 are good values to test
CACHE_DIR = "./frame_cache_vit"

os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD PRETRAINED GOOGLE VIT
# ============================================================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor.do_rescale = False
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()


# ============================================================
# DATA PREP
# ============================================================
def build_chunk_sets():
    train_vids = config_chunks.TRAIN_VIDS
    test_vids = config_chunks.TEST_VIDS

    train_samples = load_samples(train_vids, stride=1)
    test_samples = load_samples(test_vids, stride=1)

    # These shuffles only matter if you want randomized sample order before chunking.
    # If not needed, you can remove them.
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    train_chunk_samples = build_chunks(
        train_samples,
        chunk_size=config_chunks.CHUNK_SIZE,
        chunk_stride=config_chunks.CHUNK_STRIDE,
    )
    test_chunk_samples = build_chunks(
        test_samples,
        chunk_size=config_chunks.CHUNK_SIZE,
        chunk_stride=config_chunks.CHUNK_STRIDE,
    )

    print(f"Train chunks: {len(train_chunk_samples)}")
    print(f"Val chunks:   {len(test_chunk_samples)}")
    print("Example chunks:")
    pprint.pprint(train_chunk_samples[0])
    pprint.pprint(train_chunk_samples[1])

    return train_chunk_samples, test_chunk_samples


def collect_unique_frame_paths(chunk_samples):
    seen = set()
    ordered = []

    for c in chunk_samples:
        for fp in c["frames"]:
            if fp not in seen:
                seen.add(fp)
                ordered.append(fp)

    return ordered


# ============================================================
# IMAGE LOADING
# ============================================================
def load_one_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_frame_batch_parallel(frame_paths, executor):
    imgs = list(executor.map(load_one_image, frame_paths))
    return np.stack(imgs, axis=0)  # (B, H, W, 3)


# ============================================================
# VIT EMBEDDING
# ============================================================
def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, H, W, 3) uint8 or float32
    returns: (N, 768) float32
    """
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]

    with torch.no_grad():
        inputs = processor(images=frames_list, return_tensors="pt").to(device)
        out = vit_model(**inputs)
        cls = out.last_hidden_state[:, 0, :]   # (N, 768)
        cls = cls.cpu().numpy()

        # keep this because you said normalization here is okay
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        cls = cls.astype(np.float32)

    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cls


# ============================================================
# FRAME STORE BUILD / LOAD
# ============================================================
def get_store_paths(store_name):
    return {
        "emb": os.path.join(CACHE_DIR, f"{store_name}_emb.dat"),
        "paths": os.path.join(CACHE_DIR, f"{store_name}_paths.npy"),
        "meta": os.path.join(CACHE_DIR, f"{store_name}_meta.npz"),
    }


def frame_store_exists(store_name):
    paths = get_store_paths(store_name)
    return (
        os.path.exists(paths["emb"]) and
        os.path.exists(paths["paths"]) and
        os.path.exists(paths["meta"])
    )


def build_frame_store(frame_paths, batch_size=512, num_workers=16, store_name="train_val_frames"):
    paths = get_store_paths(store_name)
    n = len(frame_paths)

    emb_mm = np.memmap(
        paths["emb"],
        dtype="float32",
        mode="w+",
        shape=(n, 768),
    )

    t_total0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_paths = frame_paths[start:end]

            t0 = time.perf_counter()
            batch_imgs = load_frame_batch_parallel(batch_paths, executor)
            t_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            batch_embs = hf_vit_embed_batch(batch_imgs)
            t_embed = time.perf_counter() - t1

            emb_mm[start:end] = batch_embs

            # flush occasionally, not on every batch
            if ((start // batch_size) + 1) % 10 == 0 or end == n:
                emb_mm.flush()

            t_batch = time.perf_counter() - t0
            print(f"[frame cache] wrote {end}/{n} in {t_batch:.2f}s")
            print(f"load={t_load:.2f}s embed={t_embed:.2f}s total={t_batch:.2f}s")

            del batch_imgs, batch_embs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    emb_mm.flush()
    np.save(paths["paths"], np.array(frame_paths, dtype=object))
    np.savez_compressed(paths["meta"], n_frames=n, emb_dim=768)

    total_elapsed = time.perf_counter() - t_total0
    print(f"[frame cache] done in {total_elapsed:.2f}s")
    print(f"[frame cache] saved embeddings to {paths['emb']}")
    print(f"[frame cache] saved paths to      {paths['paths']}")
    print(f"[frame cache] saved meta to       {paths['meta']}")


def load_frame_store(store_name="train_val_frames"):
    paths = get_store_paths(store_name)
    meta = np.load(paths["meta"])

    n_frames = int(meta["n_frames"])
    emb_dim = int(meta["emb_dim"])

    emb_mm = np.memmap(
        paths["emb"],
        dtype="float32",
        mode="r",
        shape=(n_frames, emb_dim),
    )

    frame_paths = np.load(paths["paths"], allow_pickle=True).tolist()
    path_to_idx = {p: i for i, p in enumerate(frame_paths)}

    print(f"[frame cache] loaded store '{store_name}'")
    print(f"[frame cache] shape = ({n_frames}, {emb_dim})")

    return emb_mm, frame_paths, path_to_idx


# ============================================================
# CHUNK -> FRAME INDEX ARRAYS
# ============================================================
# def build_chunk_index_array(chunk_samples, path_to_idx):
#     chunk_size = len(chunk_samples[0]["frames"])

#     X_idx = np.empty((len(chunk_samples), chunk_size), dtype=np.int32)
#     y = np.empty((len(chunk_samples),), dtype=np.float32)

#     for i, c in enumerate(chunk_samples):
#         X_idx[i] = [path_to_idx[p] for p in c["frames"]]
#         y[i] = np.float32(c["label"])

#     return X_idx, y


def build_chunk_index_array(chunk_samples, path_to_idx):
    chunk_size = len(chunk_samples[0]["frames"])
    n = len(chunk_samples)

    X_idx = np.empty((n, chunk_size), dtype=np.int32)
    y = np.empty((n,), dtype=np.float32)

    vid = np.empty((n,), dtype=np.int32)
    clip = np.empty((n,), dtype=np.int32)
    side = np.empty((n,), dtype=object)
    t_center = np.empty((n,), dtype=np.float32)
    t_width = np.empty((n,), dtype=np.float32)

    for i, c in enumerate(chunk_samples):
        X_idx[i] = [path_to_idx[p] for p in c["frames"]]
        y[i] = np.float32(c["label"])

        vid[i] = np.int32(c["vid"])
        clip[i] = np.int32(c["clip"])
        side[i] = c["side"]
        t_center[i] = np.float32(c["t_center"])
        t_width[i] = np.float32(c["t_width"])

    meta = {
        "vid": vid,
        "clip": clip,
        "side": side,
        "t_center": t_center,
        "t_width": t_width,
    }

    return X_idx, y, meta

def save_chunk_index_arrays(train_chunk_indices, train_labels,
                            val_chunk_indices, val_labels,
                            store_name="train_val_frames"):
    out_path = os.path.join(CACHE_DIR, f"{store_name}_chunk_indices.npz")
    np.savez_compressed(
        out_path,
        train_chunk_indices=train_chunk_indices,
        train_labels=train_labels,
        val_chunk_indices=val_chunk_indices,
        val_labels=val_labels,
    )
    print(f"[chunk index] saved to {out_path}")

def save_chunk_metadata_arrays(train_meta, val_meta, store_name="train_val_frames"):
    out_path = os.path.join(CACHE_DIR, f"{store_name}_chunk_meta.npz")
    np.savez_compressed(
        out_path,
        train_vid=train_meta["vid"],
        train_clip=train_meta["clip"],
        train_side=train_meta["side"],
        train_t_center=train_meta["t_center"],
        train_t_width=train_meta["t_width"],

        val_vid=val_meta["vid"],
        val_clip=val_meta["clip"],
        val_side=val_meta["side"],
        val_t_center=val_meta["t_center"],
        val_t_width=val_meta["t_width"],
    )
    print(f"[chunk meta] saved to {out_path}")

def load_chunk_index_arrays(store_name="train_val_frames"):
    path = os.path.join(CACHE_DIR, f"{store_name}_chunk_indices.npz")
    data = np.load(path)

    train_chunk_indices = data["train_chunk_indices"]
    train_labels = data["train_labels"]
    val_chunk_indices = data["val_chunk_indices"]
    val_labels = data["val_labels"]

    print("[chunk index] loaded")
    print("train_chunk_indices:", train_chunk_indices.shape)
    print("train_labels:       ", train_labels.shape)
    print("val_chunk_indices:  ", val_chunk_indices.shape)
    print("val_labels:         ", val_labels.shape)

    return train_chunk_indices, train_labels, val_chunk_indices, val_labels


# ============================================================
# OPTIONAL: BATCH GATHER HELPER
# ============================================================
def gather_chunk_embedding_batch(frame_emb_mm, chunk_indices_batch):
    """
    frame_emb_mm: memmap (N_frames, 768)
    chunk_indices_batch: np array (B, T)
    returns: np array (B, T, 768)
    """
    return frame_emb_mm[chunk_indices_batch].astype(np.float32)


def summarize_chunks(name, chunk_samples):
    print(f"\n[{name}] num_chunks = {len(chunk_samples)}")
    if not chunk_samples:
        return

    widths = [c["t_width"] for c in chunk_samples]
    clips = {(c["vid"], c["clip"]) for c in chunk_samples}
    print(f"[{name}] unique clips = {len(clips)}")
    print(f"[{name}] t_width min/mean/max = "
          f"{min(widths):.4f} / {np.mean(widths):.4f} / {max(widths):.4f}")

    by_clip = {}
    for c in chunk_samples:
        key = (c["vid"], c["clip"])
        by_clip[key] = by_clip.get(key, 0) + 1

    counts = list(by_clip.values())
    print(f"[{name}] chunks per clip min/mean/max = "
          f"{min(counts)} / {np.mean(counts):.2f} / {max(counts)}")
    
# ============================================================
# MAIN
# ============================================================
def main():
    store_name = "train_val_frames_chunk8_stride2"

    train_chunk_samples, val_chunk_samples = build_chunk_sets()

    # Build frame store once if missing
    if not frame_store_exists(store_name):
        all_frame_paths = collect_unique_frame_paths(train_chunk_samples + val_chunk_samples)
        print("unique frames:", len(all_frame_paths))
        build_frame_store(
            all_frame_paths,
            batch_size=FRAME_BATCH_SIZE,
            num_workers=NUM_LOAD_WORKERS,
            store_name=store_name,
        )
    else:
        print(f"[frame cache] store '{store_name}' already exists, skipping build")

    # Load frame store
    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

    # Build chunk index arrays
    # train_chunk_indices, train_labels = build_chunk_index_array(train_chunk_samples, path_to_idx)
    # val_chunk_indices, val_labels = build_chunk_index_array(val_chunk_samples, path_to_idx)

    train_chunk_indices, train_labels, train_meta = build_chunk_index_array(train_chunk_samples, path_to_idx)
    val_chunk_indices, val_labels, val_meta = build_chunk_index_array(val_chunk_samples, path_to_idx)

    print("train_chunk_indices:", train_chunk_indices.shape)
    print("train_labels:       ", train_labels.shape)
    print("val_chunk_indices:  ", val_chunk_indices.shape)
    print("val_labels:         ", val_labels.shape)
    summarize_chunks("train", train_chunk_samples)
    summarize_chunks("val", val_chunk_samples)

    save_chunk_index_arrays(
        train_chunk_indices, train_labels,
        val_chunk_indices, val_labels,
        store_name=store_name
    )

    save_chunk_metadata_arrays(
        train_meta,
        val_meta,
        store_name=store_name
    )
    # sanity check: gather one mini-batch
    B = min(4, len(train_chunk_indices))
    batch_embs = gather_chunk_embedding_batch(frame_emb_mm, train_chunk_indices[:B])
    print("sanity batch embeddings shape:", batch_embs.shape)
    print("expected shape:              ", (B, config_chunks.CHUNK_SIZE, 768))


if __name__ == "__main__":
    main()
