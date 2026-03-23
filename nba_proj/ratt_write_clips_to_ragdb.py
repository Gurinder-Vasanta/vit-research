import os
import gc
import time
import pprint
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import chromadb
from chromadb import PersistentClient

import tensorflow as tf
import tensorflow.keras as tf_keras

import torch
from transformers import ViTModel, ViTImageProcessor

import dataset
import config_chunks_cached as config_ratt
from models.chunk_encoder import ChunkEncoder


# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = "./frame_cache_vit"
STORE_NAME = "train_val_frames" #train_val_frames_all_vids
FRAME_BATCH_SIZE = 1024
NUM_LOAD_WORKERS = 16

HIDDEN_SIZE = 768
NUM_LAYERS = 3
NUM_HEADS = 8

# set this to your trained chunk encoder checkpoint
CHUNK_ENCODER_WEIGHTS = "./chunk_encoder_ckpts_cached/chunk_encoder_best.weights.h5"


# Chroma
CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "ratt_db_chunk_encoder_all_vids_new"
COLLECTION_NAME_RELCLS = "ratt_db_chunk_encoder_all_vids_relcls_new"

UPSERT_SIZE = 512
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD PRETRAINED GOOGLE VIT
# ============================================================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()


# ============================================================
# HELPERS
# ============================================================
def resolve_all_vids():
    if hasattr(config_ratt, "TRAIN_VIDS") and hasattr(config_ratt, "TEST_VIDS"):
        vids = sorted(set(list(config_ratt.TRAIN_VIDS) + list(config_ratt.TEST_VIDS)))
    elif hasattr(config_ratt, "VIDS_TO_USE"):
        vids = list(config_ratt.VIDS_TO_USE)
    else:
        raise ValueError("Could not find TRAIN_VIDS/TEST_VIDS or VIDS_TO_USE in config_chunks_cached.")
    return vids


def make_chunk_id(c):
    return f"vid{int(c['vid'])}_clip_{int(c['clip'])}_t_{float(c['t_center']):.3f}"


# ============================================================
# DATA PREP
# ============================================================
def build_all_chunk_samples():
    vids = resolve_all_vids()
    print("Using vids:", vids)

    max_clips = getattr(config_ratt, "NUM_CLIPS_PER_VID", None)
    samples = dataset.load_samples(vids, stride=1, max_clips=max_clips)

    chunk_samples = dataset.build_chunks(
        samples,
        chunk_size=config_ratt.CHUNK_SIZE
    )

    print(f"Total chunks: {len(chunk_samples)}")
    print("Example chunk:")
    pprint.pprint(chunk_samples[0])

    return chunk_samples


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
    return np.stack(imgs, axis=0)


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

        # keep same normalization convention as your cached store
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
    os.makedirs(CACHE_DIR, exist_ok=True)
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
# CHUNK GATHER
# ============================================================
def gather_chunk_embedding_batch(frame_emb_mm, chunk_indices_batch):
    return frame_emb_mm[chunk_indices_batch].astype(np.float32)


def build_chunk_index_array(chunk_samples, path_to_idx):
    chunk_size = len(chunk_samples[0]["frames"])

    X_idx = np.empty((len(chunk_samples), chunk_size), dtype=np.int32)
    y = np.empty((len(chunk_samples),), dtype=np.float32)

    for i, c in enumerate(chunk_samples):
        X_idx[i] = [path_to_idx[p] for p in c["frames"]]
        y[i] = np.float32(c["label"])

    return X_idx, y


# ============================================================
# CHUNK ENCODER LOAD
# ============================================================
def load_chunk_encoder():
    chunk_encoder = ChunkEncoder(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_frames=config_ratt.CHUNK_SIZE,
    )

    dummy = tf.zeros((2, config_ratt.CHUNK_SIZE, HIDDEN_SIZE), dtype=tf.float32)
    _ = chunk_encoder(dummy, training=False)

    chunk_encoder.load_weights(CHUNK_ENCODER_WEIGHTS)
    print(f"[chunk encoder] loaded weights from {CHUNK_ENCODER_WEIGHTS}")
    return chunk_encoder


# ============================================================
# CHROMA
# ============================================================
def make_collections():
    client = PersistentClient(path=CHROMA_PATH)

    ratt_db = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ratt_db_relcls = client.get_or_create_collection(
        name=COLLECTION_NAME_RELCLS,
        metadata={"hnsw:space": "cosine"}
    )

    return ratt_db, ratt_db_relcls


# ============================================================
# MAIN
# ============================================================
def main():
    t0_all = time.perf_counter()

    chunk_samples = build_all_chunk_samples()
    all_frame_paths = collect_unique_frame_paths(chunk_samples)
    print("unique frames:", len(all_frame_paths))

    # Build or load frame store
    if not frame_store_exists(STORE_NAME):
        build_frame_store(
            all_frame_paths,
            batch_size=FRAME_BATCH_SIZE,
            num_workers=NUM_LOAD_WORKERS,
            store_name=STORE_NAME,
        )
    else:
        print(f"[frame cache] store '{STORE_NAME}' already exists, checking coverage")

    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(STORE_NAME)

    missing = [p for p in all_frame_paths if p not in path_to_idx]
    if missing:
        print(f"[frame cache] existing store missing {len(missing)} frames, rebuilding")
        paths = get_store_paths(STORE_NAME)
        for k in ("emb", "paths", "meta"):
            if os.path.exists(paths[k]):
                os.remove(paths[k])

        build_frame_store(
            all_frame_paths,
            batch_size=FRAME_BATCH_SIZE,
            num_workers=NUM_LOAD_WORKERS,
            store_name=STORE_NAME,
        )
        frame_emb_mm, frame_paths, path_to_idx = load_frame_store(STORE_NAME)

    chunk_indices, labels = build_chunk_index_array(chunk_samples, path_to_idx)
    print("chunk_indices:", chunk_indices.shape)
    print("labels:       ", labels.shape)

    if chunk_indices.shape[1] != config_ratt.CHUNK_SIZE:
        raise ValueError(
            f"config CHUNK_SIZE={config_ratt.CHUNK_SIZE}, "
            f"but chunk_indices width={chunk_indices.shape[1]}"
        )

    chunk_encoder = load_chunk_encoder()
    ratt_db, ratt_db_relcls = make_collections()

    # clear current contents if you want a full rebuild
    ratt_db.delete(where={"vid_num": {"$gte": 0}})
    ratt_db_relcls.delete(where={"vid_num": {"$gte": 0}})

    b_embeddings = []
    b_ids = []
    b_metadatas = []

    n_chunks = len(chunk_samples)
    print(f"[populate] starting upserts for {n_chunks} chunks")

    for start in range(0, n_chunks, UPSERT_SIZE):
        end = min(start + UPSERT_SIZE, n_chunks)

        idx_batch = chunk_indices[start:end]
        label_batch = labels[start:end]
        chunk_batch = chunk_samples[start:end]

        frame_embs_batch = gather_chunk_embedding_batch(frame_emb_mm, idx_batch)
        frame_embs_batch = tf.convert_to_tensor(frame_embs_batch, dtype=tf.float32)

        chunk_embs, class_logits = chunk_encoder(frame_embs_batch, training=False)
        chunk_embs = chunk_embs.numpy().astype(np.float32)
        class_logits = class_logits.numpy().reshape(-1).astype(np.float32)

        # optional: l2 normalize for cosine search
        chunk_embs = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)

        for i, c in enumerate(chunk_batch):
            cur_id = make_chunk_id(c)
            b_ids.append(cur_id)
            b_embeddings.append(chunk_embs[i:i+1])

            meta = {
                "vid_num": int(c["vid"]),
                "clip_num": int(c["clip"]),
                "side": str(c["side"]),
                "label": int(label_batch[i]),
                "t_center": float(c["t_center"]),
                "t_width": float(c["t_width"]),
                "class_logit": float(class_logits[i]),
            }
            b_metadatas.append(meta)

        embeddings_np = np.concatenate(b_embeddings, axis=0)

        t_up = time.perf_counter()
        ratt_db.upsert(
            embeddings=embeddings_np,
            ids=b_ids,
            metadatas=b_metadatas
        )
        ratt_db_relcls.upsert(
            embeddings=embeddings_np,
            ids=b_ids,
            metadatas=b_metadatas
        )
        t_up = time.perf_counter() - t_up

        print(f"[populate] upserted {end}/{n_chunks} chunks in {t_up:.2f}s")

        b_embeddings = []
        b_ids = []
        b_metadatas = []

    print(f"[done] total time: {time.perf_counter() - t0_all:.2f}s")


if __name__ == "__main__":
    main()
