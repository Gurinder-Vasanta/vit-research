from collections import defaultdict
import os
import pprint
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tf_keras
from chromadb import PersistentClient

# srun --jobid=614 --pty bash replace 614 with the job id and then nvidia-smi will work
from dataset import load_samples, build_chunks, build_tf_dataset_chunks
from models.vit_backbone import VisionTransformer
from models.ratt_head import RATTHead
from models.projection_head import ProjectionHead
from retrieval.ratt_chunk_retriever import RattChunkRetriever
from models.candidate_reranker import CandidateReranker
import config_stage2 as config
import time

from transformers import ViTModel, ViTImageProcessor
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle

from models.chunk_encoder import ChunkEncoder
from models.ratt_v2 import RATTHeadV2

import gc 

# support_reranker = CandidateReranker(...)
# contrast_reranker = CandidateReranker(...)
# temporal_reranker = CandidateReranker(...)

# ratt_head = RATTHeadV2(...)

tf.keras.backend.clear_session()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_processor.do_rescale = False
hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
hf_vit.eval()

layers = tf_keras.layers

SEED = 12

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

def make_chunk_key(chunk, precision=6):
    # pprint.pprint(chunk)
    # print(type(chunk))
    # _,meta,_ = chunk 
    # print(meta)
    return (
        int(chunk["vid"]),
        str(chunk["side"]),
        int(chunk["clip"]),
        # round(float(chunk["t_center"]), precision),
        int(chunk["start_idx"]),
        int(chunk["end_idx"]),
    )

def build_future_key_lookup(all_chunks, future_step=5):
    grouped = defaultdict(list)

    for chunk in all_chunks:
        grouped[(int(chunk["vid"]), int(chunk["clip"]))].append(chunk)

    future_key_lookup = {}

    for _, group in grouped.items():
        group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))
        last_idx = len(group_sorted) - 1

        for idx, chunk in enumerate(group_sorted):
            cur_key = make_chunk_key(chunk)
            fut_idx = min(idx + future_step, last_idx)
            fut_key = make_chunk_key(group_sorted[fut_idx])
            future_key_lookup[cur_key] = fut_key

    return future_key_lookup

CACHE_DIR = "./frame_cache_vit"

def get_store_paths(store_name):
    return {
        "emb": os.path.join(CACHE_DIR, f"{store_name}_emb.dat"),
        "paths": os.path.join(CACHE_DIR, f"{store_name}_paths.npy"),
        "meta": os.path.join(CACHE_DIR, f"{store_name}_meta.npz"),
    }

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

def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, 432, 768, 3) uint8 or float32
    Returns (N, 768) numpy embeddings (L2 normalized)
    """
    # frames_np = frames_np.astype(np.float32)

    # frames_np = frames_np.astype(np.float32)

    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)


    frames_np = frames_np.astype(np.float32)
        
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = hf_processor(images=frames_list, return_tensors="pt").to(device)
        out = hf_vit(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (N,768)
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)

def save_encoded_embeddings(encoded, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CACHE] Saved encoded embeddings to {path}")

def load_encoded_embeddings(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    print(f"[CACHE] Loaded encoded embeddings from {path}")
    return cache

def query_collection(query_emb, collection, n_results):
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n_results,
            include=["embeddings", "metadatas"]
        )

        raw_embs = results["embeddings"][0]
        raw_meta = results["metadatas"][0]

        out = []
        for emb, meta in zip(raw_embs, raw_meta):
            # pprint.pprint(meta)
            out.append(
                {
                    "emb": np.asarray(emb, dtype=np.float32),
                    "meta": {
                        "label": int(meta["label"]),
                        "side": str(meta["side"]),
                        "vid": int(meta["vid_num"]),
                        "clip": int(meta["clip_num"]),
                        "t_center": float(meta["t_center"]),
                        "t_width": float(meta["t_width"]),
                        "start_idx": int(meta["start_idx"]),
                        "end_idx": int(meta["end_idx"]),
                        "class_logit": float(meta["class_logit"])
                    },
                }
            )
        return out

def query_collection_batch(query_embs, collection, n_results):
    """
    query_embs: list of np.ndarray, each shape (D,)
    returns: list of candidate lists, one per query
    """
    results = collection.query(
        query_embeddings=[q.tolist() for q in query_embs],
        n_results=n_results,
        include=["embeddings", "metadatas"],
    )

    batch_out = []

    batch_raw_embs = results["embeddings"]     # list of length B
    batch_raw_meta = results["metadatas"]      # list of length B

    for raw_embs, raw_meta in zip(batch_raw_embs, batch_raw_meta):
        out = []
        for emb, meta in zip(raw_embs, raw_meta):
            out.append(
                {
                    "emb": np.asarray(emb, dtype=np.float32),
                    "meta": {
                        "label": int(meta["label"]),
                        "side": str(meta["side"]),
                        "vid": int(meta["vid_num"]),
                        "clip": int(meta["clip_num"]),
                        "t_center": float(meta["t_center"]),
                        "t_width": float(meta["t_width"]),
                        "start_idx": int(meta["start_idx"]),
                        "end_idx": int(meta["end_idx"]),
                        "class_logit": float(meta["class_logit"]),
                    },
                }
            )
        batch_out.append(out)

    return batch_out

# -----------------------------
# helpers
# -----------------------------

def extract_meta(chunk):
    return {
        "label": int(chunk["label"]),
        "side": str(chunk["side"]),
        "vid": int(chunk["vid"]),
        "clip": int(chunk["clip"]),
        "t_center": float(chunk["t_center"]),
        "t_width": float(chunk["t_width"]),
        "start_idx": int(chunk["start_idx"]),
        "end_idx": int(chunk["end_idx"]),
    }

def same_chunk_meta(meta_a, meta_b):
    return (
        int(meta_a["vid"]) == int(meta_b["vid"])
        and str(meta_a["side"]) == str(meta_b["side"])
        and int(meta_a["clip"]) == int(meta_b["clip"])
        and int(meta_a["start_idx"]) == int(meta_b["start_idx"])
        and int(meta_a["end_idx"]) == int(meta_b["end_idx"])
    )

def dedup_signature(meta):
    return (
        int(meta["vid"]),
        str(meta["side"]),
        int(meta["clip"]),
        int(meta["start_idx"]),
        int(meta["end_idx"]),
    )

def build_live_entry(
    chunk,
    future_chunk,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    search_k_content,
    search_k_temporal,
    k_sim,
    k_contrast,
    k_temporal,
):
    query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
    future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

    query_meta = extract_meta(chunk)
    future_meta = extract_meta(future_chunk)

    emb_dim = query_emb.shape[0]

    pad_meta_template = {
        "label": -1,
        "side": "PAD",
        "vid": -1,
        "clip": -1,
        "t_center": -1.0,
        "t_width": -1.0,
        "start_idx": -1,
        "end_idx": -1,
    }

    # -------------------------
    # content retrieval: sim + contrast
    # -------------------------
    content_candidates = query_collection(query_emb, collection, search_k_content)

    sim_items = []
    contrast_items = []
    used_content = set()

    for cand in content_candidates:
        cand_meta = cand["meta"]
        sig = dedup_signature(cand_meta)

        if same_chunk_meta(query_meta, cand_meta):
            continue
        if cand_meta["side"] != query_meta["side"]:
            continue

        if (
            cand_meta["label"] == query_meta["label"]
            and sig not in used_content
            and len(sim_items) < k_sim
        ):
            sim_items.append(cand)
            used_content.add(sig)
            continue

        if (
            cand_meta["label"] != query_meta["label"]
            and sig not in used_content
            and len(contrast_items) < k_contrast
        ):
            contrast_items.append(cand)
            used_content.add(sig)
            continue

        if len(sim_items) >= k_sim and len(contrast_items) >= k_contrast:
            break

    sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)
    contrast_embs, contrast_meta = pad_or_trim(
        contrast_items, k_contrast, emb_dim, pad_meta_template
    )

    # -------------------------
    # temporal retrieval
    # -------------------------
    temporal_candidates = query_collection(future_emb, collection, search_k_temporal)

    temporal_items = []
    seen_temporal = set()

    for cand in temporal_candidates:
        cand_meta = cand["meta"]
        sig = dedup_signature(cand_meta)

        if same_chunk_meta(query_meta, cand_meta):
            continue

        # optional: skip exact future anchor too
        # if same_chunk_meta(future_meta, cand_meta):
        #     continue

        if cand_meta["side"] != query_meta["side"]:
            continue
        if sig in seen_temporal:
            continue

        temporal_items.append(cand)
        seen_temporal.add(sig)

        if len(temporal_items) >= k_temporal:
            break

    temporal_embs, temporal_meta = pad_or_trim(
        temporal_items, k_temporal, emb_dim, pad_meta_template
    )

    return {
        "query_emb": query_emb,
        "sim_embs": sim_embs,
        "contrast_embs": contrast_embs,
        "temporal_embs": temporal_embs,
        "query_meta": query_meta,
        "future_meta": future_meta,
        "sim_meta": sim_meta,
        "contrast_meta": contrast_meta,
        "temporal_meta": temporal_meta,
    }

def encode_chunk(chunk, chunk_encoder, frame_emb_mm,path_to_idx):
    # pprint.pprint(chunk)
    idxs = [path_to_idx[p] for p in chunk["frames"]]          # length T
    frame_embs = frame_emb_mm[idxs].astype(np.float32)        # (T, 768)
    frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)  # (1, T, 768)

    stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)  # (1, 768)
    return stage1_chunk_emb[0].numpy().astype(np.float32)

def pad_or_trim(items, k, emb_dim, pad_meta_template):
    """
    items: list of {"emb": ..., "meta": ...}
    returns (embs, metas)
    """
    if len(items) >= k:
        items = items[:k]
    else:
        pad_count = k - len(items)
        zero_emb = np.zeros((emb_dim,), dtype=np.float32)
        for _ in range(pad_count):
            items.append(
                {
                    "emb": zero_emb.copy(),
                    "meta": dict(pad_meta_template),
                }
            )

    embs = np.stack([x["emb"] for x in items], axis=0)
    metas = [x["meta"] for x in items]
    return embs, metas

def build_retrieval_cache(
    all_chunks,
    collection,
    chunk_encoder,
    frame_emb_mm,
    frame_paths,
    path_to_idx
):
    

    # def encode_chunk(chunk):
    #     """
    #     Assumes chunk_encoder(chunk) returns a 1D embedding-like object.
    #     Adjust this wrapper if your chunk_encoder expects a different input format.
    #     """

    #     frames = []
    #     for fp in chunk["frames"]:
    #         img = tf.keras.utils.load_img(fp, target_size=(224, 224))
    #         img = tf.keras.utils.img_to_array(img)
    #         frames.append(img)

    #     frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    #     frame_embs = hf_vit_embed_batch(frames)  # (T, 768)
    #     frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)  # (1, T, 768)

    #     stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)  # (1, 768)

    #     out = stage1_chunk_emb[0].numpy()

    #     return out.astype(np.float32)
    

    # -----------------------------
    # precompute query embeddings
    # -----------------------------
    print("[CACHE] Encoding all chunk embeddings...")
    chunk_emb_lookup = {}
    meta_lookup = {}
    key_to_chunk = {}

    for i, chunk in enumerate(all_chunks):
        # print(chunk)
        key = make_chunk_key(chunk)
        chunk_emb_lookup[key] = encode_chunk(chunk, chunk_encoder, frame_emb_mm,path_to_idx)
        meta_lookup[key] = extract_meta(chunk)
        key_to_chunk[key] = chunk

        if (i + 1) % 25 == 0 or (i + 1) == len(all_chunks):
            print(f"[CACHE] encoded {i+1}/{len(all_chunks)}")
    # save_encoded_embeddings()
    # infer embedding dim
    first_key = next(iter(chunk_emb_lookup))
    emb_dim = chunk_emb_lookup[first_key].shape[0]

    # -----------------------------
    # build next-chunk lookup within each (vid, clip)
    # -----------------------------
    grouped = defaultdict(list)
    for chunk in all_chunks:
        grouped[(int(chunk["vid"]), int(chunk["clip"]))].append(chunk)

    # next_key_lookup = {}
    # for (_, _), group in grouped.items():
    #     group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))
    #     for idx, chunk in enumerate(group_sorted):
    #         cur_key = make_chunk_key(chunk)
    #         if idx < len(group_sorted) - 1:
    #             nxt_key = make_chunk_key(group_sorted[idx + 1])
    #             next_key_lookup[cur_key] = nxt_key
    #         else:
    #             next_key_lookup[cur_key] = None

    future_key_lookup = {}

    for (_, _), group in grouped.items():
        group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))

        for idx, chunk in enumerate(group_sorted):
            cur_key = make_chunk_key(chunk)

            future_idx = min(idx + config.FUTURE_CHUNK_STEP, len(group_sorted) - 1)
            future_key = make_chunk_key(group_sorted[future_idx])

            future_key_lookup[cur_key] = future_key

    # -----------------------------
    # build cache
    # -----------------------------

    # print("[CACHE] Building retrieval cache...")
    # cache = {}

    # pad_meta_template = {
    #     "label": -1,
    #     "side": "PAD",
    #     "vid": -1,
    #     "clip": -1,
    #     "t_center": -1.0,
    #     "t_width": -1.0,
    #     "start_idx": -1,
    #     "end_idx": -1,
    # }

    # QUERY_BATCH_SIZE = 16   # try 64 / 128 / 256

    # for batch_start in range(0, len(all_chunks), QUERY_BATCH_SIZE):
    #     batch_chunks = all_chunks[batch_start: batch_start + QUERY_BATCH_SIZE]

    #     batch_keys = []
    #     batch_query_embs = []
    #     batch_future_embs = []
    #     batch_query_meta = []

    #     # -------------------------
    #     # gather this batch's query/future data
    #     # -------------------------
    #     for chunk in batch_chunks:
    #         key = make_chunk_key(chunk)
    #         query_emb = chunk_emb_lookup[key]
    #         query_meta = meta_lookup[key]

    #         next_key = future_key_lookup[key]
    #         if next_key is None:
    #             future_emb = np.zeros_like(query_emb)
    #         else:
    #             future_emb = chunk_emb_lookup[next_key]

    #         batch_keys.append(key)
    #         batch_query_embs.append(query_emb)
    #         batch_future_embs.append(future_emb)
    #         batch_query_meta.append(query_meta)

    #     # -------------------------
    #     # do batched ANN queries
    #     # -------------------------
    #     batch_content_candidates = query_collection_batch(
    #         batch_query_embs, collection, config.SEARCH_K_CONTENT
    #     )

    #     batch_temporal_candidates = query_collection_batch(
    #         batch_future_embs, collection, config.SEARCH_K_TEMPORAL
    #     )

    #     # -------------------------
    #     # per-item filtering / packing
    #     # -------------------------
    #     for j, key in enumerate(batch_keys):
    #         query_emb = batch_query_embs[j]
    #         future_emb = batch_future_embs[j]
    #         query_meta = batch_query_meta[j]

    #         # ----- content: sim + contrast
    #         content_candidates = batch_content_candidates[j]

    #         sim_items = []
    #         contrast_items = []
    #         seen_sim = set()
    #         seen_contrast = set()

    #         for cand in content_candidates:
    #             cand_meta = cand["meta"]

    #             if same_chunk_meta(query_meta, cand_meta):
    #                 continue

    #             if cand_meta["side"] != query_meta["side"]:
    #                 continue

    #             sig = dedup_signature(cand_meta)

    #             if (
    #                 cand_meta["label"] == query_meta["label"]
    #                 and sig not in seen_sim
    #                 and len(sim_items) < config.K_SIM
    #             ):
    #                 sim_items.append(cand)
    #                 seen_sim.add(sig)

    #             if (
    #                 cand_meta["label"] != query_meta["label"]
    #                 and sig not in seen_contrast
    #                 and len(contrast_items) < config.K_CONTRAST
    #             ):
    #                 contrast_items.append(cand)
    #                 seen_contrast.add(sig)

    #             if len(sim_items) >= config.K_SIM and len(contrast_items) >= config.K_CONTRAST:
    #                 break

    #         sim_embs, sim_meta = pad_or_trim(
    #             sim_items, config.K_SIM, emb_dim, pad_meta_template
    #         )
    #         contrast_embs, contrast_meta = pad_or_trim(
    #             contrast_items, config.K_CONTRAST, emb_dim, pad_meta_template
    #         )

    #         # ----- temporal
    #         temporal_candidates = batch_temporal_candidates[j]

    #         temporal_items = []
    #         seen_temporal = set()

    #         for cand in temporal_candidates:
    #             cand_meta = cand["meta"]

    #             if same_chunk_meta(query_meta, cand_meta):
    #                 continue

    #             if cand_meta["side"] != query_meta["side"]:
    #                 continue

    #             sig = dedup_signature(cand_meta)
    #             if sig in seen_temporal:
    #                 continue

    #             temporal_items.append(cand)
    #             seen_temporal.add(sig)

    #             if len(temporal_items) >= config.K_TEMPORAL:
    #                 break

    #         temporal_embs, temporal_meta = pad_or_trim(
    #             temporal_items, config.K_TEMPORAL, emb_dim, pad_meta_template
    #         )

    #         # ----- save entry
    #         cache[key] = {
    #             "query_emb": query_emb,
    #             "future_emb": future_emb,
    #             "query_meta": query_meta,

    #             "sim_embs": sim_embs,
    #             "sim_meta": sim_meta,

    #             "contrast_embs": contrast_embs,
    #             "contrast_meta": contrast_meta,

    #             "temporal_embs": temporal_embs,
    #             "temporal_meta": temporal_meta,
    #         }

    #     built_so_far = min(batch_start + QUERY_BATCH_SIZE, len(all_chunks))
    #     print(f"[CACHE] built {built_so_far}/{len(all_chunks)}")

    #     if built_so_far % 1000 == 0 or built_so_far == len(all_chunks):
    #         print("[CACHE] saving cache checkpoint")
    #         save_retrieval_cache(cache, config.STAGE2_CACHE_PATH)
    print("[CACHE] Building retrieval cache...")
    cache = {}

    pad_meta_template = {
        "label": -1,
        "side": "PAD",
        "vid": -1,
        "clip": -1,
        "t_center": -1.0,
        "t_width": -1.0,
        "start_idx": -1,
        "end_idx": -1,
    }

    for i, chunk in enumerate(all_chunks):
        key = make_chunk_key(chunk)
        query_emb = chunk_emb_lookup[key]
        query_meta = meta_lookup[key]

        # -------------------------
        # future_emb = literal next chunk in same (vid, clip)
        # -------------------------
        next_key = future_key_lookup[key]
        if next_key is None:
            future_emb = np.zeros_like(query_emb)
        else:
            future_emb = chunk_emb_lookup[next_key]

        # -------------------------
        # content query: sim + contrast
        # -------------------------
        content_candidates = query_collection(query_emb, collection, config.SEARCH_K_CONTENT)

        sim_items = []
        contrast_items = []

        seen_sim = set()
        seen_contrast = set()

        for cand in content_candidates:
            cand_meta = cand["meta"]

            # skip exact self
            if same_chunk_meta(query_meta, cand_meta):
                continue

            # same side only
            if cand_meta["side"] != query_meta["side"]:
                continue
            
            sig = dedup_signature(cand_meta)

            # SIM
            if (
                cand_meta['label'] == query_meta['label']
                and sig not in seen_sim 
                and len(sim_items) < config.K_SIM
                ):
                sim_items.append(cand)
                seen_sim.add(sig)

            # CONTRAST
            if (
                cand_meta["label"] != query_meta["label"]
                and sig not in seen_contrast
                and len(contrast_items) < config.K_CONTRAST
            ):
                contrast_items.append(cand)
                seen_contrast.add(sig)

            if len(sim_items) >= config.K_SIM and len(contrast_items) >= config.K_CONTRAST:
                break

        sim_embs, sim_meta = pad_or_trim(sim_items, config.K_SIM, emb_dim, pad_meta_template)
        contrast_embs, contrast_meta = pad_or_trim(
            contrast_items, config.K_CONTRAST, emb_dim, pad_meta_template
        )

        # -------------------------
        # temporal query: use future_emb
        # -------------------------
        temporal_candidates = query_collection(future_emb, collection, config.SEARCH_K_TEMPORAL)

        temporal_items = []
        seen_temporal = set()

        for cand in temporal_candidates:
            cand_meta = cand["meta"]

            # skip exact self
            if same_chunk_meta(query_meta, cand_meta):
                continue

            # same side only
            if cand_meta["side"] != query_meta["side"]:
                continue

            sig = dedup_signature(cand_meta)
            if sig in seen_temporal:
                continue

            temporal_items.append(cand)
            seen_temporal.add(sig)

            if len(temporal_items) >= config.K_TEMPORAL:
                break

        temporal_embs, temporal_meta = pad_or_trim(
            temporal_items, config.K_TEMPORAL, emb_dim, pad_meta_template
        )

        # -------------------------
        # save entry
        # -------------------------
        cache[key] = {
            "query_emb": query_emb,
            "future_emb": future_emb,
            "query_meta": query_meta,

            "sim_embs": sim_embs,
            "sim_meta": sim_meta,

            "contrast_embs": contrast_embs,
            "contrast_meta": contrast_meta,

            "temporal_embs": temporal_embs,
            "temporal_meta": temporal_meta,
        }

        if (i + 1) % 10 == 0 or (i + 1) == len(all_chunks):
            print(f"[CACHE] built {i+1}/{len(all_chunks)}")
        if (i + 1) % 100 == 0:
            print(f"[CACHE] saving cache checkpoint")
            save_retrieval_cache(cache,config.STAGE2_CACHE_PATH)
    return cache

def save_retrieval_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CACHE] Saved retrieval cache to {path}")

def load_retrieval_cache(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    print(f"[CACHE] Loaded retrieval cache from {path}")
    return cache    

def print_random_cache_queries():
    r_inds = [random.randint(1,len(cache.keys())) for i in range(10)]
    for ind in r_inds: 
        entry = cache[list(cache.keys())[ind]]
        print('-----------------')
        print("QUERY")
        print(entry["query_meta"])

        print("\nSIM")
        for m in entry["sim_meta"][:5]:
            print(m)

        print("\nCONTRAST")
        for m in entry["contrast_meta"][:5]:
            print(m)

        print("\nTEMPORAL")
        for m in entry["temporal_meta"][:5]:
            print(m)

def _to_py_scalar(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            x = x.item()
        else:
            raise ValueError(f"Expected scalar, got array shape {x.shape}")
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return x

def make_chunk_key_from_meta(metadata, i, precision=6):
    vid = _to_py_scalar(metadata["vid"][i])
    side = _to_py_scalar(metadata["side"][i])
    clip = _to_py_scalar(metadata["clip"][i])
    t_center = _to_py_scalar(metadata["t_center"][i])
    start_idx = _to_py_scalar(metadata["start_idx"][i])
    end_idx = _to_py_scalar(metadata["end_idx"][i])

    return (
        int(vid),
        str(side),
        int(clip),
        # round(float(t_center), precision),
        int(start_idx),
        int(end_idx),
    )

def fetch_cache_batch(metadata, cache):
    batch_size = metadata["vid"].shape[0]

    query_embs = []
    support_tokens = []
    contrast_tokens = []
    temporal_tokens = []
    labels_from_cache = []

    for i in range(batch_size):
        key = make_chunk_key_from_meta(metadata, i)
        entry = cache[key]

        query_embs.append(entry["query_emb"])
        support_tokens.append(entry["sim_embs"])
        contrast_tokens.append(entry["contrast_embs"])
        temporal_tokens.append(entry["temporal_embs"])
        labels_from_cache.append(entry["query_meta"]["label"])

    query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)         # (B, D)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32) # (B, Ks, D)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32) # (B, Kc, D)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32) # (B, Kt, D)

    return query_embs, support_tokens, contrast_tokens, temporal_tokens

def fetch_live_batch(
    metadata,
    chunk_lookup,
    future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx
):
    batch_size = metadata["vid"].shape[0]

    query_embs = []
    support_tokens = []
    contrast_tokens = []
    temporal_tokens = []

    for i in range(batch_size):
        key = make_chunk_key_from_meta(metadata, i)

        chunk = chunk_lookup[key]
        future_key = future_key_lookup[key]
        future_chunk = chunk_lookup[future_key]

        entry = build_live_entry(
            chunk=chunk,
            future_chunk=future_chunk,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx,
            search_k_content=config.SEARCH_K_CONTENT,
            search_k_temporal=config.SEARCH_K_TEMPORAL,
            k_sim=config.K_SIM,
            k_contrast=config.K_CONTRAST,
            k_temporal=config.K_TEMPORAL,
        )

        query_embs.append(entry["query_emb"])
        support_tokens.append(entry["sim_embs"])
        contrast_tokens.append(entry["contrast_embs"])
        temporal_tokens.append(entry["temporal_embs"])

    query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

    return query_embs, support_tokens, contrast_tokens, temporal_tokens

def weighted_bce_with_logits(labels, logits, pos_weight):
    """
    labels: (B, 1) float32 in {0,1}
    logits: (B, 1) raw logits
    """
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        pos_weight=pos_weight,
    )
    return tf.reduce_mean(loss)

# def train_step(batch, cache, ratt_head, optimizer,pos_weight):
#     metadata, labels = batch[1], batch[2]   # assuming dataset yields (imgs, metadata, label)

#     labels = tf.cast(labels, tf.float32)
#     labels = tf.reshape(labels, (-1, 1))    # (B, 1)

#     query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_cache_batch(
#         metadata, cache
#     )
    
#     with tf.GradientTape() as tape:
#         class_logits, cls_out, aux = ratt_head(
#             chunk_embs=query_embs,
#             support_tokens=support_tokens,
#             contrast_tokens=contrast_tokens,
#             temporal_tokens=temporal_tokens,
#             training=True,
#         )

#         # loss = bce_loss_fn(labels, class_logits)
#         loss = weighted_bce_with_logits(labels, class_logits, pos_weight=pos_weight)

#         # optional regularization losses from model layers
#         if ratt_head.losses:
#             loss += tf.add_n(ratt_head.losses)

#     grads = tape.gradient(loss, ratt_head.trainable_variables)
#     optimizer.apply_gradients(zip(grads, ratt_head.trainable_variables))

#     probs = tf.sigmoid(class_logits)

#     train_loss_metric.update_state(loss)
#     train_acc_metric.update_state(labels, probs)

#     return {
#         "loss": float(loss.numpy()),
#         "acc": float(train_acc_metric.result().numpy()),
#         "logits": class_logits,
#         "probs": probs,
#         "cls_out": cls_out,
#         "aux": aux,
#     }

def train_step(batch, cache, ratt_head, optimizer, pos_weight):
    metadata, labels = batch[1], batch[2]

    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))

    query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_cache_batch(
        metadata, cache
    )

    def grad_rms(g):
        if g is None:
            return 0.0
        g = tf.cast(g, tf.float32)
        return float(tf.sqrt(tf.reduce_mean(tf.square(g))).numpy())

    # q_exp = tf.expand_dims(query_embs, axis=1)   # (B, 1, D)

    # # Make branches explicitly different
    # support_in = support_tokens
    # # contrast_in = contrast_tokens - q_exp
    # contrast_in = contrast_tokens + (contrast_tokens - q_exp)
    # temporal_in = temporal_tokens

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(query_embs)
        tape.watch(support_tokens)
        tape.watch(contrast_tokens)
        tape.watch(temporal_tokens)


        class_logits, cls_out, aux = ratt_head(
            chunk_embs=query_embs,
            support_tokens=support_tokens,
            contrast_tokens=contrast_tokens,
            temporal_tokens=temporal_tokens,
            training=True,
        )
        
        loss = weighted_bce_with_logits(labels, class_logits, pos_weight=pos_weight)

        if ratt_head.losses:
            loss += tf.add_n(ratt_head.losses)

    grads = tape.gradient(loss, ratt_head.trainable_variables)
    optimizer.apply_gradients(zip(grads, ratt_head.trainable_variables))

    

    g_query = tape.gradient(loss, query_embs)
    g_support = tape.gradient(loss, support_tokens)
    g_contrast = tape.gradient(loss, contrast_tokens)
    g_temporal = tape.gradient(loss, temporal_tokens)

    print(
        f"branch_grad_rms | "
        f"query={grad_rms(g_query):.6f} "
        f"support={grad_rms(g_support):.6f} "
        f"contrast={grad_rms(g_contrast):.6f} "
        f"temporal={grad_rms(g_temporal):.6f}"
    )

    del tape

    probs = tf.sigmoid(class_logits)

    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(labels, probs)

    return {
        "loss": float(loss.numpy()),
        "acc": float(train_acc_metric.result().numpy()),
        "logits": class_logits,
        "probs": probs,
        "cls_out": cls_out,
        "aux": aux,
    }

def eval_step(
    batch,
    ratt_head,
    val_chunk_lookup,
    val_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    pos_weight
):
    metadata, labels = batch[1], batch[2]

    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))

    query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
        metadata=metadata,
        chunk_lookup=val_chunk_lookup,
        future_key_lookup=val_future_key_lookup,
        collection=collection,
        chunk_encoder=chunk_encoder,
        frame_emb_mm=frame_emb_mm,
        path_to_idx=path_to_idx,
    )

    zeros_support = tf.zeros_like(support_tokens)
    zeros_contrast = tf.zeros_like(contrast_tokens)
    zeros_temporal = tf.zeros_like(temporal_tokens)

    class_logits, cls_out, aux = ratt_head(
        chunk_embs=query_embs,
        support_tokens=support_tokens,
        contrast_tokens=contrast_tokens,
        temporal_tokens=temporal_tokens,
        # support_tokens=zeros_support,
        # contrast_tokens=zeros_contrast,
        # temporal_tokens=zeros_temporal,
        training=False,
    )

    # loss = bce_loss_fn(labels, class_logits)
    loss = weighted_bce_with_logits(labels, class_logits, pos_weight=pos_weight)
    if ratt_head.losses:
        loss += tf.add_n(ratt_head.losses)

    probs = tf.sigmoid(class_logits)

    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(labels, probs)

    return {
        "loss": float(loss.numpy()),
        "acc": float(val_acc_metric.result().numpy()),
        "logits": class_logits,
        "probs": probs,
        'labels':labels
        # "cls_out": cls_out,
        # "aux": aux,
    }


def run_train_epoch(train_ds, cache, ratt_head, optimizer, pos_weight):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()

    for step, batch in enumerate(train_ds):
        out = train_step(batch, cache, ratt_head, optimizer, pos_weight)

        if step % 1 == 0:
            print(
                f"[train] step={step} "
                f"loss={out['loss']:.4f} "
                f"acc={train_acc_metric.result().numpy():.4f}"
            )

    return (
        float(train_loss_metric.result().numpy()),
        float(train_acc_metric.result().numpy()),
    )


# def run_val_epoch(val_ds, cache, ratt_head):
#     val_loss_metric.reset_state()
#     val_acc_metric.reset_state()

#     for step, batch in enumerate(val_ds):
#         out = eval_step(batch, cache, ratt_head)

#         if step % 10 == 0:
#             print(
#                 f"[val] step={step} "
#                 f"loss={out['loss']:.4f} "
#                 f"acc={val_acc_metric.result().numpy():.4f}"
#             )

#     return (
#         float(val_loss_metric.result().numpy()),
#         float(val_acc_metric.result().numpy()),
#     )

def compute_pos_weight(chunk_samples):
    labels = np.array([int(c["label"]) for c in chunk_samples], dtype=np.int32)

    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    if num_pos == 0:
        raise ValueError("No positive examples found.")
    if num_neg == 0:
        raise ValueError("No negative examples found.")
    # print(float(num_neg / num_pos))
    print(np.sqrt(num_neg / num_pos))
    return float(np.sqrt(num_neg / num_pos))

def run_val_epoch(
    val_ds,
    ratt_head,
    val_chunk_lookup,
    val_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    pos_weight
):
    val_loss_metric.reset_state()
    val_acc_metric.reset_state()

    for step, batch in enumerate(val_ds):
        out = eval_step(
            batch=batch,
            ratt_head=ratt_head,
            val_chunk_lookup=val_chunk_lookup,
            val_future_key_lookup=val_future_key_lookup,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx,
            pos_weight=pos_weight
        )

        batch_preds = tf.cast(out['probs'] >= 0.5, tf.float32)
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(batch_preds, out['labels']), tf.float32))
        if step % 1 == 0:
            print(
                f"[val] step={step} "
                f"loss={out['loss']:.4f} "
                f"running acc={val_acc_metric.result().numpy():.4f} "
                f"batch acc={batch_acc:.4f} "
            )
            temp = pd.DataFrame()
            
            temp['labels'] = out['labels'].numpy().flatten()
            temp['logits'] = out['logits'].numpy().flatten()
            temp['probs'] = out['probs'].numpy().flatten()
            print(temp)
            # pprint.pprint(out)


    return (
        float(val_loss_metric.result().numpy()),
        float(val_acc_metric.result().numpy()),
    )

if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = config.TRAIN_VIDS
    train_samples = load_samples(train_vids,stride=1)
    train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE)

    # label_lookup = {}
    # for c in train_chunk_samples:
    #     key = make_key(c["vid"], c["side"], c["t_center"])
    #     label_lookup[key] = int(c["label"])

    test_vids = config.TEST_VIDS
    test_samples = load_samples(test_vids,stride=1)
    test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE)

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # split 95/5
    # n = len(chunk_samples)
    print(len(train_chunk_samples))
    print(len(test_chunk_samples))

    pos_weight = compute_pos_weight(train_chunk_samples)
    # input('stop')
    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    # # train_chunks = chunk_samples[config.START_CHUNK_TRAIN:config.END_CHUNK_TRAIN] #was 2000
    # # val_chunks = chunk_samples[config.START_CHUNK_VALID:config.END_CHUNK_VALID]

    # # train_chunk_samples = train_chunk_samples[0:100]
    # # test_chunk_samples = test_chunk_samples[0:32]
    print(f"Train chunks: {len(train_chunk_samples)}")
    print(f"Val chunks:   {len(test_chunk_samples)}")

    # ---------------------------------------------
    # 2. Build TF datasets
    # ---------------------------------------------
    train_dataset = build_tf_dataset_chunks(train_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=True)
    val_dataset   = build_tf_dataset_chunks(test_chunk_samples,   batch_size=config.CHUNK_BATCH_SIZE, training=False)

    print(f"Train dataset: {(train_dataset)}")
    print(f"Val dataset:   {(val_dataset)}")

    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name=config.CHROMADB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    chunk_encoder = ChunkEncoder(
        hidden_size=768,
        num_layers=4,
        num_heads=8,
        max_frames=config.CHUNK_SIZE
    )

    dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
    _ = chunk_encoder(dummy_frame_embs, training=False)

    chunk_encoder.load_weights(config.STAGE1_WEIGHTS)
    print("[STAGE1] Loaded chunk encoder weights")

    chunk_encoder.trainable = False
    print("[STAGE1] Chunk encoder frozen")

    store_name = 'train_val_frames_chunk12_stride4'
    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)
    if(os.path.exists(config.STAGE2_CACHE_PATH)):
        cache = load_retrieval_cache(config.STAGE2_CACHE_PATH)
        # print(cache)
        print("[CACHE] loaded cache")
    else:
        cache = build_retrieval_cache(
            all_chunks=train_chunk_samples,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            frame_paths=frame_paths,
            path_to_idx=path_to_idx
            )
        save_retrieval_cache(cache,config.STAGE2_CACHE_PATH)
    
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="train_acc")

    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="val_acc")

    batch = next(iter(train_dataset))

    ratt_head = RATTHeadV2(
        hidden_size=768,
        num_heads=8,
        num_layers=2,
    )

    for v in ratt_head.trainable_variables[:5]:
        print(v.name, float(tf.reduce_mean(v).numpy()), float(tf.math.reduce_std(v).numpy()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    query_embs, support, contrast, temporal = fetch_cache_batch(batch[1], cache)

    val_chunk_lookup = {make_chunk_key(c): c for c in test_chunk_samples}
    val_future_key_lookup = build_future_key_lookup(test_chunk_samples, future_step=5)

    
    print(query_embs.shape)
    print(support.shape)
    print(contrast.shape)
    print(temporal.shape)

    logits, _, _ = ratt_head(
        chunk_embs=query_embs,
        support_tokens=support,
        contrast_tokens=contrast,
        temporal_tokens=temporal,
        training=False,
    )
    print(logits)
    print("logits:", logits.shape)
    print_random_cache_queries()
    for epoch in range(config.EPOCHS):
        print(f"\n===== EPOCH {epoch+1}/{config.EPOCHS} =====")

        train_loss, train_acc = run_train_epoch(
            train_ds=train_dataset,
            cache=cache,
            ratt_head=ratt_head,
            optimizer=optimizer,
            pos_weight=pos_weight,
        )

        val_loss, val_acc = run_val_epoch(
            val_ds=val_dataset,
            ratt_head=ratt_head,
            val_chunk_lookup=val_chunk_lookup,
            val_future_key_lookup=val_future_key_lookup,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx,
            pos_weight=pos_weight
        )

        print(
            f"[epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # -------------------------
    # save weights
    # -------------------------
    ratt_head.save_weights(config.RATT_WEIGHTS)
    print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")


    # print(len(cache))
    # pprint.pprint(list(cache.keys()))

    



    # print(cache[list(cache.keys())[1438]])
    # print(cache)
    # # ---------------------------------------------
    # # 3. Build models
    # # ---------------------------------------------
    
    

    # # proj_head.load_weights("projection_head.weights.h5")
    
    # # Retrieval DB
    # client = PersistentClient(path="./chroma_store")
    # collection = client.get_or_create_collection(
    #     name=config.CHROMADB_COLLECTION,
    #     metadata={"hnsw:space": "cosine"}
    # )
