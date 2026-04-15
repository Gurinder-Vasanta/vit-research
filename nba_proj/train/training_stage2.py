
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
from dataset import load_samples, build_chunks, build_tf_dataset_chunks, oversample_chunk_samples
from models.vit_backbone import VisionTransformer
from models.ratt_head import RATTHead
from models.projection_head import ProjectionHead
from retrieval.ratt_chunk_retriever import RattChunkRetriever
from models.candidate_reranker import CandidateReranker
# from db_maintainence.db_rebuild_chunk import rebuild_db
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

# def query_collection(query_emb, collection, n_results, ):
#         results = collection.query(
#             query_embeddings=[query_emb.tolist()],
#             n_results=n_results,
#             include=["embeddings", "metadatas"]
#         )

#         raw_embs = results["embeddings"][0]
#         raw_meta = results["metadatas"][0]

#         out = []
#         for emb, meta in zip(raw_embs, raw_meta):
#             # pprint.pprint(meta)
#             out.append(
#                 {
#                     "emb": np.asarray(emb, dtype=np.float32),
#                     "meta": {
#                         "label": int(meta["label"]),
#                         "status": str(meta["status"]),
#                         "status_id": int(meta["status_id"]),
#                         "side": str(meta["side"]),
#                         "vid": int(meta["vid_num"]),
#                         "clip": int(meta["clip_num"]),
#                         "t_center": float(meta["t_center"]),
#                         "t_width": float(meta["t_width"]),
#                         "start_idx": int(meta["start_idx"]),
#                         "end_idx": int(meta["end_idx"]),
#                         "class_logit": float(meta["class_logit"])
#                     },
#                 }
#             )
#         return out


def query_collection(
    query_emb,
    collection,
    n_results,
    side=None,
    target_status_ids=None,
):
    where_clauses = []

    if side is not None:
        where_clauses.append({"side": str(side)})

    if target_status_ids is not None:
        target_status_ids = [int(s) for s in target_status_ids]
        if len(target_status_ids) == 1:
            where_clauses.append({"status_id": target_status_ids[0]})
        else:
            where_clauses.append({
                "$or": [{"status_id": s} for s in target_status_ids]
            })

    if len(where_clauses) == 0:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    query_kwargs = {
        "query_embeddings": [query_emb.tolist()],
        "n_results": n_results,
        "include": ["embeddings", "metadatas"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    raw_embs = results["embeddings"][0]
    raw_meta = results["metadatas"][0]

    out = []
    for emb, meta in zip(raw_embs, raw_meta):
        out.append(
            {
                "emb": np.asarray(emb, dtype=np.float32),
                "meta": {
                    "label": int(meta["label"]),
                    "status": str(meta["status"]),
                    "status_id": int(meta["status_id"]),
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
    return out


def dedup_and_remove_self(candidates, query_meta):
    out = []
    seen = set()

    for cand in candidates:
        cand_meta = cand["meta"]
        sig = dedup_signature(cand_meta)

        if same_chunk_meta(query_meta, cand_meta):
            continue
        if sig in seen:
            continue

        seen.add(sig)
        out.append(cand)

    return out


def interleave_lists(a, b, n_total):
    out = []
    ia = ib = 0

    while len(out) < n_total and (ia < len(a) or ib < len(b)):
        if ia < len(a):
            out.append(a[ia])
            ia += 1
            if len(out) >= n_total:
                break
        if ib < len(b):
            out.append(b[ib])
            ib += 1

    return out


def take_unique(items, k):
    out = []
    seen = set()

    for cand in items:
        sig = dedup_signature(cand["meta"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= k:
            break

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
        "status": str(chunk["status"]),
        "status_id": int(chunk["status_id"]),
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

def get_branch_mix_counts(epoch_idx, k, training):
    """
    Returns:
        n_ideal: number of oracle tokens
        n_noisy: number of corrupted tokens
        use_unrestricted_final: if True, stop using label-based routing entirely
    """
    if not training or epoch_idx is None:
        return k, 0, False   # full oracle for validation/debug

    # curriculum:
    # 0-1   : 100% ideal
    # 2-3   : 80% ideal / 20% noisy
    # 4-5   : 60% ideal / 40% noisy
    # 6-7   : 50% ideal / 50% noisy
    # 8-9   : 20% ideal / 80% noisy
    # 10+   : unrestricted retrieval
    if epoch_idx < 2:
        return k, 0, False
    elif epoch_idx < 4:
        n_ideal = int(round(0.8 * k))
        return n_ideal, k - n_ideal, False
    elif epoch_idx < 6:
        n_ideal = int(round(0.6 * k))
        return n_ideal, k - n_ideal, False
    elif epoch_idx < 8:
        n_ideal = int(round(0.5 * k))
        return n_ideal, k - n_ideal, False
    elif epoch_idx < 10:
        n_ideal = int(round(0.2 * k))
        return n_ideal, k - n_ideal, False
    else:
        return 0, k, True


def query_collection_batch(
    query_embs,
    collection,
    n_results,
    side=None,
    target_status_ids=None,
):
    where_clauses = []

    if side is not None:
        where_clauses.append({"side": str(side)})

    if target_status_ids is not None:
        target_status_ids = [int(s) for s in target_status_ids]
        if len(target_status_ids) == 1:
            where_clauses.append({"status_id": target_status_ids[0]})
        else:
            where_clauses.append({
                "$or": [{"status_id": s} for s in target_status_ids]
            })

    if len(where_clauses) == 0:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    query_kwargs = {
        "query_embeddings": [q.tolist() for q in query_embs],
        "n_results": n_results,
        "include": ["embeddings", "metadatas"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    batch_out = []
    batch_raw_embs = results["embeddings"]
    batch_raw_meta = results["metadatas"]

    for raw_embs, raw_meta in zip(batch_raw_embs, batch_raw_meta):
        out = []
        for emb, meta in zip(raw_embs, raw_meta):
            out.append(
                {
                    "emb": np.asarray(emb, dtype=np.float32),
                    "meta": {
                        "label": int(meta["label"]),
                        "status": str(meta["status"]),
                        "status_id": int(meta["status_id"]),
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


# def assemble_live_entry_from_candidates(
#     query_emb,
#     future_emb,
#     query_meta,
#     future_meta,
#     support_oracle,
#     contrast_oracle,
#     unrestricted_candidates,
#     temporal_candidates,
#     k_sim,
#     k_contrast,
#     k_temporal,
#     training=False,
#     epoch_idx=None,
# ):
#     q_status = int(query_meta["status_id"])
#     other_statuses = [s for s in (0, 1, 2) if s != q_status]
#     other_a, other_b = other_statuses[0], other_statuses[1]
#     emb_dim = query_emb.shape[0]

#     pad_meta_template = {
#         "label": -1,
#         "side": "PAD",
#         "status": "PAD",
#         "status_id": -1,
#         "vid": -1,
#         "clip": -1,
#         "t_center": -1.0,
#         "t_width": -1.0,
#         "start_idx": -1,
#         "end_idx": -1,
#     }

#     support_oracle = dedup_and_remove_self(support_oracle, query_meta)
#     contrast_oracle = dedup_and_remove_self(contrast_oracle, query_meta)
#     unrestricted_candidates = dedup_and_remove_self(unrestricted_candidates, query_meta)
#     temporal_candidates = dedup_and_remove_self(temporal_candidates, query_meta)

#     contrast_oracle_a = [c for c in contrast_oracle if int(c["meta"]["status_id"]) == other_a]
#     contrast_oracle_b = [c for c in contrast_oracle if int(c["meta"]["status_id"]) == other_b]

#     unrestricted_same = []
#     unrestricted_diff_a = []
#     unrestricted_diff_b = []

#     for cand in unrestricted_candidates:
#         s = int(cand["meta"]["status_id"])
#         if s == q_status:
#             unrestricted_same.append(cand)
#         elif s == other_a:
#             unrestricted_diff_a.append(cand)
#         elif s == other_b:
#             unrestricted_diff_b.append(cand)

#     unrestricted_diff_balanced = interleave_lists(
#         unrestricted_diff_a,
#         unrestricted_diff_b,
#         n_total=len(unrestricted_diff_a) + len(unrestricted_diff_b),
#     )

#     sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_sim, training=training
#     )

#     if sim_unrestricted:
#         sim_items = take_unique(unrestricted_candidates, k_sim)
#     else:
#         sim_items = []
#         sim_items.extend(take_unique(support_oracle, sim_n_ideal))

#         used = {dedup_signature(x["meta"]) for x in sim_items}
#         for cand in unrestricted_diff_balanced:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             sim_items.append(cand)
#             used.add(sig)
#             if len(sim_items) >= sim_n_ideal + sim_n_noisy:
#                 break

#         if len(sim_items) < k_sim:
#             for pool in [support_oracle, unrestricted_candidates]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     sim_items.append(cand)
#                     used.add(sig)
#                     if len(sim_items) >= k_sim:
#                         break
#                 if len(sim_items) >= k_sim:
#                     break

#     sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)

#     con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_contrast, training=training
#     )

#     if con_unrestricted:
#         contrast_items = take_unique(unrestricted_diff_balanced, k_contrast)
#         if len(contrast_items) < k_contrast:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in unrestricted_candidates:
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= k_contrast:
#                     break
#     else:
#         contrast_items = []

#         n_a = con_n_ideal // 2
#         n_b = con_n_ideal - n_a
#         contrast_items.extend(take_unique(contrast_oracle_a, n_a))
#         contrast_items.extend(take_unique(contrast_oracle_b, n_b))

#         if len(contrast_items) < con_n_ideal:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9):
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= con_n_ideal:
#                     break

#         used = {dedup_signature(x["meta"]) for x in contrast_items}
#         for cand in unrestricted_same:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             contrast_items.append(cand)
#             used.add(sig)
#             if len(contrast_items) >= con_n_ideal + con_n_noisy:
#                 break

#         if len(contrast_items) < k_contrast:
#             for pool in [
#                 interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9),
#                 unrestricted_diff_balanced,
#                 unrestricted_candidates,
#             ]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     contrast_items.append(cand)
#                     used.add(sig)
#                     if len(contrast_items) >= k_contrast:
#                         break
#                 if len(contrast_items) >= k_contrast:
#                     break

#     contrast_embs, contrast_meta = pad_or_trim(
#         contrast_items, k_contrast, emb_dim, pad_meta_template
#     )

#     temporal_items = take_unique(temporal_candidates, k_temporal)
#     temporal_embs, temporal_meta = pad_or_trim(
#         temporal_items, k_temporal, emb_dim, pad_meta_template
#     )
#     print({'sim ideal count': sim_n_ideal, 'sim noise count': sim_n_noisy, 'con ideal count': con_n_ideal, 'con noise count': con_n_noisy})
#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "query_meta": query_meta,
#         "future_meta": future_meta,
#         "sim_meta": sim_meta,
#         "contrast_meta": contrast_meta,
#         "temporal_meta": temporal_meta,
#     }


# def assemble_live_entry_from_candidates(
#     query_emb,
#     future_emb,
#     query_meta,
#     future_meta,
#     support_oracle,
#     contrast_oracle,
#     unrestricted_candidates,
#     temporal_candidates,
#     k_sim,
#     k_contrast,
#     k_temporal,
#     training=False,
#     epoch_idx=None,
# ):
#     q_status = int(query_meta["status_id"])
#     other_statuses = [s for s in (0, 1, 2) if s != q_status]
#     other_a, other_b = other_statuses[0], other_statuses[1]
#     emb_dim = query_emb.shape[0]

#     pad_meta_template = {
#         "label": -1,
#         "side": "PAD",
#         "status": "PAD",
#         "status_id": -1,
#         "vid": -1,
#         "clip": -1,
#         "t_center": -1.0,
#         "t_width": -1.0,
#         "start_idx": -1,
#         "end_idx": -1,
#     }

#     support_oracle = [] if support_oracle is None else dedup_and_remove_self(support_oracle, query_meta)
#     contrast_oracle = [] if contrast_oracle is None else dedup_and_remove_self(contrast_oracle, query_meta)
#     unrestricted_candidates = [] if unrestricted_candidates is None else dedup_and_remove_self(unrestricted_candidates, query_meta)
#     temporal_candidates = [] if temporal_candidates is None else dedup_and_remove_self(temporal_candidates, query_meta)

#     contrast_oracle_a = [c for c in contrast_oracle if int(c["meta"]["status_id"]) == other_a]
#     contrast_oracle_b = [c for c in contrast_oracle if int(c["meta"]["status_id"]) == other_b]

#     unrestricted_same = []
#     unrestricted_diff_a = []
#     unrestricted_diff_b = []

#     for cand in unrestricted_candidates:
#         s = int(cand["meta"]["status_id"])
#         if s == q_status:
#             unrestricted_same.append(cand)
#         elif s == other_a:
#             unrestricted_diff_a.append(cand)
#         elif s == other_b:
#             unrestricted_diff_b.append(cand)

#     unrestricted_diff_balanced = interleave_lists(
#         unrestricted_diff_a,
#         unrestricted_diff_b,
#         n_total=len(unrestricted_diff_a) + len(unrestricted_diff_b),
#     )

#     sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_sim, training=training
#     )

#     if sim_unrestricted:
#         sim_items = take_unique(unrestricted_candidates, k_sim)
#     else:
#         sim_items = []
#         sim_items.extend(take_unique(support_oracle, sim_n_ideal))

#         used = {dedup_signature(x["meta"]) for x in sim_items}
#         for cand in unrestricted_diff_balanced:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             sim_items.append(cand)
#             used.add(sig)
#             if len(sim_items) >= sim_n_ideal + sim_n_noisy:
#                 break

#         if len(sim_items) < k_sim:
#             for pool in [support_oracle, unrestricted_candidates]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     sim_items.append(cand)
#                     used.add(sig)
#                     if len(sim_items) >= k_sim:
#                         break
#                 if len(sim_items) >= k_sim:
#                     break

#     sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)

#     con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_contrast, training=training
#     )

#     if con_unrestricted:
#         contrast_items = take_unique(unrestricted_diff_balanced, k_contrast)
#         if len(contrast_items) < k_contrast:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in unrestricted_candidates:
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= k_contrast:
#                     break
#     else:
#         contrast_items = []

#         n_a = con_n_ideal // 2
#         n_b = con_n_ideal - n_a
#         contrast_items.extend(take_unique(contrast_oracle_a, n_a))
#         contrast_items.extend(take_unique(contrast_oracle_b, n_b))

#         if len(contrast_items) < con_n_ideal:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9):
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= con_n_ideal:
#                     break

#         used = {dedup_signature(x["meta"]) for x in contrast_items}
#         for cand in unrestricted_same:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             contrast_items.append(cand)
#             used.add(sig)
#             if len(contrast_items) >= con_n_ideal + con_n_noisy:
#                 break

#         if len(contrast_items) < k_contrast:
#             for pool in [
#                 interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9),
#                 unrestricted_diff_balanced,
#                 unrestricted_candidates,
#             ]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     contrast_items.append(cand)
#                     used.add(sig)
#                     if len(contrast_items) >= k_contrast:
#                         break
#                 if len(contrast_items) >= k_contrast:
#                     break

#     contrast_embs, contrast_meta = pad_or_trim(
#         contrast_items, k_contrast, emb_dim, pad_meta_template
#     )

#     temporal_items = take_unique(temporal_candidates, k_temporal)
#     temporal_embs, temporal_meta = pad_or_trim(
#         temporal_items, k_temporal, emb_dim, pad_meta_template
#     )

#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "query_meta": query_meta,
#         "future_meta": future_meta,
#         "sim_meta": sim_meta,
#         "contrast_meta": contrast_meta,
#         "temporal_meta": temporal_meta,
#     }



def assemble_live_entry_from_candidates(
    query_emb,
    future_emb,
    query_meta,
    future_meta,
    oracle_candidates,
    unrestricted_candidates,
    temporal_candidates,
    k_sim,
    k_contrast,
    k_temporal,
    training=False,
    epoch_idx=None,
):
    q_status = int(query_meta["status_id"])
    other_statuses = [s for s in (0, 1, 2) if s != q_status]
    other_a, other_b = other_statuses[0], other_statuses[1]
    emb_dim = query_emb.shape[0]

    pad_meta_template = {
        "label": -1,
        "side": "PAD",
        "status": "PAD",
        "status_id": -1,
        "vid": -1,
        "clip": -1,
        "t_center": -1.0,
        "t_width": -1.0,
        "start_idx": -1,
        "end_idx": -1,
    }

    oracle_candidates = [] if oracle_candidates is None else dedup_and_remove_self(oracle_candidates, query_meta)
    unrestricted_candidates = [] if unrestricted_candidates is None else dedup_and_remove_self(unrestricted_candidates, query_meta)
    temporal_candidates = [] if temporal_candidates is None else dedup_and_remove_self(temporal_candidates, query_meta)

    # Split shared oracle pool locally
    support_oracle = [c for c in oracle_candidates if int(c["meta"]["status_id"]) == q_status]
    contrast_oracle_a = [c for c in oracle_candidates if int(c["meta"]["status_id"]) == other_a]
    contrast_oracle_b = [c for c in oracle_candidates if int(c["meta"]["status_id"]) == other_b]

    unrestricted_same = []
    unrestricted_diff_a = []
    unrestricted_diff_b = []

    for cand in unrestricted_candidates:
        s = int(cand["meta"]["status_id"])
        if s == q_status:
            unrestricted_same.append(cand)
        elif s == other_a:
            unrestricted_diff_a.append(cand)
        elif s == other_b:
            unrestricted_diff_b.append(cand)

    unrestricted_diff_balanced = interleave_lists(
        unrestricted_diff_a,
        unrestricted_diff_b,
        n_total=len(unrestricted_diff_a) + len(unrestricted_diff_b),
    )

    sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
        epoch_idx=epoch_idx, k=k_sim, training=training
    )

    if sim_unrestricted:
        sim_items = take_unique(unrestricted_candidates, k_sim)
    else:
        sim_items = []
        sim_items.extend(take_unique(support_oracle, sim_n_ideal))

        used = {dedup_signature(x["meta"]) for x in sim_items}
        for cand in unrestricted_diff_balanced:
            sig = dedup_signature(cand["meta"])
            if sig in used:
                continue
            sim_items.append(cand)
            used.add(sig)
            if len(sim_items) >= sim_n_ideal + sim_n_noisy:
                break

        if len(sim_items) < k_sim:
            for pool in [support_oracle, unrestricted_candidates]:
                for cand in pool:
                    sig = dedup_signature(cand["meta"])
                    if sig in used:
                        continue
                    sim_items.append(cand)
                    used.add(sig)
                    if len(sim_items) >= k_sim:
                        break
                if len(sim_items) >= k_sim:
                    break

    sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)

    con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
        epoch_idx=epoch_idx, k=k_contrast, training=training
    )

    if con_unrestricted:
        contrast_items = take_unique(unrestricted_diff_balanced, k_contrast)
        if len(contrast_items) < k_contrast:
            used = {dedup_signature(x["meta"]) for x in contrast_items}
            for cand in unrestricted_candidates:
                sig = dedup_signature(cand["meta"])
                if sig in used:
                    continue
                contrast_items.append(cand)
                used.add(sig)
                if len(contrast_items) >= k_contrast:
                    break
    else:
        contrast_items = []

        n_a = con_n_ideal // 2
        n_b = con_n_ideal - n_a
        contrast_items.extend(take_unique(contrast_oracle_a, n_a))
        contrast_items.extend(take_unique(contrast_oracle_b, n_b))

        if len(contrast_items) < con_n_ideal:
            used = {dedup_signature(x["meta"]) for x in contrast_items}
            for cand in interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9):
                sig = dedup_signature(cand["meta"])
                if sig in used:
                    continue
                contrast_items.append(cand)
                used.add(sig)
                if len(contrast_items) >= con_n_ideal:
                    break

        used = {dedup_signature(x["meta"]) for x in contrast_items}
        for cand in unrestricted_same:
            sig = dedup_signature(cand["meta"])
            if sig in used:
                continue
            contrast_items.append(cand)
            used.add(sig)
            if len(contrast_items) >= con_n_ideal + con_n_noisy:
                break

        if len(contrast_items) < k_contrast:
            for pool in [
                interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9),
                unrestricted_diff_balanced,
                unrestricted_candidates,
            ]:
                for cand in pool:
                    sig = dedup_signature(cand["meta"])
                    if sig in used:
                        continue
                    contrast_items.append(cand)
                    used.add(sig)
                    if len(contrast_items) >= k_contrast:
                        break
                if len(contrast_items) >= k_contrast:
                    break

    contrast_embs, contrast_meta = pad_or_trim(
        contrast_items, k_contrast, emb_dim, pad_meta_template
    )

    temporal_items = take_unique(temporal_candidates, k_temporal)
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


# def fetch_live_batch(
#     metadata,
#     chunk_lookup,
#     future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     training=False,
#     epoch_idx=None
# ):
#     batch_size = metadata["vid"].shape[0]

#     batch_query_meta = []
#     batch_future_meta = []
#     batch_query_embs_np = []
#     batch_future_embs_np = []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_chunk = chunk_lookup[future_key_lookup[key]]

#         query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#         future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

#         batch_query_meta.append(extract_meta(chunk))
#         batch_future_meta.append(extract_meta(future_chunk))
#         batch_query_embs_np.append(query_emb)
#         batch_future_embs_np.append(future_emb)

#     # Group content queries by (side, status_id)
#     content_groups = defaultdict(list)
#     for i, meta_i in enumerate(batch_query_meta):
#         content_groups[(str(meta_i["side"]), int(meta_i["status_id"]))].append(i)

#     support_results = [None] * batch_size
#     contrast_results = [None] * batch_size
#     unrestricted_results = [None] * batch_size

#     counter = 1
#     for (side, status_id), idxs in content_groups.items():
#         group_query_embs = [batch_query_embs_np[i] for i in idxs]
#         other_statuses = [s for s in (0, 1, 2) if s != status_id]

#         print('support')
#         print(f'{counter}/{len(content_groups.items())}')
#         counter += 1
#         group_support = query_collection_batch(
#             query_embs=group_query_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_CONTENT,
#             side=side,
#             target_status_ids=[status_id],
#         )

#         print('contrast')
#         group_contrast = query_collection_batch(
#             query_embs=group_query_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_CONTENT,
#             side=side,
#             target_status_ids=other_statuses,
#         )

#         print('restricted')
#         group_unrestricted = query_collection_batch(
#             query_embs=group_query_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_CONTENT,
#             side=side,
#             target_status_ids=None,
#         )

#         for local_j, global_i in enumerate(idxs):
#             support_results[global_i] = group_support[local_j]
#             contrast_results[global_i] = group_contrast[local_j]
#             unrestricted_results[global_i] = group_unrestricted[local_j]

#     # Group temporal queries by side only
#     temporal_groups = defaultdict(list)
#     for i, meta_i in enumerate(batch_query_meta):
#         temporal_groups[str(meta_i["side"])].append(i)

#     temporal_results = [None] * batch_size
#     for side, idxs in temporal_groups.items():
#         group_future_embs = [batch_future_embs_np[i] for i in idxs]

#         group_temporal = query_collection_batch(
#             query_embs=group_future_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_TEMPORAL,
#             side=side,
#             target_status_ids=None,
#         )

#         for local_j, global_i in enumerate(idxs):
#             temporal_results[global_i] = group_temporal[local_j]

#     query_embs = []
#     support_tokens = []
#     contrast_tokens = []
#     temporal_tokens = []

#     for i in range(batch_size):
#         entry = assemble_live_entry_from_candidates(
#             query_emb=batch_query_embs_np[i],
#             future_emb=batch_future_embs_np[i],
#             query_meta=batch_query_meta[i],
#             future_meta=batch_future_meta[i],
#             support_oracle=support_results[i],
#             contrast_oracle=contrast_results[i],
#             unrestricted_candidates=unrestricted_results[i],
#             temporal_candidates=temporal_results[i],
#             k_sim=config.K_SIM,
#             k_contrast=config.K_CONTRAST,
#             k_temporal=config.K_TEMPORAL,
#             training=training,
#             epoch_idx=epoch_idx,
#         )

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)
    
#     return query_embs, support_tokens, contrast_tokens, temporal_tokens




# def fetch_live_batch(
#     metadata,
#     chunk_lookup,
#     future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     training=False,
#     epoch_idx=None
# ):
#     batch_size = metadata["vid"].shape[0]

#     batch_query_meta = []
#     batch_future_meta = []
#     batch_query_embs_np = []
#     batch_future_embs_np = []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_chunk = chunk_lookup[future_key_lookup[key]]

#         query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#         future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

#         batch_query_meta.append(extract_meta(chunk))
#         batch_future_meta.append(extract_meta(future_chunk))
#         batch_query_embs_np.append(query_emb)
#         batch_future_embs_np.append(future_emb)

#     # curriculum needs
#     sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=config.K_SIM, training=training
#     )
#     con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=config.K_CONTRAST, training=training
#     )

#     need_support_oracle = (sim_n_ideal > 0) or (not sim_unrestricted)
#     need_contrast_oracle = (con_n_ideal > 0) or (not con_unrestricted)
#     need_unrestricted = sim_unrestricted or con_unrestricted or (sim_n_noisy > 0) or (con_n_noisy > 0)

#     search_k_support = getattr(config, "SEARCH_K_SUPPORT", config.SEARCH_K_CONTENT)
#     search_k_contrast = getattr(config, "SEARCH_K_CONTRAST", config.SEARCH_K_CONTENT)
#     search_k_unrestricted = getattr(config, "SEARCH_K_UNRESTRICTED", config.SEARCH_K_CONTENT)

#     # group content queries by (side, status_id)
#     content_groups = defaultdict(list)
#     for i, meta_i in enumerate(batch_query_meta):
#         content_groups[(str(meta_i["side"]), int(meta_i["status_id"]))].append(i)

#     support_results = [None] * batch_size
#     contrast_results = [None] * batch_size
#     unrestricted_results = [None] * batch_size

#     for (side, status_id), idxs in content_groups.items():
#         group_query_embs = [batch_query_embs_np[i] for i in idxs]
#         other_statuses = [s for s in (0, 1, 2) if s != status_id]

#         if need_support_oracle:
#             print('querying support')
#             group_support = query_collection_batch(
#                 query_embs=group_query_embs,
#                 collection=collection,
#                 n_results=search_k_support,
#                 side=side,
#                 target_status_ids=[status_id],
#             )
#             for local_j, global_i in enumerate(idxs):
#                 support_results[global_i] = group_support[local_j]

#         if need_contrast_oracle:
#             print('querying contrast')
#             group_contrast = query_collection_batch(
#                 query_embs=group_query_embs,
#                 collection=collection,
#                 n_results=search_k_contrast,
#                 side=side,
#                 target_status_ids=other_statuses,
#             )
#             for local_j, global_i in enumerate(idxs):
#                 contrast_results[global_i] = group_contrast[local_j]

#         if need_unrestricted:
#             print('querying unrestricted')
#             group_unrestricted = query_collection_batch(
#                 query_embs=group_query_embs,
#                 collection=collection,
#                 n_results=search_k_unrestricted,
#                 side=side,
#                 target_status_ids=None,
#             )
#             for local_j, global_i in enumerate(idxs):
#                 unrestricted_results[global_i] = group_unrestricted[local_j]

#     # temporal grouped only by side
#     temporal_groups = defaultdict(list)
#     for i, meta_i in enumerate(batch_query_meta):
#         temporal_groups[str(meta_i["side"])].append(i)

#     temporal_results = [None] * batch_size
#     for side, idxs in temporal_groups.items():
#         group_future_embs = [batch_future_embs_np[i] for i in idxs]

#         print('querying temporal')
#         group_temporal = query_collection_batch(
#             query_embs=group_future_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_TEMPORAL,
#             side=side,
#             target_status_ids=None,
#         )

#         for local_j, global_i in enumerate(idxs):
#             temporal_results[global_i] = group_temporal[local_j]

#     query_embs = []
#     support_tokens = []
#     contrast_tokens = []
#     temporal_tokens = []

#     for i in range(batch_size):
#         entry = assemble_live_entry_from_candidates(
#             query_emb=batch_query_embs_np[i],
#             future_emb=batch_future_embs_np[i],
#             query_meta=batch_query_meta[i],
#             future_meta=batch_future_meta[i],
#             support_oracle=support_results[i],
#             contrast_oracle=contrast_results[i],
#             unrestricted_candidates=unrestricted_results[i],
#             temporal_candidates=temporal_results[i],
#             k_sim=config.K_SIM,
#             k_contrast=config.K_CONTRAST,
#             k_temporal=config.K_TEMPORAL,
#             training=training,
#             epoch_idx=epoch_idx,
#         )

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     return query_embs, support_tokens, contrast_tokens, temporal_tokens




# def fetch_live_batch(
#     metadata,
#     chunk_lookup,
#     future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     training=False,
#     epoch_idx=None
# ):
#     batch_size = metadata["vid"].shape[0]

#     batch_query_meta = []
#     batch_future_meta = []
#     batch_query_embs_np = []
#     batch_future_embs_np = []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_chunk = chunk_lookup[future_key_lookup[key]]

#         query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#         future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

#         batch_query_meta.append(extract_meta(chunk))
#         batch_future_meta.append(extract_meta(future_chunk))
#         batch_query_embs_np.append(query_emb)
#         batch_future_embs_np.append(future_emb)

#     # curriculum needs
#     sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=config.K_SIM, training=training
#     )
#     con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=config.K_CONTRAST, training=training
#     )

#     need_oracle = (sim_n_ideal > 0) or (con_n_ideal > 0)
#     need_unrestricted = sim_unrestricted or con_unrestricted or (sim_n_noisy > 0) or (con_n_noisy > 0)

#     search_k_oracle = getattr(config, "SEARCH_K_ORACLE", config.SEARCH_K_CONTENT)
#     search_k_unrestricted = getattr(config, "SEARCH_K_UNRESTRICTED", config.SEARCH_K_CONTENT)

#     oracle_results = [None] * batch_size
#     unrestricted_results = [None] * batch_size
#     temporal_results = [None] * batch_size

#     # -------------------------------------------------
#     # ORACLE CONTENT: group by (side, status_id)
#     # one oracle query per group, not separate support/contrast
#     # -------------------------------------------------
#     if need_oracle:
#         print('querying oracle')
#         oracle_groups = defaultdict(list)
#         for i, meta_i in enumerate(batch_query_meta):
#             oracle_groups[(str(meta_i["side"]), int(meta_i["status_id"]))].append(i)

#         for (side, status_id), idxs in oracle_groups.items():
#             group_query_embs = [batch_query_embs_np[i] for i in idxs]

#             group_oracle = query_collection_batch(
#                 query_embs=group_query_embs,
#                 collection=collection,
#                 n_results=search_k_oracle,
#                 side=side,
#                 target_status_ids=[0, 1, 2],
#             )

#             for local_j, global_i in enumerate(idxs):
#                 oracle_results[global_i] = group_oracle[local_j]

#     # -------------------------------------------------
#     # UNRESTRICTED CONTENT: group ONLY by side
#     # -------------------------------------------------
#     if need_unrestricted:
#         print('querying unrestricted')
#         unrestricted_groups = defaultdict(list)
#         for i, meta_i in enumerate(batch_query_meta):
#             unrestricted_groups[str(meta_i["side"])].append(i)

#         for side, idxs in unrestricted_groups.items():
#             group_query_embs = [batch_query_embs_np[i] for i in idxs]

#             group_unrestricted = query_collection_batch(
#                 query_embs=group_query_embs,
#                 collection=collection,
#                 n_results=search_k_unrestricted,
#                 side=side,
#                 target_status_ids=None,
#             )

#             for local_j, global_i in enumerate(idxs):
#                 unrestricted_results[global_i] = group_unrestricted[local_j]

#     # -------------------------------------------------
#     # TEMPORAL: group ONLY by side
#     # -------------------------------------------------
#     temporal_groups = defaultdict(list)
#     for i, meta_i in enumerate(batch_query_meta):
#         temporal_groups[str(meta_i["side"])].append(i)

#     for side, idxs in temporal_groups.items():
#         group_future_embs = [batch_future_embs_np[i] for i in idxs]

#         group_temporal = query_collection_batch(
#             query_embs=group_future_embs,
#             collection=collection,
#             n_results=config.SEARCH_K_TEMPORAL,
#             side=side,
#             target_status_ids=None,
#         )

#         for local_j, global_i in enumerate(idxs):
#             temporal_results[global_i] = group_temporal[local_j]

#     query_embs = []
#     support_tokens = []
#     contrast_tokens = []
#     temporal_tokens = []

#     for i in range(batch_size):
#         entry = assemble_live_entry_from_candidates(
#             query_emb=batch_query_embs_np[i],
#             future_emb=batch_future_embs_np[i],
#             query_meta=batch_query_meta[i],
#             future_meta=batch_future_meta[i],
#             oracle_candidates=oracle_results[i],
#             unrestricted_candidates=unrestricted_results[i],
#             temporal_candidates=temporal_results[i],
#             k_sim=config.K_SIM,
#             k_contrast=config.K_CONTRAST,
#             k_temporal=config.K_TEMPORAL,
#             training=training,
#             epoch_idx=epoch_idx,
#         )

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     return query_embs, support_tokens, contrast_tokens, temporal_tokens



def make_padding_mask(tokens):
    """
    tokens: (B, K, D)
    returns: (B, K) bool mask, True = real token, False = pad
    """
    norms = tf.norm(tokens, axis=-1)  # (B, K)
    return norms > 1e-6               # (B, K)

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

CHROMA_SUBBATCH_SIZE = 8

def fetch_live_batch(
    metadata,
    chunk_lookup,
    future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    training=False,
    epoch_idx=None
):
    batch_size = metadata["vid"].shape[0]

    batch_query_meta = []
    batch_future_meta = []
    batch_query_embs_np = []
    batch_future_embs_np = []

    for i in range(batch_size):
        key = make_chunk_key_from_meta(metadata, i)
        chunk = chunk_lookup[key]
        future_chunk = chunk_lookup[future_key_lookup[key]]

        query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
        future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

        batch_query_meta.append(extract_meta(chunk))
        batch_future_meta.append(extract_meta(future_chunk))
        batch_query_embs_np.append(query_emb)
        batch_future_embs_np.append(future_emb)

    sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
        epoch_idx=epoch_idx, k=config.K_SIM, training=training
    )
    con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
        epoch_idx=epoch_idx, k=config.K_CONTRAST, training=training
    )

    need_oracle = (sim_n_ideal > 0) or (con_n_ideal > 0)
    need_unrestricted = sim_unrestricted or con_unrestricted or (sim_n_noisy > 0) or (con_n_noisy > 0)

    search_k = max(
        getattr(config, "SEARCH_K_ORACLE", config.SEARCH_K_CONTENT),
        getattr(config, "SEARCH_K_UNRESTRICTED", config.SEARCH_K_CONTENT),
    )

    def _query_subbatch(query_type, start, embs, n_results):
        tid = threading.get_ident()
        print(f"  [thread {tid}] {query_type} subbatch start  indices={start}..{start+len(embs)-1}")
        t0 = time.time()
        results = query_collection_batch(
            query_embs=embs,
            collection=collection,
            n_results=n_results,
            side=None,
            target_status_ids=None,
        )
        elapsed = time.time() - t0
        print(f"  [thread {tid}] {query_type} subbatch done   indices={start}..{start+len(embs)-1}  ({elapsed:.2f}s)")
        return results

    def _query_parallel(query_type, all_embs, n_results):
        subbatches = []
        for start in range(0, len(all_embs), CHROMA_SUBBATCH_SIZE):
            end = min(start + CHROMA_SUBBATCH_SIZE, len(all_embs))
            subbatches.append((start, all_embs[start:end]))

        print(f"[{query_type}] launching {len(subbatches)} subbatch threads")
        results = [None] * len(all_embs)

        with ThreadPoolExecutor(max_workers=len(subbatches)) as executor:
            future_to_start = {
                executor.submit(_query_subbatch, query_type, start, sub_embs, n_results): start
                for start, sub_embs in subbatches
            }
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                sub_results = future.result()
                for j, res in enumerate(sub_results):
                    results[start + j] = res

        print(f"[{query_type}] all subbatches done")
        return results

    if need_oracle or need_unrestricted:
        print("[fetch_live_batch] launching content + temporal outer threads")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            content_future = executor.submit(_query_parallel, "content", batch_query_embs_np, search_k)
            temporal_future = executor.submit(_query_parallel, "temporal", batch_future_embs_np, config.SEARCH_K_TEMPORAL)
            content_all = content_future.result()
            temporal_all = temporal_future.result()
        print(f"[fetch_live_batch] both outer threads done ({time.time() - t0:.2f}s)")
    else:
        content_all = [None] * batch_size
        temporal_all = _query_parallel("temporal", batch_future_embs_np, config.SEARCH_K_TEMPORAL)

    oracle_results = [None] * batch_size
    unrestricted_results = [None] * batch_size
    temporal_results = [None] * batch_size

    # for global_i in range(batch_size):
    #     if content_all[global_i] is not None:
    #         results = content_all[global_i]
    #         q_status = int(batch_query_meta[global_i]["status_id"])

    #         same_class_count = sum(
    #             1 for c in results
    #             if int(c["meta"]["status_id"]) == q_status
    #         )
    #         if same_class_count < sim_n_ideal:
    #             print(f"  [fallback] firing for global_i={global_i} q_status={q_status} same_class_count={same_class_count}")
    #             fallback = query_collection_batch(
    #                 query_embs=[batch_query_embs_np[global_i]],
    #                 collection=collection,
    #                 n_results=sim_n_ideal * 2,
    #                 side=None,
    #                 target_status_ids=[q_status],
    #             )
    #             existing_sigs = {dedup_signature(c["meta"]) for c in results}
    #             for c in fallback[0]:
    #                 sig = dedup_signature(c["meta"])
    #                 if sig not in existing_sigs:
    #                     results.append(c)
    #                     existing_sigs.add(sig)

    #         oracle_results[global_i] = results
    #         unrestricted_results[global_i] = results

    #     temporal_results[global_i] = temporal_all[global_i]

    for global_i in range(batch_size):
        if content_all[global_i] is not None:
            results = content_all[global_i]
            q_status = int(batch_query_meta[global_i]["status_id"])

            same_class_count = sum(
                1 for c in results
                if int(c["meta"]["status_id"]) == q_status
            )
            if same_class_count < sim_n_ideal:
                print(f"  [padding] firing for global_i={global_i} q_status={q_status} same_class_count={same_class_count}")
            oracle_results[global_i] = content_all[global_i]
            unrestricted_results[global_i] = content_all[global_i]
        temporal_results[global_i] = temporal_all[global_i]

    query_embs_out = []
    support_tokens = []
    contrast_tokens = []
    temporal_tokens = []

    for i in range(batch_size):
        entry = assemble_live_entry_from_candidates(
            query_emb=batch_query_embs_np[i],
            future_emb=batch_future_embs_np[i],
            query_meta=batch_query_meta[i],
            future_meta=batch_future_meta[i],
            oracle_candidates=oracle_results[i],
            unrestricted_candidates=unrestricted_results[i],
            temporal_candidates=temporal_results[i],
            k_sim=config.K_SIM,
            k_contrast=config.K_CONTRAST,
            k_temporal=config.K_TEMPORAL,
            training=training,
            epoch_idx=epoch_idx,
        )

        query_embs_out.append(entry["query_emb"])
        support_tokens.append(entry["sim_embs"])
        contrast_tokens.append(entry["contrast_embs"])
        temporal_tokens.append(entry["temporal_embs"])

    query_embs_out = tf.convert_to_tensor(np.stack(query_embs_out, axis=0), dtype=tf.float32)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

    support_mask = make_padding_mask(support_tokens)    # (B, K_SIM)
    contrast_mask = make_padding_mask(contrast_tokens)  # (B, K_CONTRAST)
    temporal_mask = make_padding_mask(temporal_tokens)  # (B, K_TEMPORAL)

    return query_embs_out, support_tokens, contrast_tokens, temporal_tokens




# def build_live_entry(
#     chunk,
#     future_chunk,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     search_k_content,
#     search_k_temporal,
#     k_sim,
#     k_contrast,
#     k_temporal,
#     training=False,
#     epoch_idx=None,
# ):
#     query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

#     query_meta = extract_meta(chunk)
#     future_meta = extract_meta(future_chunk)

#     q_status = int(query_meta["status_id"])
#     q_side = str(query_meta["side"])
#     emb_dim = query_emb.shape[0]

#     other_statuses = [s for s in (0, 1, 2) if s != q_status]
#     other_a, other_b = other_statuses[0], other_statuses[1]

#     pad_meta_template = {
#         "label": -1,
#         "side": "PAD",
#         "status": "PAD",
#         "status_id": -1,
#         "vid": -1,
#         "clip": -1,
#         "t_center": -1.0,
#         "t_width": -1.0,
#         "start_idx": -1,
#         "end_idx": -1,
#     }

#     # -------------------------
#     # oracle support pool
#     # -------------------------
#     print('building support')
#     support_oracle = query_collection(
#         query_emb=query_emb,
#         collection=collection,
#         n_results=search_k_content,
#         side=q_side,
#         target_status_ids=[q_status],
#     )
#     support_oracle = dedup_and_remove_self(support_oracle, query_meta)

    
#     # -------------------------
#     # oracle contrast pools
#     # -------------------------
#     print('building contrast a')
#     contrast_oracle_a = query_collection(
#         query_emb=query_emb,
#         collection=collection,
#         n_results=search_k_content,
#         side=q_side,
#         target_status_ids=[other_a],
#     )
#     contrast_oracle_a = dedup_and_remove_self(contrast_oracle_a, query_meta)

#     print('building contrast b')
#     contrast_oracle_b = query_collection(
#         query_emb=query_emb,
#         collection=collection,
#         n_results=search_k_content,
#         side=q_side,
#         target_status_ids=[other_b],
#     )
#     contrast_oracle_b = dedup_and_remove_self(contrast_oracle_b, query_meta)

#     # -------------------------
#     # unrestricted noisy pool
#     # -------------------------
#     print('building unrestricted')
#     unrestricted_candidates = query_collection(
#         query_emb=query_emb,
#         collection=collection,
#         n_results=search_k_content,
#         side=q_side,
#         target_status_ids=None,
#     )
#     unrestricted_candidates = dedup_and_remove_self(unrestricted_candidates, query_meta)

#     unrestricted_same = []
#     unrestricted_diff = []

#     for cand in unrestricted_candidates:
#         s = int(cand["meta"]["status_id"])
#         if s == q_status:
#             unrestricted_same.append(cand)
#         elif s in (0, 1, 2):
#             unrestricted_diff.append(cand)

#     unrestricted_diff_balanced = interleave_lists(
#         [c for c in unrestricted_diff if int(c["meta"]["status_id"]) == other_a],
#         [c for c in unrestricted_diff if int(c["meta"]["status_id"]) == other_b],
#         n_total=len(unrestricted_diff),
#     )

#     # -------------------------
#     # support curriculum
#     # -------------------------
#     sim_n_ideal, sim_n_noisy, sim_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_sim, training=training
#     )

#     if sim_unrestricted:
#         sim_items = take_unique(unrestricted_candidates, k_sim)
#     else:
#         sim_items = []

#         # ideal support = oracle same-class
#         sim_items.extend(take_unique(support_oracle, sim_n_ideal))

#         # noisy support = unrestricted non-query classes
#         used = {dedup_signature(x["meta"]) for x in sim_items}
#         noisy_support = []
#         for cand in unrestricted_diff_balanced:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             noisy_support.append(cand)
#             used.add(sig)
#             if len(noisy_support) >= sim_n_noisy:
#                 break

#         sim_items.extend(noisy_support)

#         # backfill support from oracle same-class, then unrestricted
#         if len(sim_items) < k_sim:
#             used = {dedup_signature(x["meta"]) for x in sim_items}
#             for pool in [support_oracle, unrestricted_candidates]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     sim_items.append(cand)
#                     used.add(sig)
#                     if len(sim_items) >= k_sim:
#                         break
#                 if len(sim_items) >= k_sim:
#                     break

#     sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)

#     # -------------------------
#     # contrast curriculum
#     # -------------------------
#     con_n_ideal, con_n_noisy, con_unrestricted = get_branch_mix_counts(
#         epoch_idx=epoch_idx, k=k_contrast, training=training
#     )

#     print({'sim ideal count': sim_n_ideal, 'sim noise count': sim_n_noisy, 'con ideal count': con_n_ideal, 'con noise count': con_n_noisy})
#     if con_unrestricted:
#         # fully real contrast = later unrestricted slice if possible
#         contrast_items = take_unique(unrestricted_diff_balanced, k_contrast)
#         if len(contrast_items) < k_contrast:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in unrestricted_candidates:
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= k_contrast:
#                     break
#     else:
#         contrast_items = []

#         # ideal contrast = balanced oracle from the two other classes
#         n_a = con_n_ideal // 2
#         n_b = con_n_ideal - n_a

#         contrast_items.extend(take_unique(contrast_oracle_a, n_a))
#         contrast_items.extend(take_unique(contrast_oracle_b, n_b))

#         # backfill ideal contrast if one side is short
#         if len(contrast_items) < con_n_ideal:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for cand in interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9):
#                 sig = dedup_signature(cand["meta"])
#                 if sig in used:
#                     continue
#                 contrast_items.append(cand)
#                 used.add(sig)
#                 if len(contrast_items) >= con_n_ideal:
#                     break

#         # noisy contrast = unrestricted same-class tokens
#         used = {dedup_signature(x["meta"]) for x in contrast_items}
#         noisy_contrast = []
#         for cand in unrestricted_same:
#             sig = dedup_signature(cand["meta"])
#             if sig in used:
#                 continue
#             noisy_contrast.append(cand)
#             used.add(sig)
#             if len(noisy_contrast) >= con_n_noisy:
#                 break

#         contrast_items.extend(noisy_contrast)

#         # backfill contrast from oracle contrast pools, then unrestricted diff
#         if len(contrast_items) < k_contrast:
#             used = {dedup_signature(x["meta"]) for x in contrast_items}
#             for pool in [
#                 interleave_lists(contrast_oracle_a, contrast_oracle_b, n_total=10**9),
#                 unrestricted_diff_balanced,
#                 unrestricted_candidates,
#             ]:
#                 for cand in pool:
#                     sig = dedup_signature(cand["meta"])
#                     if sig in used:
#                         continue
#                     contrast_items.append(cand)
#                     used.add(sig)
#                     if len(contrast_items) >= k_contrast:
#                         break
#                 if len(contrast_items) >= k_contrast:
#                     break

#     contrast_embs, contrast_meta = pad_or_trim(
#         contrast_items, k_contrast, emb_dim, pad_meta_template
#     )

#     # -------------------------
#     # temporal retrieval
#     # -------------------------
#     print('building temporal ')
#     temporal_candidates = query_collection(
#         query_emb=future_emb,
#         collection=collection,
#         n_results=search_k_temporal,
#         side=q_side,
#         target_status_ids=None,
#     )
#     temporal_candidates = dedup_and_remove_self(temporal_candidates, query_meta)

#     temporal_items = take_unique(temporal_candidates, k_temporal)

#     temporal_embs, temporal_meta = pad_or_trim(
#         temporal_items, k_temporal, emb_dim, pad_meta_template
#     )

#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "query_meta": query_meta,
#         "future_meta": future_meta,
#         "sim_meta": sim_meta,
#         "contrast_meta": contrast_meta,
#         "temporal_meta": temporal_meta,
#     }

scce_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)

def compute_class_weights(chunk_samples, power=0.5):
    labels = np.array([int(c["status_id"]) for c in chunk_samples], dtype=np.int32)
    counts = np.bincount(labels, minlength=3).astype(np.float32)

    if np.any(counts == 0):
        raise ValueError(f"Missing class in training data. Counts: {counts}")

    max_count = counts.max()
    weights = (max_count / counts) ** power
    weights = weights / weights[0]

    print("train class counts:", counts.astype(int).tolist())
    print("class weights:", weights.tolist())
    return tf.constant(weights, dtype=tf.float32)

def weighted_scce_loss(labels, logits, class_weights):
    per_example_loss = scce_no_reduce(labels, logits)             # (B,)
    weights = tf.gather(class_weights, tf.cast(labels, tf.int32)) # (B,)
    return tf.reduce_mean(per_example_loss * weights)

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

# def print_random_cache_queries():
#     r_inds = [random.randint(1,len(cache.keys())) for i in range(10)]
#     for ind in r_inds: 
#         entry = cache[list(cache.keys())[ind]]
#         print('-----------------')
#         print("QUERY")
#         print(entry["query_meta"])

#         print("\nSIM")
#         for m in entry["sim_meta"][:5]:
#             print(m)

#         print("\nCONTRAST")
#         for m in entry["contrast_meta"][:5]:
#             print(m)

#         print("\nTEMPORAL")
#         for m in entry["temporal_meta"][:5]:
#             print(m)

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
# come back
    query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)         # (B, D)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32) # (B, Ks, D)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32) # (B, Kc, D)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32) # (B, Kt, D)

    return query_embs, support_tokens, contrast_tokens, temporal_tokens

# def fetch_live_batch(
#     metadata,
#     chunk_lookup,
#     future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     training=False,
#     epoch_idx=None
# ):
#     batch_size = metadata["vid"].shape[0]

#     query_embs = []
#     support_tokens = []
#     contrast_tokens = []
#     temporal_tokens = []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         # key = (7, 'left', 6, 24, 35) 
#         chunk = chunk_lookup[key]
#         future_key = future_key_lookup[key]
#         future_chunk = chunk_lookup[future_key]

#         # print(f"key: {key} "
#         #       f"future key: {future_key} "
#         #       f"future chunk: {future_chunk} ")
        
#         # q = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#         # print(q[:20])
#         # print(np.linalg.norm(q))
#         entry = build_live_entry(
#             chunk=chunk,
#             future_chunk=future_chunk,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             search_k_content=config.SEARCH_K_CONTENT,
#             search_k_temporal=config.SEARCH_K_TEMPORAL,
#             k_sim=config.K_SIM,
#             k_contrast=config.K_CONTRAST,
#             k_temporal=config.K_TEMPORAL,
#             training = training,
#             epoch_idx=epoch_idx
#         )

#         # print('--------------------------------')
#         # pprint.pprint(entry["sim_meta"][:5])
#         # print()
#         # pprint.pprint(entry["contrast_meta"][:5])
#         # print()
#         # pprint.pprint(entry["temporal_meta"][:5])
#         # print('***********************************')

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     return query_embs, support_tokens, contrast_tokens, temporal_tokens

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

def train_step(batch,
    ratt_head,
    train_chunk_lookup,
    train_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    # pos_weight
    class_weights,
    epoch_idx=None
    ):
    # metadata, labels = batch[1], batch[2]

    # labels = tf.cast(labels, tf.float32)
    # labels = tf.reshape(labels, (-1, 1))

    metadata = batch[1]
    labels = tf.cast(metadata["status_id"], tf.int32)
    labels = tf.reshape(labels, (-1,))

    # query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_cache_batch(
    #     metadata, cache
    # )
    query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
        metadata=metadata,
        chunk_lookup=train_chunk_lookup,
        future_key_lookup=train_future_key_lookup,
        collection=collection,
        chunk_encoder=chunk_encoder,
        frame_emb_mm=frame_emb_mm,
        path_to_idx=path_to_idx,
        training=True,
        epoch_idx=epoch_idx
    )
    zeros_query = tf.zeros_like(query_embs)
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
            # chunk_embs=query_embs,
            chunk_embs=zeros_query,
            support_tokens=support_tokens,
            contrast_tokens=contrast_tokens,
            temporal_tokens=temporal_tokens,
            training=True,
        )
        
        # loss = weighted_bce_with_logits(labels, class_logits, pos_weight=pos_weight)
        loss = weighted_scce_loss(labels, class_logits, class_weights)

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

    # probs = tf.sigmoid(class_logits)
    probs = tf.nn.softmax(class_logits, axis=-1)
    batch_preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)
# ctrlf
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(labels, probs)

    # batch_preds = tf.cast(probs >= 0.5, tf.float32)
    batch_acc = tf.reduce_mean(tf.cast(tf.equal(batch_preds, labels), tf.float32))
    print(f"batch acc={batch_acc:.6f}")
    return {
        "loss": float(loss.numpy()),
        "acc": float(train_acc_metric.result().numpy()),
        "logits": class_logits,
        "probs": probs,
        "cls_out": cls_out,
        "aux": aux,
        "labels": labels,
        "preds": batch_preds
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
    # pos_weight
    class_weights
):
    # metadata, labels = batch[1], batch[2]

    # labels = tf.cast(labels, tf.float32)
    # labels = tf.reshape(labels, (-1, 1))

    metadata = batch[1]
    labels = tf.cast(metadata["status_id"], tf.int32)
    labels = tf.reshape(labels, (-1,))

    query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
        metadata=metadata,
        chunk_lookup=val_chunk_lookup,
        future_key_lookup=val_future_key_lookup,
        collection=collection,
        chunk_encoder=chunk_encoder,
        frame_emb_mm=frame_emb_mm,
        path_to_idx=path_to_idx,
        training=False
    )

    zeros_query = tf.zeros_like(query_embs)
    zeros_support = tf.zeros_like(support_tokens)
    zeros_contrast = tf.zeros_like(contrast_tokens)
    zeros_temporal = tf.zeros_like(temporal_tokens)

    class_logits, cls_out, aux = ratt_head(
        # chunk_embs=query_embs,
        support_tokens=support_tokens,
        contrast_tokens=contrast_tokens,
        temporal_tokens=temporal_tokens,
        # support_tokens=zeros_support,
        # contrast_tokens=zeros_contrast,
        # temporal_tokens=zeros_temporal,
        chunk_embs=zeros_query,
        training=False,
    )

    # loss = bce_loss_fn(labels, class_logits)
    # loss = weighted_bce_with_logits(labels, class_logits, pos_weight=pos_weight)
    loss = weighted_scce_loss(labels, class_logits, class_weights)

    if ratt_head.losses:
        loss += tf.add_n(ratt_head.losses)

    # probs = tf.sigmoid(class_logits)
    probs = tf.nn.softmax(class_logits, axis=-1)
    preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)

    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(labels, probs)

    return {
        "loss": float(loss.numpy()),
        "acc": float(val_acc_metric.result().numpy()),
        "logits": class_logits,
        "probs": probs,
        "preds": preds,
        'labels':labels
        # "cls_out": cls_out,
        # "aux": aux,
    }


def run_train_epoch(train_ds,
    ratt_head,
    train_chunk_lookup,
    train_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    # pos_weight
    class_weight,
    epoch_idx=None
    ):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()

    for step, batch in enumerate(train_ds):
        out = train_step(
            batch=batch,
            ratt_head=ratt_head,
            train_chunk_lookup=train_chunk_lookup,
            train_future_key_lookup=train_future_key_lookup,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx,
            # pos_weight=pos_weight
            class_weights=class_weight,
            epoch_idx=epoch_idx
        )

        if step % 1 == 0:
            print(
                f"[train] step={step} "
                f"loss={out['loss']:.4f} "
                f"acc={train_acc_metric.result().numpy():.4f}"
            )

            temp = pd.DataFrame({
                "label": out["labels"].numpy(),
                "pred": out["preds"].numpy(),
                "logit_0": out["logits"].numpy()[:, 0],
                "logit_1": out["logits"].numpy()[:, 1],
                "logit_2": out["logits"].numpy()[:, 2],
                "prob_0": out["probs"].numpy()[:, 0],
                "prob_1": out["probs"].numpy()[:, 1],
                "prob_2": out["probs"].numpy()[:, 2],
            })
            print(temp)

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
    # pos_weight
    class_weight
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
            # pos_weight=pos_weight
            class_weights=class_weight
        )

        # batch_preds = tf.cast(out['probs'] >= 0.5, tf.float32)
        # batch_preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(out['preds'], out['labels']), tf.float32))
        if step % 1 == 0:
            print(
                f"[val] step={step} "
                f"loss={out['loss']:.4f} "
                f"running acc={val_acc_metric.result().numpy():.4f} "
                f"batch acc={batch_acc:.4f} "
            )
            # temp = pd.DataFrame()
            

            # temp['labels'] = out['labels'].numpy().flatten()
            # temp['logits'] = out['logits'].numpy().flatten()
            # temp['probs'] = out['probs'].numpy().flatten()
            # print(temp)

            temp = pd.DataFrame({
                "label": out["labels"].numpy(),
                "pred": out["preds"].numpy(),
                "logit_0": out["logits"].numpy()[:, 0],
                "logit_1": out["logits"].numpy()[:, 1],
                "logit_2": out["logits"].numpy()[:, 2],
                "prob_0": out["probs"].numpy()[:, 0],
                "prob_1": out["probs"].numpy()[:, 1],
                "prob_2": out["probs"].numpy()[:, 2],
            })
            print(temp)
            # pprint.pprint(out)


    return (
        float(val_loss_metric.result().numpy()),
        float(val_acc_metric.result().numpy()),
    )

if __name__ == "__main__":

    print("SEARCH_K_CONTENT", config.SEARCH_K_CONTENT)
    print("SEARCH_K_TEMPORAL", config.SEARCH_K_TEMPORAL)
    print("K_SIM", config.K_SIM)
    print("K_CONTRAST", config.K_CONTRAST)
    print("K_TEMPORAL", config.K_TEMPORAL)
    print("FUTURE_CHUNK_STEP", config.FUTURE_CHUNK_STEP)
    print("CHUNK_SIZE", config.CHUNK_SIZE)
    print("CHROMADB_COLLECTION", config.CHROMADB_COLLECTION)
    print("STAGE1_WEIGHTS", config.STAGE1_WEIGHTS)
    print("RATT_WEIGHTS", config.RATT_WEIGHTS)

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = config.TRAIN_VIDS
    train_samples = load_samples(train_vids,stride=1)
    train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE)
    train_chunk_samples = oversample_chunk_samples(train_chunk_samples,0.3)
    # train_chunk_samples = train_chunk_samples[0:100]
    # label_lookup = {}
    # for c in train_chunk_samples:
    #     key = make_key(c["vid"], c["side"], c["t_center"])
    #     label_lookup[key] = int(c["label"])

    test_vids = config.TEST_VIDS
    test_samples = load_samples(test_vids,stride=1)
    test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE)
    # test_chunk_samples = train_chunk_samples[0:32]
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # split 95/5
    # n = len(chunk_samples)
    print(len(train_chunk_samples))
    print(len(test_chunk_samples))

    pos_weight = compute_pos_weight(train_chunk_samples)
    class_weight = compute_class_weights(train_chunk_samples)
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
        num_layers=1,
        num_heads=4,
        max_frames=config.CHUNK_SIZE
    )

    dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
    _ = chunk_encoder(dummy_frame_embs, training=False)

    chunk_encoder.load_weights(config.STAGE1_WEIGHTS)

    #come back
    for i in range(chunk_encoder.num_layers):
        block = getattr(chunk_encoder, f"transformer_block_{i}")
        with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
            weights = pickle.load(f)
        block.set_weights(weights)

    print("[STAGE1] Loaded chunk encoder weights")

    chunk_encoder.trainable = False
    print("[STAGE1] Chunk encoder frozen")

    store_name = 'train_val_frames_chunk8_stride2'
    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)
    # if(os.path.exists(config.STAGE2_CACHE_PATH)):
    #     cache = load_retrieval_cache(config.STAGE2_CACHE_PATH)
    #     # print(cache)
    #     print("[CACHE] loaded cache")
    # else:
    #     cache = build_retrieval_cache(
    #         all_chunks=train_chunk_samples,
    #         collection=collection,
    #         chunk_encoder=chunk_encoder,
    #         frame_emb_mm=frame_emb_mm,
    #         frame_paths=frame_paths,
    #         path_to_idx=path_to_idx
    #         )
    #     save_retrieval_cache(cache,config.STAGE2_CACHE_PATH)
    
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    # train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="train_acc")
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")


    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    # val_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="val_acc")
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    batch = next(iter(train_dataset))

    ratt_head = RATTHeadV2(
        hidden_size=768,
        num_heads=8,
        num_layers=config.NUM_LAYERS,
    )

    for v in ratt_head.trainable_variables[:5]:
        print(v.name, float(tf.reduce_mean(v).numpy()), float(tf.math.reduce_std(v).numpy()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    # query_embs, support, contrast, temporal = fetch_cache_batch(batch[1], cache)
    
    query_embs = np.zeros((config.CHUNK_BATCH_SIZE,768))
    support = np.zeros((config.CHUNK_BATCH_SIZE,config.K_SIM,768))
    contrast = np.zeros((config.CHUNK_BATCH_SIZE,config.K_CONTRAST,768))
    temporal = np.zeros((config.CHUNK_BATCH_SIZE,config.K_TEMPORAL,768))
    train_chunk_lookup = {make_chunk_key(c): c for c in train_chunk_samples}
    train_future_key_lookup = build_future_key_lookup(train_chunk_samples, future_step=5)

    val_chunk_lookup = {make_chunk_key(c): c for c in test_chunk_samples}
    val_future_key_lookup = build_future_key_lookup(test_chunk_samples, future_step=5)

    # come back
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
    # print_random_cache_queries()
    for epoch in range(config.EPOCHS):
        print(f"\n===== EPOCH {epoch+1}/{config.EPOCHS} =====")

        train_loss, train_acc = run_train_epoch(
            train_ds=train_dataset,
            ratt_head=ratt_head,
            train_chunk_lookup=train_chunk_lookup,
            train_future_key_lookup=train_future_key_lookup,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx,
            # pos_weight=pos_weight
            class_weight=class_weight,
            epoch_idx=epoch+1
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
            # pos_weight=pos_weight
            class_weight=class_weight
        )

        print(
            f"[epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # -------------------------
    # save weights
    # -------------------------
    # =============================================
    # SAVE WEIGHTS
    # =============================================

    ratt_head.save_weights(config.RATT_WEIGHTS)
    print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")

    os.makedirs("rag_weights", exist_ok=True)

    for i in range(config.NUM_LAYERS):
        block = getattr(ratt_head, f"transformer_block_{i}")
        with open(f"rag_weights/{config.RUN_ID}_transformer_block_{i}.pkl", "wb") as f:
            pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[MAIN] saved transformer block {i} weights")

