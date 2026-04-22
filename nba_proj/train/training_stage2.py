
from collections import defaultdict
import math
import os
import pprint
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tf_keras
from chromadb import PersistentClient

from dataset import load_samples, build_chunks, build_tf_dataset_chunks, oversample_chunk_samples
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


# ─────────────────────────────────────────────
# Key helpers
# ─────────────────────────────────────────────

def make_chunk_key(chunk, precision=6):
    return (
        int(chunk["vid"]),
        str(chunk["side"]),
        int(chunk["clip"]),
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
    emb_mm = np.memmap(paths["emb"], dtype="float32", mode="r", shape=(n_frames, emb_dim))
    frame_paths = np.load(paths["paths"], allow_pickle=True).tolist()
    path_to_idx = {p: i for i, p in enumerate(frame_paths)}
    print(f"[frame cache] loaded store '{store_name}'")
    print(f"[frame cache] shape = ({n_frames}, {emb_dim})")
    return emb_mm, frame_paths, path_to_idx


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


def pad_or_trim(items, k, emb_dim, pad_meta_template):
    if len(items) >= k:
        items = items[:k]
    else:
        pad_count = k - len(items)
        zero_emb = np.zeros((emb_dim,), dtype=np.float32)
        for _ in range(pad_count):
            items.append({"emb": zero_emb.copy(), "meta": dict(pad_meta_template)})
    embs = np.stack([x["emb"] for x in items], axis=0)
    metas = [x["meta"] for x in items]
    return embs, metas


def make_padding_mask(tokens):
    norms = tf.norm(tokens, axis=-1)
    return norms > 1e-6


# ─────────────────────────────────────────────
# ChromaDB query helpers
# ─────────────────────────────────────────────

def query_collection(query_emb, collection, n_results, side=None, target_status_ids=None):
    where_clauses = []
    if side is not None:
        where_clauses.append({"side": str(side)})
    if target_status_ids is not None:
        target_status_ids = [int(s) for s in target_status_ids]
        if len(target_status_ids) == 1:
            where_clauses.append({"status_id": target_status_ids[0]})
        else:
            where_clauses.append({"$or": [{"status_id": s} for s in target_status_ids]})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
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
        out.append({
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
        })
    return out


def query_collection_batch(query_embs, collection, n_results, side=None, target_status_ids=None):
    where_clauses = []
    if side is not None:
        where_clauses.append({"side": str(side)})
    if target_status_ids is not None:
        target_status_ids = [int(s) for s in target_status_ids]
        if len(target_status_ids) == 1:
            where_clauses.append({"status_id": target_status_ids[0]})
        else:
            where_clauses.append({"$or": [{"status_id": s} for s in target_status_ids]})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
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
    for raw_embs, raw_meta in zip(results["embeddings"], results["metadatas"]):
        out = []
        for emb, meta in zip(raw_embs, raw_meta):
            out.append({
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
            })
        batch_out.append(out)
    return batch_out


# ─────────────────────────────────────────────
# Curriculum — always unrestricted
# ─────────────────────────────────────────────

def get_branch_mix_counts(epoch_idx, k, training):
    """Always fully unrestricted — no oracle filtering."""
    return k, 0, True


# ─────────────────────────────────────────────
# Entry assembly — always unrestricted
# ─────────────────────────────────────────────

def assemble_live_entry_from_candidates(
    query_emb,
    future_emb,
    query_meta,
    future_meta,
    unrestricted_candidates,
    temporal_candidates,
    k_sim,
    k_contrast,
    k_temporal,
):
    emb_dim = query_emb.shape[0]
    pad_meta_template = {
        "label": -1, "side": "PAD", "status": "PAD", "status_id": -1,
        "vid": -1, "clip": -1, "t_center": -1.0, "t_width": -1.0,
        "start_idx": -1, "end_idx": -1,
    }

    unrestricted_candidates = dedup_and_remove_self(unrestricted_candidates or [], query_meta)
    temporal_candidates = dedup_and_remove_self(temporal_candidates or [], query_meta)

    q_status = int(query_meta["status_id"])
    other_statuses = [s for s in (0, 1, 2) if s != q_status]
    other_a, other_b = other_statuses[0], other_statuses[1]

    unrestricted_diff_a = [c for c in unrestricted_candidates if int(c["meta"]["status_id"]) == other_a]
    unrestricted_diff_b = [c for c in unrestricted_candidates if int(c["meta"]["status_id"]) == other_b]
    unrestricted_diff_balanced = interleave_lists(
        unrestricted_diff_a, unrestricted_diff_b,
        n_total=len(unrestricted_diff_a) + len(unrestricted_diff_b),
    )

    # support — unrestricted (mixed class), model learns to identify relevant tokens
    sim_items = take_unique(unrestricted_candidates, k_sim)
    sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)

    # contrast — prefer different-class tokens first, fall back to any unrestricted
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
    contrast_embs, contrast_meta = pad_or_trim(contrast_items, k_contrast, emb_dim, pad_meta_template)

    # temporal — always unrestricted future-embedding retrieval
    temporal_items = take_unique(temporal_candidates, k_temporal)
    temporal_embs, temporal_meta = pad_or_trim(temporal_items, k_temporal, emb_dim, pad_meta_template)

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


# ─────────────────────────────────────────────
# Live batch fetching
# ─────────────────────────────────────────────

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
    epoch_idx=None,
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

    search_k = config.SEARCH_K_CONTENT

    def _query_subbatch(query_type, start, embs, n_results):
        tid = threading.get_ident()
        t0 = time.time()
        results = query_collection_batch(
            query_embs=embs,
            collection=collection,
            n_results=n_results,
        )
        elapsed = time.time() - t0
        print(f"  [thread {tid}] {query_type} subbatch {start}..{start+len(embs)-1} ({elapsed:.2f}s)")
        return results

    def _query_parallel(query_type, all_embs, n_results):
        subbatches = []
        for start in range(0, len(all_embs), CHROMA_SUBBATCH_SIZE):
            end = min(start + CHROMA_SUBBATCH_SIZE, len(all_embs))
            subbatches.append((start, all_embs[start:end]))

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
        return results

    print("[fetch_live_batch] launching content + temporal outer threads")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        content_future = executor.submit(_query_parallel, "content", batch_query_embs_np, search_k)
        temporal_future = executor.submit(_query_parallel, "temporal", batch_future_embs_np, config.SEARCH_K_TEMPORAL)
        content_all = content_future.result()
        temporal_all = temporal_future.result()
    print(f"[fetch_live_batch] both outer threads done ({time.time() - t0:.2f}s)")

    query_embs_out = []
    support_tokens = []
    contrast_tokens = []
    temporal_tokens = []
    sim_metas = []
    contrast_metas = []

    for i in range(batch_size):
        entry = assemble_live_entry_from_candidates(
            query_emb=batch_query_embs_np[i],
            future_emb=batch_future_embs_np[i],
            query_meta=batch_query_meta[i],
            future_meta=batch_future_meta[i],
            unrestricted_candidates=content_all[i],
            temporal_candidates=temporal_all[i],
            k_sim=config.K_SIM,
            k_contrast=config.K_CONTRAST,
            k_temporal=config.K_TEMPORAL,
        )

        query_embs_out.append(entry["query_emb"])
        support_tokens.append(entry["sim_embs"])
        contrast_tokens.append(entry["contrast_embs"])
        temporal_tokens.append(entry["temporal_embs"])
        sim_metas.append(entry["sim_meta"])
        contrast_metas.append(entry["contrast_meta"])

    query_embs_out = tf.convert_to_tensor(np.stack(query_embs_out, axis=0), dtype=tf.float32)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

    support_mask = make_padding_mask(support_tokens)
    contrast_mask = make_padding_mask(contrast_tokens)
    temporal_mask = make_padding_mask(temporal_tokens)

    return (
        query_embs_out,
        support_tokens, contrast_tokens, temporal_tokens,
        support_mask, contrast_mask, temporal_mask,
        sim_metas, contrast_metas,
    )


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

scce_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# grads = tape.gradient(loss, ratt_head.trainable_variables)
# grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
# optimizer.apply_gradients(zip(grads, ratt_head.trainable_variables))
# print(f"global_norm={float(global_norm):.4f}")

def weighted_scce_loss(labels, logits, class_weights):
    per_example_loss = scce_no_reduce(labels, logits)
    weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
    return tf.reduce_mean(per_example_loss * weights)

# job 2089 loss but margin = 0.5
# def attention_token_consistency_loss(cls_attn, sim_meta, contrast_meta, query_labels, Ks, Kc, margin=0.2):
#     """
#     Ranking loss on normalized within-branch attention.
#     Support: same-class tokens should get higher attention than different-class tokens.
#     Contrast: different-class tokens should get higher attention than same-class tokens.
#     """
#     support_attn = cls_attn[:, 2:2+Ks]
#     contrast_attn = cls_attn[:, 3+Ks:3+Ks+Kc]

#     # normalize within branch
#     support_attn_norm = support_attn / (tf.reduce_sum(support_attn, axis=1, keepdims=True) + 1e-8)
#     contrast_attn_norm = contrast_attn / (tf.reduce_sum(contrast_attn, axis=1, keepdims=True) + 1e-8)

#     support_labels = tf.constant(
#         [[m["status_id"] for m in row] for row in sim_meta], dtype=tf.int32
#     )
#     contrast_labels = tf.constant(
#         [[m["status_id"] for m in row] for row in contrast_meta], dtype=tf.int32
#     )

#     query_labels_exp = tf.expand_dims(query_labels, axis=1)

#     support_match = tf.cast(tf.equal(support_labels, query_labels_exp), tf.float32)
#     support_valid = tf.cast(tf.not_equal(support_labels, -1), tf.float32)
#     contrast_mismatch = tf.cast(tf.not_equal(contrast_labels, query_labels_exp), tf.float32)
#     contrast_valid = tf.cast(tf.not_equal(contrast_labels, -1), tf.float32)

#     n_sup_match    = tf.reduce_sum(support_match * support_valid, axis=1) + 1e-8
#     n_sup_mismatch = tf.reduce_sum((1 - support_match) * support_valid, axis=1) + 1e-8
#     n_con_mismatch = tf.reduce_sum(contrast_mismatch * contrast_valid, axis=1) + 1e-8
#     n_con_match    = tf.reduce_sum((1 - contrast_mismatch) * contrast_valid, axis=1) + 1e-8

#     sup_match_attn    = tf.reduce_sum(support_attn_norm * support_match * support_valid, axis=1) / n_sup_match
#     sup_mismatch_attn = tf.reduce_sum(support_attn_norm * (1 - support_match) * support_valid, axis=1) / n_sup_mismatch
#     con_mismatch_attn = tf.reduce_sum(contrast_attn_norm * contrast_mismatch * contrast_valid, axis=1) / n_con_mismatch
#     con_match_attn    = tf.reduce_sum(contrast_attn_norm * (1 - contrast_mismatch) * contrast_valid, axis=1) / n_con_match

#     support_loss  = tf.reduce_mean(tf.maximum(0.0, sup_mismatch_attn - sup_match_attn + margin))
#     contrast_loss = tf.reduce_mean(tf.maximum(0.0, con_match_attn - con_mismatch_attn + margin))

#     tf.print("sup_match_attn:", sup_match_attn[:5])
#     tf.print("sup_mismatch_attn:", sup_mismatch_attn[:5])
#     tf.print("support_loss per example:", tf.maximum(0.0, sup_mismatch_attn - sup_match_attn + margin)[:5])
    
#     for i in range(min(3, len(sim_meta))):
#         query_label = query_labels[i].numpy()
#         sup_labels_i = [m["status_id"] for m in sim_meta[i]]
#         sup_attn_i = support_attn_norm[i].numpy()
        
#         # pair labels with attention weights and sort by attention (highest first)
#         pairs = sorted(zip(sup_labels_i, sup_attn_i), key=lambda x: -x[1])
        
#         print(f"\n[example {i}] query_label={query_label}")
#         print(f"  top 10 support tokens by attention (label, attn):")
#         for lbl, attn in pairs[:10]:
#             match = "✓" if lbl == query_label else "✗" if lbl != -1 else "PAD"
#             print(f"    label={lbl} attn={attn:.4f} {match}")
#     return support_loss + contrast_loss

# job 2088 loss
def attention_token_consistency_loss(cls_attn, sim_meta, contrast_meta, query_labels, Ks, Kc):
    support_attn = cls_attn[:, 1:1+Ks]
    contrast_attn = cls_attn[:, 1+Ks:1+Ks+Kc]

    support_labels = tf.constant(
        [[m["status_id"] for m in row] for row in sim_meta], dtype=tf.int32
    )
    contrast_labels = tf.constant(
        [[m["status_id"] for m in row] for row in contrast_meta], dtype=tf.int32
    )

    query_labels_exp = tf.expand_dims(query_labels, axis=1)

    support_match = tf.cast(tf.equal(support_labels, query_labels_exp), tf.float32)
    support_valid = tf.cast(tf.not_equal(support_labels, -1), tf.float32)
    contrast_mismatch = tf.cast(tf.not_equal(contrast_labels, query_labels_exp), tf.float32)
    contrast_valid = tf.cast(tf.not_equal(contrast_labels, -1), tf.float32)

    support_valid_attn = support_attn * support_valid + 1e-9
    support_attn_norm = support_valid_attn / (tf.reduce_sum(support_valid_attn, axis=1, keepdims=True))

    match_prob = tf.reduce_sum(support_attn_norm * support_match, axis=1)

    has_mismatch = tf.cast(
        tf.reduce_sum((1 - support_match) * support_valid, axis=1) > 0, tf.float32
    )

    support_loss = -tf.reduce_sum(
        has_mismatch * tf.math.log(match_prob + 1e-8)
    ) / (tf.reduce_sum(has_mismatch) + 1e-8)

    contrast_valid_attn = contrast_attn * contrast_valid + 1e-9
    contrast_attn_norm = contrast_valid_attn / (tf.reduce_sum(contrast_valid_attn, axis=1, keepdims=True))

    mismatch_prob = tf.reduce_sum(contrast_attn_norm * contrast_mismatch, axis=1)

    has_match_in_contrast = tf.cast(
        tf.reduce_sum((1 - contrast_mismatch) * contrast_valid, axis=1) > 0, tf.float32
    )

    contrast_loss = -tf.reduce_sum(
        has_match_in_contrast * tf.math.log(mismatch_prob + 1e-8)
    ) / (tf.reduce_sum(has_match_in_contrast) + 1e-8)

    tf.print("match_prob:", match_prob[:5])
    tf.print("mismatch_prob:", mismatch_prob[:5])

    # top attended tokens visualization for first 3 examples
    support_attn_norm_np = support_attn_norm.numpy()
    for i in range(min(3, len(sim_meta))):
        query_label = query_labels[i].numpy()
        sup_labels_i = [m["status_id"] for m in sim_meta[i]]
        sup_attn_i = support_attn_norm_np[i]
        pairs = sorted(zip(sup_labels_i, sup_attn_i), key=lambda x: -x[1])
        print(f"\n[example {i}] query_label={query_label}")
        for lbl, attn in pairs[:10]:
            match = "✓" if lbl == query_label else "✗" if lbl != -1 else "PAD"
            print(f"    label={lbl} attn={attn:.4f} {match}")

    return support_loss + contrast_loss


# ─────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────

def encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx):
    idxs = [path_to_idx[p] for p in chunk["frames"]]
    frame_embs = frame_emb_mm[idxs].astype(np.float32)
    frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)
    stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)
    return stage1_chunk_emb[0].numpy().astype(np.float32)


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


def compute_pos_weight(chunk_samples):
    labels = np.array([int(c["label"]) for c in chunk_samples], dtype=np.int32)
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    if num_pos == 0:
        raise ValueError("No positive examples found.")
    if num_neg == 0:
        raise ValueError("No negative examples found.")
    return float(np.sqrt(num_neg / num_pos))


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
    start_idx = _to_py_scalar(metadata["start_idx"][i])
    end_idx = _to_py_scalar(metadata["end_idx"][i])
    return (int(vid), str(side), int(clip), int(start_idx), int(end_idx))


# ─────────────────────────────────────────────
# Train / eval steps
# ─────────────────────────────────────────────

def train_step(
    batch,
    ratt_head,
    train_chunk_lookup,
    train_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    class_weights,
    epoch_idx=None,
):
    metadata = batch[1]
    labels = tf.cast(metadata["status_id"], tf.int32)
    labels = tf.reshape(labels, (-1,))

    (query_embs, support_tokens, contrast_tokens, temporal_tokens,
     support_mask, contrast_mask, temporal_mask,
     sim_metas, contrast_metas) = fetch_live_batch(
        metadata=metadata,
        chunk_lookup=train_chunk_lookup,
        future_key_lookup=train_future_key_lookup,
        collection=collection,
        chunk_encoder=chunk_encoder,
        frame_emb_mm=frame_emb_mm,
        path_to_idx=path_to_idx,
        training=True,
        epoch_idx=epoch_idx,
    )

    def grad_rms(g):
        if g is None:
            return 0.0
        g = tf.cast(g, tf.float32)
        return float(tf.sqrt(tf.reduce_mean(tf.square(g))).numpy())

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
            support_mask=support_mask,
            contrast_mask=contrast_mask,
            temporal_mask=temporal_mask,
            training=True,
        )

        last_attn = aux["attn_scores"][-1]
        attn_mean = tf.reduce_mean(last_attn, axis=1)
        cls_attn = attn_mean[:, 0, :]

        const_loss = attention_token_consistency_loss(
            cls_attn=cls_attn,
            sim_meta=sim_metas,
            contrast_meta=contrast_metas,
            query_labels=labels,
            Ks=config.K_SIM,
            Kc=config.K_CONTRAST,
        )

        class_loss = weighted_scce_loss(labels, class_logits, class_weights)
        loss = 1*class_loss + 0.1 * const_loss

        if ratt_head.losses:
            loss += tf.add_n(ratt_head.losses)

    print(f"class_loss={float(class_loss):.4f} const_loss={float(const_loss):.4f} total={float(loss):.4f}")

    grads = tape.gradient(loss, ratt_head.trainable_variables)
    # grads = tape.gradient(loss, ratt_head.trainable_variables)
    # grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # optimizer.apply_gradients(zip(grads, ratt_head.trainable_variables))
    # print(f"global_norm={float(global_norm):.4f}")
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

    probs = tf.nn.softmax(class_logits, axis=-1)
    batch_preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)

    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(labels, probs)

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
        "preds": batch_preds,
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
    class_weights,
):
    metadata = batch[1]
    labels = tf.cast(metadata["status_id"], tf.int32)
    labels = tf.reshape(labels, (-1,))

    (query_embs, support_tokens, contrast_tokens, temporal_tokens,
     support_mask, contrast_mask, temporal_mask,
     sim_metas, contrast_metas) = fetch_live_batch(
        metadata=metadata,
        chunk_lookup=val_chunk_lookup,
        future_key_lookup=val_future_key_lookup,
        collection=collection,
        chunk_encoder=chunk_encoder,
        frame_emb_mm=frame_emb_mm,
        path_to_idx=path_to_idx,
        training=False,
    )

    class_logits, cls_out, aux = ratt_head(
        chunk_embs=query_embs,
        support_tokens=support_tokens,
        contrast_tokens=contrast_tokens,
        temporal_tokens=temporal_tokens,
        support_mask=support_mask,
        contrast_mask=contrast_mask,
        temporal_mask=temporal_mask,
        training=False,
    )

    loss = weighted_scce_loss(labels, class_logits, class_weights)

    if ratt_head.losses:
        loss += tf.add_n(ratt_head.losses)

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
        "labels": labels,
    }


# ─────────────────────────────────────────────
# Epoch runners
# ─────────────────────────────────────────────

def run_train_epoch(
    train_ds,
    ratt_head,
    train_chunk_lookup,
    train_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    class_weight,
    epoch_idx=None,
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
            class_weights=class_weight,
            epoch_idx=epoch_idx,
        )

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


def run_val_epoch(
    val_ds,
    ratt_head,
    val_chunk_lookup,
    val_future_key_lookup,
    collection,
    chunk_encoder,
    frame_emb_mm,
    path_to_idx,
    class_weight,
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
            class_weights=class_weight,
        )

        batch_acc = tf.reduce_mean(tf.cast(tf.equal(out["preds"], out["labels"]), tf.float32))
        print(
            f"[val] step={step} "
            f"loss={out['loss']:.4f} "
            f"running acc={val_acc_metric.result().numpy():.4f} "
            f"batch acc={batch_acc:.4f}"
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
        float(val_loss_metric.result().numpy()),
        float(val_acc_metric.result().numpy()),
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

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

    # ── 1. Load samples ──────────────────────
    train_vids = config.TRAIN_VIDS
    train_samples = load_samples(train_vids, stride=1)
    train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE)
    # equal oversampling: {0: N, 1: N, 2: N}
    train_chunk_samples = oversample_chunk_samples(train_chunk_samples, 1.0)

    test_vids = config.TEST_VIDS
    test_samples = load_samples(test_vids, stride=1)
    test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE)

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    print(f"Train chunks: {len(train_chunk_samples)}")
    print(f"Val chunks:   {len(test_chunk_samples)}")

    pos_weight = compute_pos_weight(train_chunk_samples)
    class_weight = compute_class_weights(train_chunk_samples)

    # ── 2. Build TF datasets ─────────────────
    train_dataset = build_tf_dataset_chunks(train_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=True)
    val_dataset   = build_tf_dataset_chunks(test_chunk_samples,  batch_size=config.CHUNK_BATCH_SIZE, training=False)

    # ── 3. ChromaDB ──────────────────────────
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name=config.CHROMADB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ── 4. Stage 1 chunk encoder ─────────────
    chunk_encoder = ChunkEncoder(
        hidden_size=768, num_layers=1, num_heads=4, max_frames=config.CHUNK_SIZE
    )
    dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
    _ = chunk_encoder(dummy_frame_embs, training=False)
    chunk_encoder.load_weights(config.STAGE1_WEIGHTS)

    for i in range(chunk_encoder.num_layers):
        block = getattr(chunk_encoder, f"transformer_block_{i}")
        with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
            weights = pickle.load(f)
        block.set_weights(weights)

    print("[STAGE1] Loaded chunk encoder weights")
    chunk_encoder.trainable = False
    print("[STAGE1] Chunk encoder frozen")

    # ── 5. Frame store ───────────────────────
    store_name = "train_val_frames_chunk8_stride2"
    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

    # ── 6. Metrics ───────────────────────────
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric  = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")
    val_acc_metric    = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # ── 7. Model ─────────────────────────────
    ratt_head = RATTHeadV2(
        hidden_size=768,
        num_heads=8,
        num_layers=config.NUM_LAYERS,
    )

    # warm up model weights
    query_embs = np.zeros((config.CHUNK_BATCH_SIZE, 768))
    support    = np.zeros((config.CHUNK_BATCH_SIZE, config.K_SIM, 768))
    contrast   = np.zeros((config.CHUNK_BATCH_SIZE, config.K_CONTRAST, 768))
    temporal   = np.zeros((config.CHUNK_BATCH_SIZE, config.K_TEMPORAL, 768))

    logits, _, _ = ratt_head(
        chunk_embs=query_embs,
        support_tokens=support,
        contrast_tokens=contrast,
        temporal_tokens=temporal,
        training=False,
    )
    print("logits:", logits.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    train_chunk_lookup      = {make_chunk_key(c): c for c in train_chunk_samples}
    train_future_key_lookup = build_future_key_lookup(train_chunk_samples, future_step=5)
    val_chunk_lookup        = {make_chunk_key(c): c for c in test_chunk_samples}
    val_future_key_lookup   = build_future_key_lookup(test_chunk_samples, future_step=5)

    # ── 8. Training loop ─────────────────────
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
            class_weight=class_weight,
            epoch_idx=epoch + 1,
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
            class_weight=class_weight,
        )

        print(
            f"[epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # ── 9. Save weights ──────────────────────
    ratt_head.save_weights(config.RATT_WEIGHTS)
    print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")

    os.makedirs("rag_weights", exist_ok=True)
    for i in range(config.NUM_LAYERS):
        block = getattr(ratt_head, f"transformer_block_{i}")
        with open(f"rag_weights/{config.RUN_ID}_transformer_block_{i}.pkl", "wb") as f:
            pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[MAIN] saved transformer block {i} weights")