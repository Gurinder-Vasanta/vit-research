import os
import json
from collections import defaultdict
import pickle
import pprint
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from chromadb import PersistentClient

from dataset import load_samples, build_chunks, build_tf_dataset_chunks
from models.chunk_encoder import ChunkEncoder
from models.ratt_v2 import RATTHeadV2
import config_stage3 as config
import gc

# reuse these from your current stage2 file
# load_frame_store
# fetch_live_batch
# build_future_key_lookup
# make_chunk_key

tf.keras.backend.clear_session()
gc.collect()

SEED = 12

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

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

def z_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return x
    return (x - x.mean()) / (x.std() + 1e-6)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

        # print(key)
        chunk = chunk_lookup[key]
        future_key = future_key_lookup[key]
        future_chunk = chunk_lookup[future_key]

        # print(f"key: {key} "
        #       f"future key: {future_key} "
        #       f"future chunk: {future_chunk} ")

        # q = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
        # print(q[:20])
        # print(np.linalg.norm(q))
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

        # print('--------------------------------')
        # pprint.pprint(entry["sim_meta"][:5])
        # print()
        # pprint.pprint(entry["contrast_meta"][:5])
        # print()
        # pprint.pprint(entry["temporal_meta"][:5])
        # print('***********************************')

        query_embs.append(entry["query_emb"])
        support_tokens.append(entry["sim_embs"])
        contrast_tokens.append(entry["contrast_embs"])
        temporal_tokens.append(entry["temporal_embs"])

    query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
    support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
    contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
    temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

    return query_embs, support_tokens, contrast_tokens, temporal_tokens


def build_and_load_ratt_head():
    fresh_head = RATTHeadV2(
        hidden_size=768,
        num_heads=8,
        num_layers=2,   # or whatever your config uses
    )

    dummy_q = tf.zeros((1, 768), dtype=tf.float32)
    dummy_s = tf.zeros((1, config.K_SIM, 768), dtype=tf.float32)
    dummy_c = tf.zeros((1, config.K_CONTRAST, 768), dtype=tf.float32)
    dummy_t = tf.zeros((1, config.K_TEMPORAL, 768), dtype=tf.float32)

    fresh_head.query_proj.build((None, 1, 768))
    fresh_head.support_proj.build((None, config.K_SIM, 768))
    fresh_head.contrast_proj.build((None, config.K_CONTRAST, 768))
    fresh_head.temporal_proj.build((None, config.K_TEMPORAL, 768))
    fresh_head.classifier.build((None, 2 * 768))

    total_tokens = 4 + config.K_SIM + config.K_CONTRAST + config.K_TEMPORAL
    fresh_head.norm.build((None, total_tokens, 768))

    _ = fresh_head(
        chunk_embs=dummy_q,
        support_tokens=dummy_s,
        contrast_tokens=dummy_c,
        temporal_tokens=dummy_t,
        training=False,
    )

    fresh_head.load_weights(config.RATT_WEIGHTS)
    print(f'loaded ratt weights')
    ratt_weights = [
        "20260329-102828_vtest-vid10_db-ratt_db_chunk_encoder_all_vids_overlap_chunks_ret75_k32_sk100_dt005_ch12_L2H8Q32_bs16_acc1_e3_lr1e-03to1e-03_reb3_8e5b_transformer_block_0.pkl",
        "20260329-102828_vtest-vid10_db-ratt_db_chunk_encoder_all_vids_overlap_chunks_ret75_k32_sk100_dt005_ch12_L2H8Q32_bs16_acc1_e3_lr1e-03to1e-03_reb3_8e5b_transformer_block_1.pkl"
    ]
    for i in range(fresh_head.num_layers):
        block = getattr(fresh_head, f"transformer_block_{i}")
        with open(f"rag_weights/{ratt_weights[i]}", "rb") as f:
            print(f'loaded transformer_block_{i}')
            weights = pickle.load(f)
        block.set_weights(weights)
        # with open(f"rag_weights/transformer_block_{i}.pkl", "rb") as f:
        #     print(f'loaded transformer_block_{i}')
        #     weights = pickle.load(f)
        # block.set_weights(weights)

    print("RATT_WEIGHTS:", config.RATT_WEIGHTS)
    print("RATT_WEIGHTS mtime:", os.path.getmtime(config.RATT_WEIGHTS))

    for i in range(config.NUM_LAYERS):
        # p = f"rag_weights/transformer_block_{i}.pkl"
        # print(p, "exists:", os.path.exists(p), "mtime:", os.path.getmtime(p))
        p = f"rag_weights/{ratt_weights[i]}"
        print(p, "exists:", os.path.exists(p), "mtime:", os.path.getmtime(p))
    return fresh_head

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
    # --------------------------------------------------------
    # 1. load training chunks
    # --------------------------------------------------------
    train_vids = config.TRAIN_VIDS
    train_samples = load_samples(train_vids, stride=1)
    print(train_samples[0])
    train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE)

    print(f"Train chunks: {len(train_chunk_samples)}")

    train_dataset = build_tf_dataset_chunks(
        train_chunk_samples,
        batch_size=config.CHUNK_BATCH_SIZE,
        training=False
    )

    # --------------------------------------------------------
    # 2. live retrieval lookups
    # --------------------------------------------------------
    train_chunk_lookup = {make_chunk_key(c): c for c in train_chunk_samples}
    train_future_key_lookup = build_future_key_lookup(
        train_chunk_samples,
        future_step=config.FUTURE_CHUNK_STEP
    )

    # --------------------------------------------------------
    # 3. retrieval db
    # --------------------------------------------------------
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name=config.CHROMADB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # --------------------------------------------------------
    # 4. chunk encoder
    # --------------------------------------------------------
    chunk_encoder = ChunkEncoder(
        hidden_size=768,
        num_layers=4,
        num_heads=8,
        max_frames=config.CHUNK_SIZE
    )

    dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
    _ = chunk_encoder(dummy_frame_embs, training=False)

    chunk_encoder.load_weights(config.STAGE1_WEIGHTS)

    for i in range(chunk_encoder.num_layers):
        block = getattr(chunk_encoder, f"transformer_block_{i}")
        with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
            weights = pickle.load(f)
        block.set_weights(weights)
    chunk_encoder.trainable = False
    print("[MODEL] Loaded chunk encoder weights")

    # --------------------------------------------------------
    # 5. frame cache
    # --------------------------------------------------------
    store_name = "train_val_frames_chunk12_stride4"
    frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

    # --------------------------------------------------------
    # 6. ratt head v2
    # --------------------------------------------------------
    # ratt_head = RATTHeadV2(
    #     hidden_size=768,
    #     num_heads=8,
    #     num_layers=2,
    # )

    # dummy_q = tf.zeros((1, 768), dtype=tf.float32)
    # dummy_s = tf.zeros((1, config.K_SIM, 768), dtype=tf.float32)
    # dummy_c = tf.zeros((1, config.K_CONTRAST, 768), dtype=tf.float32)
    # dummy_t = tf.zeros((1, config.K_TEMPORAL, 768), dtype=tf.float32)

    # _ = ratt_head(
    #     chunk_embs=dummy_q,
    #     support_tokens=dummy_s,
    #     contrast_tokens=dummy_c,
    #     temporal_tokens=dummy_t,
    #     training=False,
    # )

    # path = config.RATT_WEIGHTS
    # import os

    # path = config.RATT_WEIGHTS

    # print("cwd:", os.getcwd())
    # print("path raw:", repr(path))
    # print("abs path:", os.path.abspath(path))
    # print("exists:", os.path.exists(path))
    # print("isfile:", os.path.isfile(path))
    # print("readable:", os.access(path, os.R_OK))

    # parent = os.path.dirname(path) or "."
    # print("parent exists:", os.path.exists(parent))
    # print("parent abs:", os.path.abspath(parent))
    # print("parent contents:", os.listdir(parent) if os.path.exists(parent) else "MISSING")
    # ratt_head.load_weights(config.RATT_WEIGHTS)
    # ratt_head.trainable = False
    # print("[MODEL] Loaded RATTHeadV2 weights")

    # --------------------------------------------------------
    # 7. run inference on training set
    # --------------------------------------------------------
    clip_outputs = defaultdict(list)

    fresh_head = build_and_load_ratt_head()

    for step, batch in enumerate(train_dataset):
        metadata = batch[1]
        labels = batch[2]

        query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
            metadata=metadata,
            chunk_lookup=train_chunk_lookup,
            future_key_lookup=train_future_key_lookup,
            collection=collection,
            chunk_encoder=chunk_encoder,
            frame_emb_mm=frame_emb_mm,
            path_to_idx=path_to_idx
        )

        # THIS matches your best ablation:
        zeros_query = tf.zeros_like(query_embs)

        # class_logits, cls_out, aux = ratt_head(
        #     chunk_embs=zeros_query,
        #     support_tokens=support_tokens,
        #     contrast_tokens=contrast_tokens,
        #     temporal_tokens=temporal_tokens,
        #     training=False,
        # )

        

        class_logits, cls_out, aux = fresh_head(
            chunk_embs=zeros_query,
            support_tokens=support_tokens,
            contrast_tokens=contrast_tokens,
            temporal_tokens=temporal_tokens,
            training=False,
        )

        logits = class_logits.numpy().reshape(-1)
        probs = sigmoid(logits)
        preds = (probs >= 0.5).astype(np.int32)

        batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        # print(logits)
        # print(labels)
        temp = pd.DataFrame()
            
        temp['labels'] = labels.numpy().flatten()
        temp['logits'] = logits.flatten()
        temp['probs'] = probs.flatten()
        print(temp)
        print(f"batch acc={batch_acc:.6f}")
              
        for i in range(len(logits)):
            vid = int(_to_py_scalar(metadata["vid"][i]))
            clip = int(_to_py_scalar(metadata["clip"][i]))
            side = str(_to_py_scalar(metadata["side"][i]))
            t_center = float(_to_py_scalar(metadata["t_center"][i]))
            start_idx = int(_to_py_scalar(metadata["start_idx"][i]))
            end_idx = int(_to_py_scalar(metadata["end_idx"][i]))
            label = int(_to_py_scalar(labels[i]))

            clip_key = f"vid{vid}_clip_{clip}"

            clip_outputs[clip_key].append({
                "vid": vid,
                "clip": clip,
                "side": side,
                "label": label,
                "t_center": t_center,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "pred": int(preds[i]),
            })

            pprint.pprint(clip_outputs[clip_key])
        print(f"[infer] step={step} complete")

    # --------------------------------------------------------
    # 8. sort per clip and save sequences
    # --------------------------------------------------------
    rows = []

    for clip_key, seq in clip_outputs.items():
        seq = sorted(seq, key=lambda x: x["start_idx"])

        raw_sequence = [x["logit"] for x in seq]

        rows.append({
            "clip_key": clip_key,
            "vid": seq[0]["vid"],
            "clip": seq[0]["clip"],
            "side": seq[0]["side"],
            "label": seq[0]["label"],
            "num_chunks": len(seq),
            "start_idxs": [x["start_idx"] for x in seq],
            "end_idxs": [x["end_idx"] for x in seq],
            "t_centers": [x["t_center"] for x in seq],
            "raw_sequence": raw_sequence,
            "z_sequence": z_normalize(raw_sequence).tolist(),
            "prob_sequence": [x["prob"] for x in seq],
            "pred_sequence": [x["pred"] for x in seq],
        })

    rows.sort(key=lambda x: (x["vid"], x["clip"]))

    os.makedirs("test", exist_ok=True)

    out_json = "test/test_new_clip_logit_sequences_live_zeroquery.json"
    out_csv = "test/test_new_clip_logit_sequences_live_zeroquery.csv"

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[DONE] saved {out_json}")
    print(f"[DONE] saved {out_csv}")