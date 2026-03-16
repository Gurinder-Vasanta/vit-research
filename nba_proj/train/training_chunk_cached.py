import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from chromadb import PersistentClient

# srun --jobid=614 --pty bash replace 614 with the job id and then nvidia-smi will work
from dataset import load_samples, build_chunks, build_tf_dataset_chunks
from models.vit_backbone import VisionTransformer
from models.ratt_head import RATTHead
from models.projection_head import ProjectionHead
from retrieval.ratt_chunk_retriever import RattChunkRetriever
from db_maintainence.db_rebuild_chunk import rebuild_db
import config_chunks_cached as config
import time

from transformers import ViTModel, ViTImageProcessor
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle

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

# print("TRAIN CWD:", os.getcwd())
# print("TRAIN chroma path:", os.path.abspath("./chroma_store"))
# input('stop')

# another idea: 
# we can prolly use the manually labelled intervals 
# you can actually have it so that the ragdb collection 
# has all of the manual interval clips (most likely add a piece of metadata called 'manual')
# and then you can throw in the auto labelled ones as well (with mdata 'auto')
# this allows you to: 
# a: have a solid number of clips from each of the videos (even clips from vid5 can be used)
# b: actually use the manually labelled clips for more than just autolabelling 
# c: allow you to keep a set of embeddings that you know are correct so that 
# maybe when you rebuild, you only rebuild things with the tag of 'auto' (or something like that)

# usage: python -m train.training
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_processor.do_rescale = False
hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
hf_vit.eval()

layers = tf_keras.layers


np.random.seed(1234)

def coarse_time_bin(t_center):
    return int(t_center // config.DELTA_T_NORM)

def embed_rep_chunk(rep):
    """
    rep: one chunk dict (like the one you printed)
    Returns: (768,) numpy embedding
    """
    # load frames
    frames = []
    for fp in rep["frames"]:
        img = tf.keras.utils.load_img(fp, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(img)
        frames.append(img)

    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    # HF ViT embedding (already normalized)
    frame_embs = hf_vit_embed_batch(frames)  # (T, 768)

    # pool to chunk embedding (match training semantics)
    mean = frame_embs.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-8)

    return mean.astype(np.float32)

from collections import Counter, defaultdict

def build_retrieval_cache(
    collection,
    all_chunks,
    C=20,
    query_mult=100,
    max_per_video=100,  # 10
    max_global_appearances=5,
    min_time_gap=0.05,   # 0.1
):
    cache = {}

    label_lookup = {}
    for c in all_chunks:
        key = make_key(c["vid"], c["side"], c["t_center"])
        # print(key)
        label_lookup[key] = int(c["label"])
    

    # group all chunks by (side, coarse_time_bin)
    bin_chunks = defaultdict(list)
    for c in all_chunks:
        key = (c["side"], coarse_time_bin(c["t_center"]))
        bin_chunks[key].append(c)

    print(f"[CACHE] Building retrieval cache for {len(bin_chunks)} bins")

    total_count = collection.count()
    global_counts = Counter()
# sig = (vid, side, round(t_center, 4))
    rep_items = list(bin_chunks.items())
    random.shuffle(rep_items)

    # soft penalty for globally overused chunks
    lambda_global = 0.5

    # number of anchors to use per bin
    num_anchors_per_bin = 3

    nums = []
    for s in config.TRAIN_VIDS:
        nums.append(int(s.split('vid')[1]))

    for (side, bin_id), chunks_in_bin in rep_items:
        start = time.perf_counter()

        # -----------------------------
        # choose up to 3 anchors per bin
        # prefer anchors from different videos
        # -----------------------------
        chunks_shuf = list(chunks_in_bin)
        random.shuffle(chunks_shuf)

        by_vid = defaultdict(list)
        for c in chunks_shuf:
            by_vid[int(c["vid"])].append(c)

        candidate_vids = list(by_vid.keys())
        random.shuffle(candidate_vids)

        anchors = []

        # first pass: one per distinct video
        for vid in candidate_vids:
            if len(anchors) >= num_anchors_per_bin:
                break
            anchors.append(by_vid[vid][0])

        # second pass: fill remaining from leftover chunks if needed
        if len(anchors) < num_anchors_per_bin:
            used_ids = set(id(a) for a in anchors)
            for c in chunks_shuf:
                if len(anchors) >= num_anchors_per_bin:
                    break
                if id(c) not in used_ids:
                    anchors.append(c)
                    used_ids.add(id(c))

        # fallback: if somehow no anchors, create empty cache entry
        if len(anchors) == 0:
            cache[(side, bin_id)] = {
                "embeddings": np.zeros((0, 768), dtype=np.float32),
                "vid": np.zeros((0,), dtype=np.int32),
                "side": np.asarray([], dtype=object),
                "t_center": np.zeros((0,), dtype=np.float32),
                "label": np.zeros((0,), dtype=np.int32),
            }
            print(f"[CACHE] ({side}, {bin_id}) missing=0/0")
            print(
                f"[CACHE] ({side}, {bin_id}) "
                f"raw=0 kept=0 time={time.perf_counter() - start:.2f}s"
            )
            continue

        anchor_embs = [embed_rep_chunk(a).tolist() for a in anchors]
        raw_n = min(query_mult * C, total_count)

        result = collection.query(
            query_embeddings=anchor_embs,
            n_results=raw_n,
            where={
                "$and": [
                    {"side": {"$eq": side}},
                    {"vid_num": {"$in": nums}}
                ]
            },
            include=["embeddings", "metadatas", "distances"]
        )
        # print(anchor_embs[0])
        # input(result)

        # -----------------------------------------
        # merge candidates across all anchors
        # keep best score per unique chunk signature
        # -----------------------------------------
        merged = {}

        n_queries = len(anchor_embs)
        for q_idx in range(n_queries):
            raw_embs = np.asarray(result["embeddings"][q_idx], dtype=np.float32)
            raw_metas = result["metadatas"][q_idx]
            raw_dists = result.get("distances", [[None] * len(raw_metas) for _ in range(n_queries)])[q_idx]

            for rank, (emb, m, dist) in enumerate(zip(raw_embs, raw_metas, raw_dists)):
                vid = int(m["vid_num"])
                t_center = float(m["t_center"])
                sig = (vid, side, round(t_center, 5))

                lookup_key = make_key(vid, side, t_center)
                lbl = label_lookup.get(lookup_key, -1)

                if dist is None:
                    base_score = -float(rank)
                else:
                    base_score = -float(dist)

                # keep the best score seen for this candidate across all anchors
                prev = merged.get(sig)
                if prev is None or base_score > prev["base_score"]:
                    merged[sig] = {
                        "emb": emb,
                        "vid": vid,
                        "side": side,
                        "t_center": t_center,
                        "sig": sig,
                        "label": lbl,
                        "base_score": base_score,
                    }

        candidates = list(merged.values())
        candidates.sort(key=lambda x: x["base_score"], reverse=True)

        kept_embs = []
        kept_vids = []
        kept_sides = []
        kept_tcenters = []
        kept_labels = []

        selected_sigs = set()
        video_counts = {}
        video_times = defaultdict(list)

        # greedy selection from merged candidate pool
        while len(kept_embs) < C:
            best_idx = None
            best_score = -1e18

            for i, cand in enumerate(candidates):
                sig = cand["sig"]
                vid = cand["vid"]
                t_center = cand["t_center"]

                if sig in selected_sigs:
                    continue

                if video_counts.get(vid, 0) >= max_per_video:
                    continue

                if global_counts[sig] >= max_global_appearances:
                    continue

                if any(abs(t_center - prev_t) < min_time_gap for prev_t in video_times[vid]):
                    continue

                adjusted_score = cand["base_score"] - lambda_global * global_counts[sig]

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = i

            if best_idx is None:
                break

            cand = candidates[best_idx]
            sig = cand["sig"]
            vid = cand["vid"]
            t_center = cand["t_center"]
            lbl = cand["label"]

            selected_sigs.add(sig)
            video_counts[vid] = video_counts.get(vid, 0) + 1
            video_times[vid].append(t_center)
            global_counts[sig] += 1

            kept_embs.append(cand["emb"])
            kept_vids.append(vid)
            kept_sides.append(side)
            kept_tcenters.append(t_center)
            kept_labels.append(lbl)

            if lbl == -1:
                print("[MISS]", make_key(vid, side, t_center), "raw_t_center=", t_center)

        num_missing = int(np.sum(np.array(kept_labels) == -1))
        print(f"[CACHE] ({side}, {bin_id}) missing={num_missing}/{len(kept_labels)}")

        if len(candidates) > 0:
            emb_dim = candidates[0]["emb"].shape[0]
        else:
            emb_dim = 768

        kept_embs = np.asarray(kept_embs, dtype=np.float32).reshape(-1, emb_dim)
        kept_vids = np.asarray(kept_vids, dtype=np.int32)
        kept_sides = np.asarray(kept_sides)
        kept_tcenters = np.asarray(kept_tcenters, dtype=np.float32)
        kept_labels = np.asarray(kept_labels, dtype=np.int32)

        cache[(side, bin_id)] = {
            "embeddings": kept_embs,
            "vid": kept_vids,
            "side": kept_sides,
            "t_center": kept_tcenters,
            "label": kept_labels,
        }

        # input(cache[(side, bin_id)])
        print(
            f"[CACHE] ({side}, {bin_id}) "
            f"raw={len(candidates)} kept={len(kept_vids)} "
            f"time={time.perf_counter() - start:.2f}s"
        )

    print("Top global chunk frequencies:")
    for sig, cnt in global_counts.most_common(30):
        print(sig, cnt)

    return cache


USE_GRAYSCALE = False

def to_grayscale_3ch(frames_np):
    """
    frames_np: (N, H, W, 3)
    returns:   (N, H, W, 3) grayscale replicated across channels
    """
    # RGB -> grayscale using standard luminance weights
    gray = (
        0.2989 * frames_np[..., 0] +
        0.5870 * frames_np[..., 1] +
        0.1140 * frames_np[..., 2]
    )
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray_3ch = np.stack([gray, gray, gray], axis=-1)
    return gray_3ch

def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, 432, 768, 3) uint8 or float32
    Returns (N, 768) numpy embeddings (L2 normalized)
    """
    # frames_np = frames_np.astype(np.float32)

    # frames_np = frames_np.astype(np.float32)

    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

    if USE_GRAYSCALE:
        frames_np = to_grayscale_3ch(frames_np)
    else: 
        frames_np = frames_np.astype(np.float32)
        
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = hf_processor(images=frames_list, return_tensors="pt").to(device)
        out = hf_vit(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (N,768)
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)
    
def attention_entropy_loss(importance, eps=1e-8):
    ent = -tf.reduce_sum(
        importance * tf.math.log(importance + eps),
        axis=1
    )
    return tf.reduce_mean(ent)

def simple_retrieval_contrastive_loss(q, retrieved, importance):
    """
    q:          (B, D)   projected chunk embeddings (normalized)
    retrieved:  (B, K, D) retrieved embeddings (normalized, stop_grad)
    importance: (B, K)   attention weights (sum to 1)
    """

    # print(retrieved)
    # print(importance)
    # ---- attention-weighted retrieval ----
    # (B, D)
    r_attn = tf.reduce_sum(
        importance[:, :, None] * retrieved,
        axis=1
    )

    # ---- positive pull ----
    # cosine similarity since vectors are normalized
    pos_sim = tf.reduce_sum(q * r_attn, axis=-1)  # (B,)
    pull = 1.0 - pos_sim

    # ---- negative push (batch-shift) ----
    r_attn_neg = tf.roll(r_attn, shift=1, axis=0)
    neg_sim = tf.reduce_sum(q * r_attn_neg, axis=-1)
    push = neg_sim

    # ---- final loss ----
    loss = tf.reduce_mean(pull + push)
    return loss

# ---------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------
# def compute_accuracy(labels, logits):
#     preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
#     return tf.reduce_mean(tf.cast(preds == labels, tf.float32))

def compute_accuracy(labels, logits):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)   # (B,)
    logits = tf.reshape(logits, [-1])                      # (B,)
    preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    # print('raw sigmoid ')
    # print(tf.sigmoid(logits))
    # print('preds')
    # print(preds)
    return tf.reduce_mean(tf.cast(preds == labels, tf.float32))

def find_best_f1(labels, probs):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_f1, best_t



# ---------------------------------------------------------
# TRAIN STEP WITH GRADIENT ACCUMULATION
# ---------------------------------------------------------

class Accumulator:
    def __init__(self, rag_head, proj_head, accum_steps):
        self.accum_steps = accum_steps
        self.step = 0
        self.vars = rag_head.trainable_variables + proj_head.trainable_variables
        self.gradients = [tf.zeros_like(v) for v in self.vars]

    def accumulate(self, grads):
        # self.gradients = [g_old + g_new for g_old, g_new in zip(self.gradients, grads)]
        new_grads = []
        for g_old, g_new in zip(self.gradients, grads):
            if g_new is None:
                new_grads.append(g_old)
            else:
                new_grads.append(g_old + g_new)
        self.gradients = new_grads
        self.step += 1

    def apply(self, optimizer):
        if self.step == self.accum_steps:
            avg_grads = [g / self.accum_steps for g in self.gradients]
            optimizer.apply_gradients(zip(avg_grads, self.vars))
            self.gradients = [tf.zeros_like(v) for v in self.vars]
            self.step = 0

# def get_retrieval_cache(B,metadata,retrieval_cache):
#     retrieved = []
#     retrieved_labels = []

#     for i in range(B):
#         side = metadata["side"][i].numpy().decode()
#         t_center = float(metadata["t_center"][i].numpy())
#         t_width  = float(metadata["t_width"][i].numpy())
#         vid      = int(metadata["vid"][i].numpy())

#         bin_id = coarse_time_bin(t_center)
#         pool = retrieval_cache[(side, bin_id)]

#         mask = (
#             (pool["vid"] != vid) &
#             (np.abs(pool["t_center"] - t_center) <= t_width / 2)
#         )

#         candidates = pool["embeddings"][mask]
#         candidate_labels = pool["label"][mask]

#         if len(candidates) >= config.TOP_K:
#             candidates = candidates[:config.TOP_K]
#             candidate_labels = candidate_labels[:config.TOP_K]
#         else:
#             pad = np.zeros(
#                 (config.TOP_K - len(candidates), candidates.shape[1]),
#                 dtype=np.float32
#             )
#             candidates = np.vstack([candidates, pad])

#             pad_labels = np.zeros(config.TOP_K - len(candidate_labels))
#             candidate_labels = np.concatenate([candidate_labels, pad_labels])

#         retrieved.append(candidates)
#         retrieved_labels.append(candidate_labels)

#     retrieved = tf.nn.l2_normalize(
#         tf.stop_gradient(tf.convert_to_tensor(np.stack(retrieved), tf.float32)),
#         axis=2
#     )

#     retrieved_labels = np.stack(retrieved_labels)
#     return retrieved, retrieved_labels

def get_retrieval_cache(B, metadata, retrieval_cache, label_lookup):
    retrieved = []
    retrieved_labels = []

    for i in range(B):
        side = metadata["side"][i].numpy().decode()
        t_center = float(metadata["t_center"][i].numpy())
        t_width  = float(metadata["t_width"][i].numpy())
        vid      = int(metadata["vid"][i].numpy())

        bin_id = coarse_time_bin(t_center)
        pool = retrieval_cache[(side, bin_id)]

        mask = (pool["vid"] != vid)

        # mask = (
        #     (pool["vid"] != vid) &
        #     (np.abs(pool["t_center"] - t_center) <= t_width / 2)
        # )

        candidates = pool["embeddings"][mask]
        cand_vids = pool["vid"][mask]
        cand_tcenters = pool["t_center"][mask]

        cand_labels = []
        for rv, rt in zip(cand_vids, cand_tcenters):
            key = make_key(rv, side, rt)
            cand_labels.append(label_lookup.get(key, -1))  # -1 = missing

        cand_labels = np.array(cand_labels, dtype=np.int32)

        if len(candidates) >= config.TOP_K:
            candidates = candidates[:config.TOP_K]
            cand_labels = cand_labels[:config.TOP_K]
        else:
            emb_dim = candidates.shape[1] if len(candidates) > 0 else 768
            pad_n = config.TOP_K - len(candidates)

            if len(candidates) == 0:
                candidates = np.zeros((config.TOP_K, emb_dim), dtype=np.float32)
                cand_labels = np.full((config.TOP_K,), -1, dtype=np.int32)
            else:
                pad = np.zeros((pad_n, emb_dim), dtype=np.float32)
                candidates = np.vstack([candidates, pad])

                pad_labels = np.full((pad_n,), -1, dtype=np.int32)
                cand_labels = np.concatenate([cand_labels, pad_labels])

        retrieved.append(candidates)
        retrieved_labels.append(cand_labels)

    retrieved = tf.nn.l2_normalize(
        tf.stop_gradient(tf.convert_to_tensor(np.stack(retrieved), tf.float32)),
        axis=2
    )

    retrieved_labels = np.stack(retrieved_labels)
    return retrieved, retrieved_labels


def supervised_contrastive_loss(z, labels, temperature=0.1):
    """
    z:      (B, D) L2-normalized embeddings
    labels: (B,) int or float labels in {0,1}
    """
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    sim = tf.matmul(z, z, transpose_b=True) / temperature   # (B, B)

    # mask out self-comparisons
    batch_size = tf.shape(z)[0]
    self_mask = tf.eye(batch_size, dtype=tf.bool)

    # positives = same label, not self
    label_eq = tf.equal(labels[:, None], labels[None, :])
    pos_mask = tf.logical_and(label_eq, tf.logical_not(self_mask))

    # for numerical stability
    sim_max = tf.reduce_max(sim, axis=1, keepdims=True)
    sim = sim - sim_max

    exp_sim = tf.exp(sim) * tf.cast(tf.logical_not(self_mask), tf.float32)
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8)

    pos_mask_f = tf.cast(pos_mask, tf.float32)
    pos_count = tf.reduce_sum(pos_mask_f, axis=1)

    # only average over anchors that have at least one positive
    mean_log_prob_pos = tf.reduce_sum(pos_mask_f * log_prob, axis=1) / (pos_count + 1e-8)

    valid = pos_count > 0
    loss = -tf.reduce_mean(tf.boolean_mask(mean_log_prob_pos, valid))
    return loss


def train_step(rag_head, proj_head, retriever, retrieval_cache, optimizer, loss_fn,
               frames, metadata, labels, accum,contrastive_coefficient, print_attention,label_lookup):

    B = tf.shape(frames)[0]
    T = tf.shape(frames)[1]

    # print([B,T])
    
    start = time.perf_counter()
    frames_np = tf.numpy_function(
        hf_vit_embed_batch,
        [tf.reshape(frames, (-1, 432, 768, 3))],
        tf.float32
    )
    end = time.perf_counter()
    # print(f'embed took: {end-start}')
    frame_embs = tf.reshape(frames_np, (B, T, 768))
    deltas = frame_embs[:, 1:, :] - frame_embs[:, :-1, :]

    # ----- Raw chunk pool -----
    # raw_chunk = tf.reduce_mean(frame_embs, axis=1)
    # raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=1) 
    # raw_chunk = tf.stop_gradient(raw_chunk)

    # print()
    # print('frame embds')
    # input(frame_embs)
    
    # for 
    # try this next 
    mean = tf.reduce_mean(frame_embs, axis=1)
    mean_deltas = tf.reduce_mean(deltas, axis=1)
    std_deltas = tf.math.reduce_std(deltas, axis=1)

    # std  = tf.math.reduce_std(frame_embs, axis=1)
    # max_ = tf.reduce_max(frame_embs, axis=1)

    # raw_chunk = tf.concat([mean, std, max_], axis=-1)
    raw_chunk = tf.concat([mean, mean_deltas, std_deltas], axis=-1)
    raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=1)
    
    # input(raw_chunk)
    # ----- Forward projection (learnable) -----
    with tf.GradientTape() as tape:
        chunk_embs = proj_head(raw_chunk, training=True)
        chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)

        # print()
        # print('chunk embds')
        # input(chunk_embs)
        # ----- Retrieval (stop gradient) -----
        start = time.perf_counter()

        retrieved, retrieved_labels = get_retrieval_cache(B,metadata,retrieval_cache,label_lookup)
        # retrieved = retriever(chunk_embs,metadata)
        
        end = time.perf_counter()
        # print(f'retrieval took: {end - start}')
        # print()
        # print('retrieved embds')
        # input(retrieved)


        # if(disable_cls == True): 
        #     chunk_embs = tf.zeros_like(chunk_embs)
        # logits, fused_cls, attn_scores = rag_head(chunk_embs, retrieved, disable_cls=disable_cls, training=True)
        
        class_logit, relevance_logit, fused_cls, attn_scores = rag_head(chunk_embs, retrieved, disable_cls=disable_cls, training=True)

        # print()
        # print('-------- raw class then relevance logits ----------')
        # print(class_logit)
        # print()
        # print()
        # print(relevance_logit)
        # print()
        # print()
        relevance = 0 #tf.nn.sigmoid(relevance_logit)
        # print(relevance)
        # print('---------------- end logits ----------------')
        # print()
        
        cls_to_ret = attn_scores[-1][:, :, 0, 1:]
        # print(cls_to_ret)
        importance = tf.reduce_mean(cls_to_ret, axis=1)

        
        # print(importance)

        # doesnt affect learning: 
        # importance = importance + tf.one_hot(0, K) * 0.2
        # importance = importance / tf.reduce_sum(importance, axis=1, keepdims=True)

        # ----- NEW: simple retrieval-aware contrast -----
        loss_contrast = simple_retrieval_contrastive_loss(chunk_embs, retrieved, importance)
        loss_entropy = attention_entropy_loss(importance)

        importance = tf.reduce_mean(cls_to_ret, axis=1).numpy()
        # if(print_attention == True):
        #     print('----------------------------------------------- start attention score')
        #     print(importance)
        #     print('-------------------------------------------------------------- end attn score')
        # loss_cls = loss_fn(labels, logits)
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, (-1, 1))

        chunk_loss = tf.keras.losses.binary_crossentropy(
            labels,
            class_logit,
            from_logits=True
        )

        
        # loss_cls = tf.reduce_mean(relevance * chunk_loss)
        # loss_cls = loss_fn(labels, logits)
        loss_cls = loss_fn(labels, class_logit)
        

        z = chunk_embs                           # (B, D)
        z = tf.nn.l2_normalize(z, axis=1)

        labels_flat = tf.reshape(tf.cast(labels, tf.int32), [-1])
        loss_supcon = supervised_contrastive_loss(z, labels_flat, temperature=0.1)

        sim_matrix = tf.matmul(z, z, transpose_b=True)  # (B, B)

        batch_size = tf.shape(z)[0]
        labels_ibn = tf.range(batch_size)

        loss_ibn = tf.keras.losses.sparse_categorical_crossentropy(
            labels_ibn,
            sim_matrix,
            from_logits=True
        )
        loss_ibn = tf.reduce_mean(loss_ibn)
        
        loss_entropy_diff = -tf.reduce_mean(
            relevance * tf.math.log(relevance + 1e-8)
        )

        # labels = labels.numpy().reshape(-1)
        # agreement = (retrieved_labels == labels[:, None])

        query_labels = labels.numpy().reshape(-1)   # (B,)
        valid_mask = (retrieved_labels != -1)       # (B, K)

        matches = (retrieved_labels == query_labels[:, None]) & valid_mask
        agreement = matches.sum() / np.maximum(valid_mask.sum(), 1)

        per_query_agreement = matches.sum(axis=1) / np.maximum(valid_mask.sum(axis=1), 1)
        mean_agreement = per_query_agreement.mean()

        # agreement = agreement.mean()

        entropy = -tf.reduce_sum(
            importance * tf.math.log(importance + 1e-8),
            axis=1
        )
        print()
        print("retrieval attention entropy:", tf.reduce_mean(entropy))

        print(f'TRAIN MEAN agreement: {agreement.mean()} TRAIN PER QUERY MEAN agreement: {mean_agreement}')
        
        # (B, D)
        q = chunk_embs

        # (B, K, D)
        r = retrieved

        # cosine similarity to retrieved
        sim_pos = tf.reduce_mean(
            tf.reduce_sum(q[:, None, :] * r, axis=-1),
            axis=1
        )

        # random negatives from batch
        r_neg = tf.roll(r, shift=1, axis=0)

        sim_neg = tf.reduce_mean(
            tf.reduce_sum(q[:, None, :] * r_neg, axis=-1),
            axis=1
        )

        gap = sim_pos - sim_neg
        print(f"retrieval similarity gap: {tf.reduce_mean(gap)}")
        print()

        # contrastive_coefficient = 0
        # combine them
        loss = (loss_cls 
                + contrastive_coefficient * 1 * loss_contrast 
                + 0.1 * 0 * loss_ibn 
                + 0.1 * 0 * loss_entropy
                + 0.1 * 0 * loss_supcon)       # λ = 0.1 to start
        # loss = loss_cls +  0.1 * loss_ibn  + 0.1 * loss_entropy      # λ = 0.1 to start

    # Get grads for BOTH heads
    grads = tape.gradient(loss,
            rag_head.trainable_variables + proj_head.trainable_variables)

    # optimizer.apply_gradients(zip(grads, rag_head.trainable_variables + proj_head.trainable_variables))
    accum.accumulate(grads)
    accum.apply(optimizer)

    # agreement = None
    # print(labels)
    # print(logits)
    # input('stop')
    acc = compute_accuracy(labels, class_logit)
    # true_acc = compute_true_accuracy(labels, logits)
    # print(acc)
    # print(true_acc)
    # input('stop')
    # print("loss:", float(loss), "acc:", float(acc))
    return float(loss), float(acc), agreement

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
def evaluate(val_ds, rag_head, proj_head, retriever, retrieval_cache, loss_fn,label_lookup):
    losses = []
    accs = []

    comb_sims = []
    retr_sims = []
    
    c1s = []
    c2s = []

    all_val_logits = []
    all_val_labels = []

    for frames, metadata, labels in val_ds:

        B = tf.shape(frames)[0]
        T = tf.shape(frames)[1]  # 60

        # input(frames)

        frames_np = tf.numpy_function(
            hf_vit_embed_batch,
            [tf.reshape(frames, (-1, 432, 768, 3))],
            tf.float32
        )
        frame_embs = tf.reshape(frames_np, (B, T, 768))

        # ---- chunk pooling ----
        # chunk_embs = tf.reduce_max(frame_embs, axis=1)  # (B, 768) #TODO: reduce to max
        # raw_chunk = tf.reduce_mean(frame_embs, axis=1)
        # raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=-1)

        # try this next 
        # mean = tf.reduce_mean(frame_embs, axis=1)
        # std  = tf.math.reduce_std(frame_embs, axis=1)
        # max_ = tf.reduce_max(frame_embs, axis=1)

        deltas = frame_embs[:, 1:, :] - frame_embs[:, :-1, :]

        mean = tf.reduce_mean(frame_embs, axis=1)
        mean_deltas = tf.reduce_mean(deltas, axis=1)
        std_deltas = tf.math.reduce_std(deltas, axis=1)

        # raw_chunk = tf.concat([mean, std, max_], axis=-1)
        raw_chunk = tf.concat([mean, mean_deltas, std_deltas], axis=-1)
        raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=1)

        chunk_embs = proj_head(raw_chunk, training=False)
        chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)
        
        # retrieved_np = retriever(chunk_embs, metadata)

        # --- new retriever start ----
        retrieved,retrieved_labels = get_retrieval_cache(B,metadata,retrieval_cache,label_lookup)
        
        # retrieved = retriever(chunk_embs,metadata)
        # ---- new retriever end -----

        # logits, fused_cls, attn_scores = rag_head(chunk_embs, retrieved, disable_cls=False, training=False)

        class_logit, relevance_logit, fused_cls, attn_scores = rag_head(chunk_embs, retrieved, disable_cls=False, training=False)

        # print()
        # print('-------- raw class then relevance logits ----------')
        # print(class_logit)
        # print()
        # print()
        # print(relevance_logit)
        # print()
        # print()
        relevance = 0#tf.nn.sigmoid(relevance_logit)
        # print(relevance)
        # print('---------------- end logits ----------------')
        # print()

        all_val_logits.append(class_logit.numpy())
        all_val_labels.append(labels.numpy())

        # print()
        # print('eval logits')
        # print(logits)
        # print()

        # --- retrieved similarity ---
        # retrieved shape: (B, top_k, 768)
        r1 = retrieved[0]
        r2 = retrieved[1]



        # (B, D)
        q = chunk_embs

        # (B, K, D)
        r = retrieved

        # cosine similarity to retrieved
        sim_pos = tf.reduce_mean(
            tf.reduce_sum(q[:, None, :] * r, axis=-1),
            axis=1
        )

        # random negatives from batch
        r_neg = tf.roll(r, shift=1, axis=0)

        sim_neg = tf.reduce_mean(
            tf.reduce_sum(q[:, None, :] * r_neg, axis=-1),
            axis=1
        )

        gap = sim_pos - sim_neg
        print(f"retrieval similarity gap: {tf.reduce_mean(gap)}")

        # entropy = -tf.reduce_sum(
        #     importance * tf.math.log(importance + 1e-8),
        #     axis=1
        # )

        # print("retrieval attention entropy:", tf.reduce_mean(entropy))

        # REAL cosine similarity (correct)
        cos_cls = -tf.keras.losses.cosine_similarity(chunk_embs[0], chunk_embs[1])
        print()
        # print("CLS true cosine similarity:", float(cos_cls))

        # Retrieved similarity (REAL)
        cos_retr = -tf.reduce_mean(tf.keras.losses.cosine_similarity(r1, r2))
        print("Retrieved true cosine similarity:", float(cos_retr))
        retr_sims.append(cos_retr)

        # # --- combined feature similarity (optional) ---
        z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)

        # Combined similarity (REAL)
        cos_comb = -tf.keras.losses.cosine_similarity(z1, z2)
        print("Combined true cosine similarity:", float(cos_comb))
        comb_sims.append(cos_comb)

        q1 = chunk_embs[0]
        q2 = chunk_embs[1]

        cos_qr1 = -tf.reduce_mean(
            tf.keras.losses.cosine_similarity(q1[None, :], r1)
        )
        cos_qr2 = -tf.reduce_mean(
            tf.keras.losses.cosine_similarity(q2[None, :], r2)
        )

        print(f'C1 retrieval purity: {cos_qr1}, C2 retrieval purity: {cos_qr2}')
        c1s.append(cos_qr1)
        c2s.append(cos_qr2)
        # val_probs = 1 / (1 + np.exp(-logits))  # sigmoid
        # roc_auc = roc_auc_score(labels, val_probs)
        # # print("ROC-AUC:", roc_auc)

        # val_preds = (val_probs > 0.5).astype(int)
        # f1_default = f1_score(labels, val_preds)
        # # print("F1 @ 0.5 threshold:", f1_default)

        # best_f1, best_threshold = find_best_f1(labels, val_probs)
        # # print("Best F1:", best_f1)
        # # print("Best threshold:", best_threshold)

        # print(f'ROC-AUC: {roc_auc} F1 @ 0.5 threshold: {f1_default}')

        # cos_cls = 1 - tf.keras.losses.cosine_similarity(chunk_embs[0], chunk_embs[1])
        # print("CLS cosine similarity:", float(cos_cls))

       
        # # average similarity over top-k retrieved vectors
        # cos_retr = tf.reduce_mean(1 - tf.keras.losses.cosine_similarity(r1, r2))
        # print("Retrieved cosine similarity:", float(cos_retr))

        # # --- combined feature similarity (optional) ---
        # z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        # z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)
        # cos_comb = 1 - tf.keras.losses.cosine_similarity(z1, z2)
        # print("Combined feature similarity:", float(cos_comb))
        # print("--------------------------------------------------")
        
        # loss = loss_fn(labels, logits)

        # relevance = tf.nn.sigmoid(relevance_logit)

        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, (-1, 1))

        chunk_loss = tf.keras.losses.binary_crossentropy(
            labels,
            class_logit,
            from_logits=True
        )

        # loss = tf.reduce_mean(relevance * chunk_loss)

        loss_entropy = -tf.reduce_mean(
            relevance * tf.math.log(relevance + 1e-8)
        )

        loss = loss_fn(labels, class_logit)
        # loss = loss + 0.01 * loss_entropy
        acc = compute_accuracy(labels, class_logit)

        print(f"val batch loss: {loss:.4f}, val batch acc: {acc:.4f}")
        print()

        losses.append(loss.numpy())
        accs.append(acc.numpy())

    all_val_logits = np.concatenate(all_val_logits, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    val_probs = 1 / (1 + np.exp(-all_val_logits))  # sigmoid
    # roc_auc = roc_auc_score(all_val_labels, val_probs)
    # print("ROC-AUC:", roc_auc)

    val_preds = (val_probs > 0.5).astype(int)

    # labels = labels.numpy().reshape(-1)
    # agreement = (retrieved_labels == labels[:, None])
    
    query_labels = labels.numpy().reshape(-1)   # (B,)
    valid_mask = (retrieved_labels != -1)       # (B, K)

    matches = (retrieved_labels == query_labels[:, None]) & valid_mask
    agreement = matches.sum() / np.maximum(valid_mask.sum(), 1)

    per_query_agreement = matches.sum(axis=1) / np.maximum(valid_mask.sum(axis=1), 1)
    mean_agreement = per_query_agreement.mean()

    # agreement = agreement.mean()

    
    
    # f1_default = f1_score(all_val_labels, val_preds)
    # print("F1 @ 0.5 threshold:", f1_default)

    # best_f1, best_threshold = find_best_f1(all_val_labels, val_probs)
    # print("Best F1:", best_f1)
    # print("Best threshold:", best_threshold)

    # print(f'ROC-AUC: {roc_auc} F1 @ 0.5 threshold: {f1_default}')
    # print(f'Best F1: {best_f1} Best threshold: {best_threshold}')
    print(f"VAL loss: {np.mean(losses):.4f}, VAL acc: {np.mean(accs):.4f}")
    print(f'MEAN comb sim:  {np.mean(comb_sims):.4f}, MEAN retr sim: {np.mean(retr_sims):.4f}')
    print(f'STDEV comb sim:  {np.std(comb_sims):.4f}, STDEV retr sim: {np.std(retr_sims):.4f}')
    print(f'MEAN c1s:  {np.mean(c1s):.4f}, MEAN c2s: {np.mean(c2s):.4f}')
    print(f'LABEL agreement: {agreement}, MEAN agreement: {agreement.mean()}')
    print(f'TEST agreement: {matches[0]}\n MEAN agreement: {agreement.mean()}')
    print(f'TEST PER QUERY agreement: {per_query_agreement}\n MEAN agreement: {mean_agreement}')
    print("----------------------------------------------------")

def make_key(vid, side, t_center, ndigits=5):
    return (int(vid), str(side), round(float(t_center), ndigits))
# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = config.TRAIN_VIDS
    train_samples = load_samples(train_vids,stride=1)
    train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE)

    label_lookup = {}
    for c in train_chunk_samples:
        key = make_key(c["vid"], c["side"], c["t_center"])
        label_lookup[key] = int(c["label"])

    test_vids = config.TEST_VIDS
    test_samples = load_samples(test_vids,stride=1)
    test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE)

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # split 95/5
    # n = len(chunk_samples)
    print(len(train_chunk_samples))
    print(len(test_chunk_samples))
    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    # train_chunks = chunk_samples[config.START_CHUNK_TRAIN:config.END_CHUNK_TRAIN] #was 2000
    # val_chunks = chunk_samples[config.START_CHUNK_VALID:config.END_CHUNK_VALID]

    # train_chunk_samples = train_chunk_samples[0:100]
    # test_chunk_samples = test_chunk_samples[0:32]
    print(f"Train chunks: {len(train_chunk_samples)}")
    print(f"Val chunks:   {len(test_chunk_samples)}")

    # ---------------------------------------------
    # 2. Build TF datasets
    # ---------------------------------------------
    train_dataset = build_tf_dataset_chunks(train_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=True)
    val_dataset   = build_tf_dataset_chunks(test_chunk_samples,   batch_size=config.CHUNK_BATCH_SIZE, training=False)

    print(f"Train dataset: {(train_dataset)}")
    print(f"Val dataset:   {(val_dataset)}")

    # ---------------------------------------------
    # 3. Build models
    # ---------------------------------------------
    
    # RAG head (trainable)
    ratt_head = RATTHead(hidden_size=768, num_queries=config.NUM_QUERIES, num_layers=config.NUM_LAYERS,num_heads=config.NUM_HEADS)

    proj_head = ProjectionHead(input_dim=2304, hidden_dim=768*8, proj_dim=768)

    # proj_head.load_weights("projection_head.weights.h5")
    
    # Retrieval DB
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name=config.CHROMADB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    
    retriever = RattChunkRetriever(collection, top_k=config.TOP_K, search_k=config.SEARCH_K)

    dummy_chunk = tf.zeros((1, 768))
    dummy_retrieved = tf.zeros((1, 10, 768))
    _ = ratt_head(dummy_chunk, dummy_retrieved, disable_cls=False, training=False)
    # rag_head.save_weights(config.RAG_WEIGHTS)
    # rag_head.load_weights('rag_head_5vid_new_v2.weights.h5')

    dummy = tf.zeros((1, 2304), dtype=tf.float32)
    _ = proj_head(dummy)   # builds variables
    # proj_head.save_weights(config.PROJ_WEIGHTS)
    # proj_head.load_weights("projection_head_5vid_new_v2.weights.h5")

    # ---------------------------------------------
    # 4. Optimizer / loss
    # ---------------------------------------------
    optimizer = tf.keras.optimizers.Adam(config.PHASE_1_LEARNING_RATE)
    # optimizer = torch.optim.AdamW([
    #     {"params": hf_vit.encoder.layer[-1].parameters(), "lr": 1e-6},
    #     {"params": hf_vit.layernorm.parameters(), "lr": 1e-6},
    #     {"params": rag_head.parameters(), "lr": 1e-5},
    #     {"params": projector.parameters(), "lr": 1e-5},
    # ])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # ---------------------------------------------
    # 5. Training loop
    # ---------------------------------------------
    EPOCHS = config.EPOCHS

    # lower lr to 1e-5 and make contrastive coefficient 0.1 at epoch 24 (rebuild every 6)
    # stopped at epoch 39 (so if you were to start this run again, start it at epoch 40)
    # results look great, val loss (old treshold > 0.5 loss) is 0.2171, train loss is 0.2649, train acc is 0.75
    accum_steps = config.ACCUM_BATCH_SIZE  # effective batch = physical batch * accum_steps
    accum = Accumulator(ratt_head, proj_head, accum_steps)

    contrastive_coefficient = 0.1
    for epoch in range(1,EPOCHS+1):
        disable_cls = False
        # if(epoch < 5):
        #     continue
        print(f"\n================= EPOCH {epoch} =================")
        print('collection count in training ')
        print(collection.count())
        losses = []
        accs = []
        agrees = []
        batch_counter = 0
        
        
        if(epoch >= int(EPOCHS/2)+1): 
            optimizer.learning_rate.assign(config.PHASE_2_LEARNING_RATE)
            # contrastive_coefficient = config.PHASE_2_CONTRASTIVE_LOSS
        else: 
            optimizer.learning_rate.assign(config.PHASE_1_LEARNING_RATE)
            # contrastive_coefficient = config.PHASE_1_CONTRASTIVE_LOSS
        print_attention = False
        if(epoch >= 6):
            print_attention = True
        
        # retrieval_cache = build_retrieval_cache(
        #     collection,
        #     train_chunks,
        #     C=config.SEARCH_K
        # )

        

        if os.path.exists(config.CACHE_PATH):
            retrieval_cache = load_retrieval_cache(config.CACHE_PATH)
        else:
            retrieval_cache = build_retrieval_cache(
                collection,
                train_chunk_samples,
                C=config.BUILD_RET_C
            )
            save_retrieval_cache(retrieval_cache, config.CACHE_PATH)

        # input('stop')
        # retrieval_cache = None
        for frames_batch, metadata_batch, labels_batch in train_dataset:
            curloss, curacc, curagree = train_step(
                ratt_head, proj_head, retriever, retrieval_cache,
                optimizer, bce,
                frames_batch, metadata_batch, labels_batch,
                accum,contrastive_coefficient,print_attention,
                label_lookup
            )

            batch_counter += 1
            losses.append(curloss)
            accs.append(curacc)
            agrees.append(curagree)
            if(batch_counter % config.PRINT_EVERY == 0):
                print(f"EPOCH {epoch} BATCH {batch_counter} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f}")
        print(f"EPOCH {epoch} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f} TRAIN agree: {np.mean(agrees):.4f}")

        w = proj_head.trainable_variables[0].numpy()
        print(
            "[TRAIN] proj W0 mean/std:",
            float(w.mean()),
            float(w.std())
        )

        # CTRLF
        proj_head.save_weights(config.PROJ_WEIGHTS)
        ratt_head.save_weights(config.RATT_WEIGHTS)
        # validation at end of every epoch
        evaluate(val_dataset, ratt_head, proj_head, retriever, retrieval_cache,bce,label_lookup)
        # rebuild_db()
        if(epoch % config.ADJUST_CONTRASTIVE_LOSS_EVERY == 0 and epoch >= config.ADJUST_CONTRASTIVE_LOSS_EVERY): 
            contrastive_coefficient += config.INCREMENT_CONTRASTIVE_LOSS_BY
        if((epoch) % config.REBUILD_EVERY == 0 and (epoch) >= config.REBUILD_EVERY): 
            # CTRLF
            rebuild_db()
            client = PersistentClient(path="./chroma_store")
            collection = client.get_or_create_collection(
                name=config.CHROMADB_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            retriever = RattChunkRetriever(collection, top_k=config.TOP_K, search_k=config.SEARCH_K)

            # CTRLF
            retrieval_cache = build_retrieval_cache(
                collection,
                train_chunk_samples,
                C=config.BUILD_RET_C
            )
            save_retrieval_cache(retrieval_cache, config.CACHE_PATH)
