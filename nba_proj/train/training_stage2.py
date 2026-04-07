# from collections import defaultdict
# import os
# import pprint
# import random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import tensorflow.keras as tf_keras
# from chromadb import PersistentClient

# from dataset import load_samples, build_chunks, build_tf_dataset_chunks
# from models.vit_backbone import VisionTransformer
# from models.ratt_head import RATTHead
# from models.projection_head import ProjectionHead
# from retrieval.ratt_chunk_retriever import RattChunkRetriever
# from models.candidate_reranker import CandidateReranker
# import config_stage2 as config
# import time

# from transformers import ViTModel, ViTImageProcessor
# import torch

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
# import pickle

# from models.chunk_encoder import ChunkEncoder
# from models.ratt_v2 import RATTHeadV2

# import gc

# tf.keras.backend.clear_session()
# gc.collect()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# hf_processor.do_rescale = False
# hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
# hf_vit.eval()

# layers = tf_keras.layers

# SEED = 12
# os.environ["PYTHONHASHSEED"] = str(SEED)

# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# try:
#     tf.config.experimental.enable_op_determinism()
# except Exception:
#     pass

# ALIGN_LOSS_WEIGHT = 0.3
# TEACHER_K = config.K_SIM

# # gradual teacher->student support schedule
# MIX_LAMBDA_START = 0.90
# MIX_LAMBDA_END = 0.00
# MIX_SCHEDULE = "linear"  # "linear" or "cosine"


# def get_mix_lambda(epoch_idx, total_epochs):
#     if total_epochs <= 1:
#         return MIX_LAMBDA_END

#     t = epoch_idx / float(total_epochs - 1)
#     t = min(max(t, 0.0), 1.0)

#     if MIX_SCHEDULE == "cosine":
#         alpha = 0.5 * (1.0 + np.cos(np.pi * t))
#     else:
#         alpha = 1.0 - t

#     return float(MIX_LAMBDA_END + (MIX_LAMBDA_START - MIX_LAMBDA_END) * alpha)


# class SupportSelector(tf.keras.Model):
#     def __init__(self, hidden_dim=256, name="support_selector"):
#         super().__init__(name=name)
#         self.d1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
#         self.d2 = tf.keras.layers.Dense(256, activation="relu")
#         self.d3 = tf.keras.layers.Dense(1)

#     def call(self, query_embs, support_tokens, support_mask=None, training=False):
#         q = tf.expand_dims(query_embs, axis=1)
#         q = tf.repeat(q, tf.shape(support_tokens)[1], axis=1)

#         x = tf.concat(
#             [q, support_tokens, q - support_tokens, q * support_tokens],
#             axis=-1
#         )

#         h = self.d1(x, training=training)
#         h = self.d2(h, training=training)
#         scores = self.d3(h, training=training)[..., 0]

#         if support_mask is not None:
#             very_neg = tf.constant(-1e9, dtype=scores.dtype)
#             scores = tf.where(support_mask > 0, scores, very_neg)

#         alpha = tf.nn.softmax(scores, axis=-1)

#         pooled_support_rep = tf.reduce_sum(
#             support_tokens * alpha[..., None],
#             axis=1
#         )

#         pooled_support_token = tf.expand_dims(pooled_support_rep, axis=1)

#         return pooled_support_token, pooled_support_rep, alpha, scores


# def make_chunk_key(chunk, precision=6):
#     return (
#         int(chunk["vid"]),
#         str(chunk["side"]),
#         int(chunk["clip"]),
#         int(chunk["start_idx"]),
#         int(chunk["end_idx"]),
#     )


# def build_future_key_lookup(all_chunks, future_step=5):
#     grouped = defaultdict(list)

#     for chunk in all_chunks:
#         grouped[(int(chunk["vid"]), int(chunk["clip"]))].append(chunk)

#     future_key_lookup = {}

#     for _, group in grouped.items():
#         group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))
#         last_idx = len(group_sorted) - 1

#         for idx, chunk in enumerate(group_sorted):
#             cur_key = make_chunk_key(chunk)
#             fut_idx = min(idx + future_step, last_idx)
#             fut_key = make_chunk_key(group_sorted[fut_idx])
#             future_key_lookup[cur_key] = fut_key

#     return future_key_lookup


# CACHE_DIR = "./frame_cache_vit"


# def get_store_paths(store_name):
#     return {
#         "emb": os.path.join(CACHE_DIR, f"{store_name}_emb.dat"),
#         "paths": os.path.join(CACHE_DIR, f"{store_name}_paths.npy"),
#         "meta": os.path.join(CACHE_DIR, f"{store_name}_meta.npz"),
#     }


# def load_frame_store(store_name="train_val_frames"):
#     paths = get_store_paths(store_name)
#     meta = np.load(paths["meta"])

#     n_frames = int(meta["n_frames"])
#     emb_dim = int(meta["emb_dim"])

#     emb_mm = np.memmap(
#         paths["emb"],
#         dtype="float32",
#         mode="r",
#         shape=(n_frames, emb_dim),
#     )

#     frame_paths = np.load(paths["paths"], allow_pickle=True).tolist()
#     path_to_idx = {p: i for i, p in enumerate(frame_paths)}

#     print(f"[frame cache] loaded store '{store_name}'")
#     print(f"[frame cache] shape = ({n_frames}, {emb_dim})")

#     return emb_mm, frame_paths, path_to_idx


# def hf_vit_embed_batch(frames_np):
#     if frames_np.dtype != np.uint8:
#         frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

#     frames_np = frames_np.astype(np.float32)

#     frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
#     with torch.no_grad():
#         inputs = hf_processor(images=frames_list, return_tensors="pt").to(device)
#         out = hf_vit(**inputs)
#         cls = out.last_hidden_state[:, 0, :]
#         cls = cls.cpu().numpy()
#         cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
#         return cls.astype(np.float32)


# def save_encoded_embeddings(encoded, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "wb") as f:
#         pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"[CACHE] Saved encoded embeddings to {path}")


# def load_encoded_embeddings(path):
#     with open(path, "rb") as f:
#         cache = pickle.load(f)
#     print(f"[CACHE] Loaded encoded embeddings from {path}")
#     return cache


# def query_collection(query_emb, collection, n_results):
#     results = collection.query(
#         query_embeddings=[query_emb.tolist()],
#         n_results=n_results,
#         include=["embeddings", "metadatas"]
#     )

#     raw_embs = results["embeddings"][0]
#     raw_meta = results["metadatas"][0]

#     out = []
#     for emb, meta in zip(raw_embs, raw_meta):
#         out.append(
#             {
#                 "emb": np.asarray(emb, dtype=np.float32),
#                 "meta": {
#                     "label": int(meta["label"]),
#                     "status": str(meta["status"]),
#                     "status_id": int(meta["status_id"]),
#                     "side": str(meta["side"]),
#                     "vid": int(meta["vid_num"]),
#                     "clip": int(meta["clip_num"]),
#                     "t_center": float(meta["t_center"]),
#                     "t_width": float(meta["t_width"]),
#                     "start_idx": int(meta["start_idx"]),
#                     "end_idx": int(meta["end_idx"]),
#                     "class_logit": float(meta["class_logit"])
#                 },
#             }
#         )
#     return out


# def extract_meta(chunk):
#     return {
#         "label": int(chunk["label"]),
#         "status": str(chunk["status"]),
#         "status_id": int(chunk["status_id"]),
#         "side": str(chunk["side"]),
#         "vid": int(chunk["vid"]),
#         "clip": int(chunk["clip"]),
#         "t_center": float(chunk["t_center"]),
#         "t_width": float(chunk["t_width"]),
#         "start_idx": int(chunk["start_idx"]),
#         "end_idx": int(chunk["end_idx"]),
#     }


# def same_chunk_meta(meta_a, meta_b):
#     return (
#         int(meta_a["vid"]) == int(meta_b["vid"])
#         and str(meta_a["side"]) == str(meta_b["side"])
#         and int(meta_a["clip"]) == int(meta_b["clip"])
#         and int(meta_a["start_idx"]) == int(meta_b["start_idx"])
#         and int(meta_a["end_idx"]) == int(meta_b["end_idx"])
#     )


# def dedup_signature(meta):
#     return (
#         int(meta["vid"]),
#         str(meta["side"]),
#         int(meta["clip"]),
#         int(meta["start_idx"]),
#         int(meta["end_idx"]),
#     )


# def pad_or_trim(items, k, emb_dim, pad_meta_template):
#     if len(items) >= k:
#         items = items[:k]
#     else:
#         pad_count = k - len(items)
#         zero_emb = np.zeros((emb_dim,), dtype=np.float32)
#         for _ in range(pad_count):
#             items.append(
#                 {
#                     "emb": zero_emb.copy(),
#                     "meta": dict(pad_meta_template),
#                 }
#             )

#     embs = np.stack([x["emb"] for x in items], axis=0)
#     metas = [x["meta"] for x in items]
#     return embs, metas


# def encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx):
#     idxs = [path_to_idx[p] for p in chunk["frames"]]
#     frame_embs = frame_emb_mm[idxs].astype(np.float32)
#     frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)

#     stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)
#     return stage1_chunk_emb[0].numpy().astype(np.float32)


# def build_teacher_support_items(query_meta, content_candidates, k_teacher):
#     teacher_items = []
#     seen = set()

#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)

#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) != int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue

#         teacher_items.append(cand)
#         seen.add(sig)

#         if len(teacher_items) >= k_teacher:
#             break

#     return teacher_items


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
#     k_teacher=TEACHER_K,
# ):
#     query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)

#     query_meta = extract_meta(chunk)
#     future_meta = extract_meta(future_chunk)

#     emb_dim = query_emb.shape[0]

#     pad_meta_template = {
#         "label": -1,
#         "status": "PAD",
#         "status_id": -1,
#         "side": "PAD",
#         "vid": -1,
#         "clip": -1,
#         "t_center": -1.0,
#         "t_width": -1.0,
#         "start_idx": -1,
#         "end_idx": -1,
#     }

#     content_candidates = query_collection(query_emb, collection, search_k_content)

#     sim_items = []
#     contrast_items = []
#     used_content = set()
#     filtered_candidates = []

#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)

#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in used_content:
#             continue

#         filtered_candidates.append(cand)
#         used_content.add(sig)

#     sim_items = filtered_candidates[:k_sim]
#     remaining = filtered_candidates[k_sim:]

#     if len(filtered_candidates) >= 150:
#         contrast_pool = filtered_candidates[50:150]
#     else:
#         half = len(remaining) // 2
#         contrast_pool = remaining[half:]

#     if len(contrast_pool) < k_contrast:
#         contrast_pool = remaining

#     if len(contrast_pool) > k_contrast:
#         contrast_items = random.sample(contrast_pool, k_contrast)
#     else:
#         contrast_items = contrast_pool[:k_contrast]

#     sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)
#     contrast_embs, contrast_meta = pad_or_trim(
#         contrast_items, k_contrast, emb_dim, pad_meta_template
#     )

#     teacher_items = build_teacher_support_items(query_meta, content_candidates, k_teacher)
#     teacher_support_embs, teacher_support_meta = pad_or_trim(
#         teacher_items, k_teacher, emb_dim, pad_meta_template
#     )

#     temporal_candidates = query_collection(future_emb, collection, search_k_temporal)

#     temporal_items = []
#     seen_temporal = set()

#     for cand in temporal_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)

#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in seen_temporal:
#             continue

#         temporal_items.append(cand)
#         seen_temporal.add(sig)

#         if len(temporal_items) >= k_temporal:
#             break

#     temporal_embs, temporal_meta = pad_or_trim(
#         temporal_items, k_temporal, emb_dim, pad_meta_template
#     )

#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "teacher_support_embs": teacher_support_embs,
#         "query_meta": query_meta,
#         "future_meta": future_meta,
#         "sim_meta": sim_meta,
#         "contrast_meta": contrast_meta,
#         "temporal_meta": temporal_meta,
#         "teacher_support_meta": teacher_support_meta,
#     }


# def _to_py_scalar(x):
#     if hasattr(x, "numpy"):
#         x = x.numpy()
#     if isinstance(x, np.ndarray):
#         if x.ndim == 0:
#             x = x.item()
#         else:
#             raise ValueError(f"Expected scalar, got array shape {x.shape}")
#     if isinstance(x, bytes):
#         x = x.decode("utf-8")
#     return x


# def make_chunk_key_from_meta(metadata, i, precision=6):
#     vid = _to_py_scalar(metadata["vid"][i])
#     side = _to_py_scalar(metadata["side"][i])
#     clip = _to_py_scalar(metadata["clip"][i])
#     start_idx = _to_py_scalar(metadata["start_idx"][i])
#     end_idx = _to_py_scalar(metadata["end_idx"][i])

#     return (
#         int(vid),
#         str(side),
#         int(clip),
#         int(start_idx),
#         int(end_idx),
#     )


# def fetch_live_batch(
#     metadata,
#     chunk_lookup,
#     future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     return_teacher=False,
# ):
#     batch_size = metadata["vid"].shape[0]

#     query_embs = []
#     support_tokens = []
#     contrast_tokens = []
#     temporal_tokens = []
#     teacher_support_tokens = []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_key = future_key_lookup[key]
#         future_chunk = chunk_lookup[future_key]

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
#             k_teacher=TEACHER_K,
#         )

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#         if return_teacher:
#             teacher_support_tokens.append(entry["teacher_support_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     if not return_teacher:
#         return query_embs, support_tokens, contrast_tokens, temporal_tokens

#     teacher_support_tokens = tf.convert_to_tensor(
#         np.stack(teacher_support_tokens, axis=0), dtype=tf.float32
#     )

#     return (
#         query_embs,
#         support_tokens,
#         contrast_tokens,
#         temporal_tokens,
#         teacher_support_tokens,
#     )


# scce_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True,
#     reduction="none"
# )


# def weighted_scce_loss(labels, logits, class_weights):
#     per_example_loss = scce_no_reduce(labels, logits)
#     weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
#     return tf.reduce_mean(per_example_loss * weights)


# def compute_class_weights(chunk_samples, power=0.5):
#     labels = np.array([int(c["status_id"]) for c in chunk_samples], dtype=np.int32)

#     counts = np.bincount(labels, minlength=3).astype(np.float32)
#     if np.any(counts == 0):
#         raise ValueError(f"Missing class in training data. Counts: {counts}")

#     max_count = counts.max()
#     weights = (max_count / counts) ** power
#     weights = weights / weights[0]

#     print("train class counts:", counts.astype(int).tolist())
#     print("class weights:", weights.tolist())

#     return tf.constant(weights, dtype=tf.float32)


# def support_mask_from_tokens(support_tokens):
#     token_norm = tf.reduce_sum(tf.abs(support_tokens), axis=-1)
#     return tf.cast(token_norm > 0.0, tf.float32)


# def grad_rms(g):
#     if g is None:
#         return 0.0
#     g = tf.cast(g, tf.float32)
#     return float(tf.sqrt(tf.reduce_mean(tf.square(g))).numpy())


# def mix_teacher_student_support(student_support_rep, teacher_support_rep, mix_lambda):
#     mixed_rep = (1.0 - mix_lambda) * student_support_rep + mix_lambda * tf.stop_gradient(teacher_support_rep)
#     mixed_token = tf.expand_dims(mixed_rep, axis=1)
#     return mixed_token, mixed_rep


# def train_step(
#     batch,
#     ratt_head,
#     support_selector,
#     train_chunk_lookup,
#     train_future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     class_weights,
#     mix_lambda,
# ):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     (
#         query_embs,
#         support_tokens,
#         contrast_tokens,
#         temporal_tokens,
#         teacher_support_tokens,
#     ) = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=True,
#     )

#     support_mask = support_mask_from_tokens(support_tokens)
#     zeros_contrast = tf.zeros_like(contrast_tokens)

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(query_embs)
#         tape.watch(support_tokens)
#         tape.watch(temporal_tokens)

#         student_support_token, student_support_rep, support_alpha, support_scores = support_selector(
#             query_embs=query_embs,
#             support_tokens=support_tokens,
#             support_mask=support_mask,
#             training=True,
#         )

#         teacher_support_rep = tf.reduce_mean(teacher_support_tokens, axis=1)

#         mixed_support_token, mixed_support_rep = mix_teacher_student_support(
#             student_support_rep=student_support_rep,
#             teacher_support_rep=teacher_support_rep,
#             mix_lambda=mix_lambda,
#         )

#         class_logits, cls_out, aux = ratt_head(
#             chunk_embs=query_embs,
#             support_tokens=mixed_support_token,
#             contrast_tokens=zeros_contrast,
#             temporal_tokens=temporal_tokens,
#             training=True,
#         )

#         cls_loss = weighted_scce_loss(labels, class_logits, class_weights)

#         align_loss = tf.reduce_mean(
#             tf.reduce_sum(
#                 tf.square(student_support_rep - tf.stop_gradient(teacher_support_rep)),
#                 axis=-1,
#             )
#         )

#         loss = cls_loss + ALIGN_LOSS_WEIGHT * align_loss

#         if ratt_head.losses:
#             loss += tf.add_n(ratt_head.losses)

#     train_vars = ratt_head.trainable_variables + support_selector.trainable_variables
#     grads = tape.gradient(loss, train_vars)
#     optimizer.apply_gradients(zip(grads, train_vars))

#     g_query = tape.gradient(loss, query_embs)
#     g_support = tape.gradient(loss, support_tokens)
#     g_temporal = tape.gradient(loss, temporal_tokens)

#     print(
#         f"branch_grad_rms | "
#         f"query={grad_rms(g_query):.6f} "
#         f"support={grad_rms(g_support):.6f} "
#         f"temporal={grad_rms(g_temporal):.6f}"
#     )
#     print({
#         "mix_lambda": float(mix_lambda),
#         "cls_loss": float(cls_loss.numpy()),
#         "align_loss": float(align_loss.numpy()),
#         "loss": float(loss.numpy()),
#     })

#     del tape

#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)
#     probs = tf.nn.softmax(class_logits, axis=-1)

#     train_loss_metric.update_state(loss)
#     train_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()
#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print(
#         "alpha stats:",
#         "mix_lambda=", float(mix_lambda),
#         "align_w=", ALIGN_LOSS_WEIGHT,
#         "mean=", tf.reduce_mean(support_alpha).numpy(),
#         "std=", tf.math.reduce_std(support_alpha).numpy()
#     )

#     return {
#         "loss": float(loss.numpy()),
#         "acc": float(train_acc_metric.result().numpy()),
#         "logits": class_logits,
#         "probs": probs,
#         "cls_out": cls_out,
#         "aux": aux,
#         "support_alpha": support_alpha,
#         "mix_lambda": float(mix_lambda),
#     }


# def eval_step(
#     batch,
#     ratt_head,
#     support_selector,
#     val_chunk_lookup,
#     val_future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     class_weights,
# ):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=val_chunk_lookup,
#         future_key_lookup=val_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     support_mask = support_mask_from_tokens(support_tokens)
#     zeros_contrast = tf.zeros_like(contrast_tokens)

#     student_support_token, student_support_rep, support_alpha, support_scores = support_selector(
#         query_embs=query_embs,
#         support_tokens=support_tokens,
#         support_mask=support_mask,
#         training=False,
#     )

#     class_logits, cls_out, aux = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_token,
#         contrast_tokens=zeros_contrast,
#         temporal_tokens=temporal_tokens,
#         training=False,
#     )

#     loss = weighted_scce_loss(labels, class_logits, class_weights)
#     if ratt_head.losses:
#         loss += tf.add_n(ratt_head.losses)

#     probs = tf.nn.softmax(class_logits, axis=-1)
#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)

#     val_loss_metric.update_state(loss)
#     val_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()

#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print(
#         "val alpha stats:",
#         "align_w=", ALIGN_LOSS_WEIGHT,
#         "mean=", tf.reduce_mean(support_alpha).numpy(),
#         "std=", tf.math.reduce_std(support_alpha).numpy()
#     )

#     return {
#         "loss": float(loss.numpy()),
#         "acc": float(val_acc_metric.result().numpy()),
#         "logits": class_logits,
#         "probs": probs,
#         "labels": labels,
#         "support_alpha": support_alpha,
#     }


# def run_train_epoch(
#     train_ds,
#     ratt_head,
#     support_selector,
#     train_chunk_lookup,
#     train_future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     class_weight,
#     mix_lambda,
# ):
#     train_loss_metric.reset_state()
#     train_acc_metric.reset_state()

#     for step, batch in enumerate(train_ds):
#         out = train_step(
#             batch=batch,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             train_chunk_lookup=train_chunk_lookup,
#             train_future_key_lookup=train_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weights=class_weight,
#             mix_lambda=mix_lambda,
#         )

#         if step % 1 == 0:
#             print(
#                 f"[train] step={step} "
#                 f"mix_lambda={mix_lambda:.4f} "
#                 f"loss={out['loss']:.4f} "
#                 f"acc={train_acc_metric.result().numpy():.4f}"
#             )

#     return (
#         float(train_loss_metric.result().numpy()),
#         float(train_acc_metric.result().numpy()),
#     )


# def run_val_epoch(
#     val_ds,
#     ratt_head,
#     support_selector,
#     val_chunk_lookup,
#     val_future_key_lookup,
#     collection,
#     chunk_encoder,
#     frame_emb_mm,
#     path_to_idx,
#     class_weight,
# ):
#     val_loss_metric.reset_state()
#     val_acc_metric.reset_state()

#     for step, batch in enumerate(val_ds):
#         out = eval_step(
#             batch=batch,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             val_chunk_lookup=val_chunk_lookup,
#             val_future_key_lookup=val_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weights=class_weight,
#         )

#         if step % 1 == 0:
#             print(
#                 f"[val] step={step} "
#                 f"loss={out['loss']:.4f} "
#                 f"running acc={val_acc_metric.result().numpy():.4f}"
#             )

#     return (
#         float(val_loss_metric.result().numpy()),
#         float(val_acc_metric.result().numpy()),
#     )


# if __name__ == "__main__":

#     print("SEARCH_K_CONTENT", config.SEARCH_K_CONTENT)
#     print("SEARCH_K_TEMPORAL", config.SEARCH_K_TEMPORAL)
#     print("K_SIM", config.K_SIM)
#     print("K_CONTRAST", config.K_CONTRAST)
#     print("K_TEMPORAL", config.K_TEMPORAL)
#     print("FUTURE_CHUNK_STEP", config.FUTURE_CHUNK_STEP)
#     print("CHUNK_SIZE", config.CHUNK_SIZE)
#     print("CHROMADB_COLLECTION", config.CHROMADB_COLLECTION)
#     print("STAGE1_WEIGHTS", config.STAGE1_WEIGHTS)
#     print("RATT_WEIGHTS", config.RATT_WEIGHTS)
#     print("ALIGN_LOSS_WEIGHT", ALIGN_LOSS_WEIGHT)
#     print("TEACHER_K", TEACHER_K)
#     print("MIX_LAMBDA_START", MIX_LAMBDA_START)
#     print("MIX_LAMBDA_END", MIX_LAMBDA_END)
#     print("MIX_SCHEDULE", MIX_SCHEDULE)

#     train_vids = config.TRAIN_VIDS
#     train_samples = load_samples(train_vids, stride=1)
#     train_chunk_samples = build_chunks(
#         train_samples,
#         chunk_size=config.CHUNK_SIZE,
#         chunk_stride=config.CHUNK_STRIDE
#     )

#     test_vids = config.TEST_VIDS
#     test_samples = load_samples(test_vids, stride=1)
#     test_chunk_samples = build_chunks(
#         test_samples,
#         chunk_size=config.CHUNK_SIZE,
#         chunk_stride=config.CHUNK_STRIDE
#     )

#     random.shuffle(train_samples)
#     random.shuffle(test_samples)

#     print(len(train_chunk_samples))
#     print(len(test_chunk_samples))

#     class_weight = compute_class_weights(train_chunk_samples)

#     print(f"Train chunks: {len(train_chunk_samples)}")
#     print(f"Val chunks:   {len(test_chunk_samples)}")

#     train_dataset = build_tf_dataset_chunks(
#         train_chunk_samples,
#         batch_size=config.CHUNK_BATCH_SIZE,
#         training=True
#     )
#     val_dataset = build_tf_dataset_chunks(
#         test_chunk_samples,
#         batch_size=config.CHUNK_BATCH_SIZE,
#         training=False
#     )

#     print(f"Train dataset: {train_dataset}")
#     print(f"Val dataset:   {val_dataset}")

#     client = PersistentClient(path="./chroma_store")
#     collection = client.get_or_create_collection(
#         name=config.CHROMADB_COLLECTION,
#         metadata={"hnsw:space": "cosine"}
#     )

#     chunk_encoder = ChunkEncoder(
#         hidden_size=768,
#         num_layers=4,
#         num_heads=8,
#         max_frames=config.CHUNK_SIZE
#     )

#     dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
#     _ = chunk_encoder(dummy_frame_embs, training=False)

#     print("==== DEBUG STAGE2 ====")
#     print("cwd:", os.getcwd())
#     print("weights path:", config.STAGE1_WEIGHTS)
#     print("abs path:", os.path.abspath(config.STAGE1_WEIGHTS))
#     print("exists:", os.path.exists(config.STAGE1_WEIGHTS))
#     print("======================")

#     chunk_encoder.load_weights(config.STAGE1_WEIGHTS)

#     for i in range(chunk_encoder.num_layers):
#         block = getattr(chunk_encoder, f"transformer_block_{i}")
#         with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
#             weights = pickle.load(f)
#         block.set_weights(weights)

#     print("[STAGE1] Loaded chunk encoder weights")

#     chunk_encoder.trainable = False
#     print("[STAGE1] Chunk encoder frozen")

#     store_name = "train_val_frames_chunk8_stride2"
#     frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

#     train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
#     train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

#     val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
#     val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

#     batch = next(iter(train_dataset))

#     ratt_head = RATTHeadV2(
#         hidden_size=768,
#         num_heads=8,
#         num_layers=config.NUM_LAYERS,
#     )

#     support_selector = SupportSelector(hidden_dim=1024)

#     train_chunk_lookup = {make_chunk_key(c): c for c in train_chunk_samples}
#     train_future_key_lookup = build_future_key_lookup(train_chunk_samples, future_step=5)

#     val_chunk_lookup = {make_chunk_key(c): c for c in test_chunk_samples}
#     val_future_key_lookup = build_future_key_lookup(test_chunk_samples, future_step=5)

#     query_embs, support, contrast, temporal = fetch_live_batch(
#         metadata=batch[1],
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     _ = support_selector(
#         query_embs,
#         support,
#         support_mask=support_mask_from_tokens(support),
#         training=False
#     )

#     for v in ratt_head.trainable_variables[:5]:
#         print(v.name, float(tf.reduce_mean(v).numpy()), float(tf.math.reduce_std(v).numpy()))

#     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

#     print(query_embs.shape)
#     print(support.shape)
#     print(contrast.shape)
#     print(temporal.shape)

#     student_support_token, student_support_rep, support_alpha, _ = support_selector(
#         query_embs, support, support_mask=support_mask_from_tokens(support), training=False
#     )

#     logits, _, _ = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_token,
#         contrast_tokens=tf.zeros_like(contrast),
#         temporal_tokens=temporal,
#         training=False,
#     )
#     print(logits)
#     print("logits:", logits.shape)

#     for epoch in range(config.EPOCHS):
#         print(f"\\n===== EPOCH {epoch+1}/{config.EPOCHS} =====")
#         mix_lambda = get_mix_lambda(epoch_idx=epoch, total_epochs=config.EPOCHS)
#         print(f"[epoch {epoch+1}] mix_lambda={mix_lambda:.4f}")

#         train_loss, train_acc = run_train_epoch(
#             train_ds=train_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             train_chunk_lookup=train_chunk_lookup,
#             train_future_key_lookup=train_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#             mix_lambda=mix_lambda,
#         )

#         val_loss, val_acc = run_val_epoch(
#             val_ds=val_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             val_chunk_lookup=val_chunk_lookup,
#             val_future_key_lookup=val_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#         )

#         print(
#             f"[epoch {epoch+1}] "
#             f"mix_lambda={mix_lambda:.4f} "
#             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
#             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
#         )

#     ratt_head.save_weights(config.RATT_WEIGHTS)
#     print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")

#     os.makedirs("rag_weights", exist_ok=True)

#     for i in range(config.NUM_LAYERS):
#         block = getattr(ratt_head, f"transformer_block_{i}")
#         with open(f"rag_weights/{config.RUN_ID}_transformer_block_{i}.pkl", "wb") as f:
#             pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"[MAIN] saved transformer block {i} weights")






# See conversation rewrite: teacher-forced support+temporal curriculum based on uploaded training script.
# This file was generated to keep the original structure but:
# 1) builds teacher support and teacher temporal pools
# 2) mixes teacher/student support+temporal branch inputs during training
# 3) uses only student branches at validation/inference
# 4) keeps the 3-class objective

# from collections import defaultdict
# import os
# import pprint
# import random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from chromadb import PersistentClient
# from dataset import load_samples, build_chunks, build_tf_dataset_chunks
# import config_stage2 as config
# from transformers import ViTModel, ViTImageProcessor
# import torch
# import pickle
# from models.chunk_encoder import ChunkEncoder
# from models.ratt_v2 import RATTHeadV2
# import gc

# tf.keras.backend.clear_session()
# gc.collect()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# hf_processor.do_rescale = False
# hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
# hf_vit.eval()

# SEED = 12
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# try:
#     tf.config.experimental.enable_op_determinism()
# except Exception:
#     pass

# ALIGN_LOSS_WEIGHT = 0.3
# TEMP_ALIGN_LOSS_WEIGHT = 0.3
# TEACHER_K = config.K_SIM
# MIX_LAMBDA_START = 0.8
# MIX_LAMBDA_END = 0.00
# MIX_SCHEDULE = "cosine"


# def get_mix_lambda(epoch_idx, total_epochs):
#     if total_epochs <= 1:
#         return MIX_LAMBDA_END
#     t = epoch_idx / float(total_epochs - 1)
#     t = min(max(t, 0.0), 1.0)
#     if MIX_SCHEDULE == "cosine":
#         alpha = 0.5 * (1.0 + np.cos(np.pi * t))
#     else:
#         alpha = 1.0 - t
#     return float(MIX_LAMBDA_END + (MIX_LAMBDA_START - MIX_LAMBDA_END) * alpha)


# class SupportSelector(tf.keras.Model):
#     def __init__(self, hidden_dim=256, name="support_selector"):
#         super().__init__(name=name)
#         self.d1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
#         self.d2 = tf.keras.layers.Dense(256, activation="relu")
#         self.d3 = tf.keras.layers.Dense(1)

#     def call(self, query_embs, support_tokens, support_mask=None, training=False):
#         q = tf.expand_dims(query_embs, axis=1)
#         q = tf.repeat(q, tf.shape(support_tokens)[1], axis=1)
#         x = tf.concat([q, support_tokens, q - support_tokens, q * support_tokens], axis=-1)
#         h = self.d1(x, training=training)
#         h = self.d2(h, training=training)
#         scores = self.d3(h, training=training)[..., 0]
#         if support_mask is not None:
#             scores = tf.where(support_mask > 0, scores, tf.constant(-1e9, dtype=scores.dtype))
#         alpha = tf.nn.softmax(scores, axis=-1)
#         pooled_rep = tf.reduce_sum(support_tokens * alpha[..., None], axis=1)
#         pooled_token = tf.expand_dims(pooled_rep, axis=1)
#         return pooled_token, pooled_rep, alpha, scores


# def make_chunk_key(chunk, precision=6):
#     return (int(chunk["vid"]), str(chunk["side"]), int(chunk["clip"]), int(chunk["start_idx"]), int(chunk["end_idx"]))


# def build_future_key_lookup(all_chunks, future_step=5):
#     grouped = defaultdict(list)
#     for chunk in all_chunks:
#         grouped[(int(chunk["vid"]), int(chunk["clip"]))].append(chunk)
#     future_key_lookup = {}
#     for _, group in grouped.items():
#         group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))
#         last_idx = len(group_sorted) - 1
#         for idx, chunk in enumerate(group_sorted):
#             cur_key = make_chunk_key(chunk)
#             fut_idx = min(idx + future_step, last_idx)
#             fut_key = make_chunk_key(group_sorted[fut_idx])
#             future_key_lookup[cur_key] = fut_key
#     return future_key_lookup


# CACHE_DIR = "./frame_cache_vit"


# def get_store_paths(store_name):
#     return {
#         "emb": os.path.join(CACHE_DIR, f"{store_name}_emb.dat"),
#         "paths": os.path.join(CACHE_DIR, f"{store_name}_paths.npy"),
#         "meta": os.path.join(CACHE_DIR, f"{store_name}_meta.npz"),
#     }


# def load_frame_store(store_name="train_val_frames"):
#     paths = get_store_paths(store_name)
#     meta = np.load(paths["meta"])
#     n_frames = int(meta["n_frames"])
#     emb_dim = int(meta["emb_dim"])
#     emb_mm = np.memmap(paths["emb"], dtype="float32", mode="r", shape=(n_frames, emb_dim))
#     frame_paths = np.load(paths["paths"], allow_pickle=True).tolist()
#     path_to_idx = {p: i for i, p in enumerate(frame_paths)}
#     print(f"[frame cache] loaded store '{store_name}'")
#     print(f"[frame cache] shape = ({n_frames}, {emb_dim})")
#     return emb_mm, frame_paths, path_to_idx


# def query_collection(query_emb, collection, n_results):
#     results = collection.query(
#         query_embeddings=[query_emb.tolist()],
#         n_results=n_results,
#         include=["embeddings", "metadatas"]
#     )
#     raw_embs = results["embeddings"][0]
#     raw_meta = results["metadatas"][0]
#     out = []
#     for emb, meta in zip(raw_embs, raw_meta):
#         out.append({"emb": np.asarray(emb, dtype=np.float32), "meta": {
#             "label": int(meta["label"]),
#             "status": str(meta["status"]),
#             "status_id": int(meta["status_id"]),
#             "side": str(meta["side"]),
#             "vid": int(meta["vid_num"]),
#             "clip": int(meta["clip_num"]),
#             "t_center": float(meta["t_center"]),
#             "t_width": float(meta["t_width"]),
#             "start_idx": int(meta["start_idx"]),
#             "end_idx": int(meta["end_idx"]),
#             "class_logit": float(meta["class_logit"])
#         }})
#     return out


# def extract_meta(chunk):
#     return {
#         "label": int(chunk["label"]),
#         "status": str(chunk["status"]),
#         "status_id": int(chunk["status_id"]),
#         "side": str(chunk["side"]),
#         "vid": int(chunk["vid"]),
#         "clip": int(chunk["clip"]),
#         "t_center": float(chunk["t_center"]),
#         "t_width": float(chunk["t_width"]),
#         "start_idx": int(chunk["start_idx"]),
#         "end_idx": int(chunk["end_idx"]),
#     }


# def same_chunk_meta(meta_a, meta_b):
#     return (
#         int(meta_a["vid"]) == int(meta_b["vid"])
#         and str(meta_a["side"]) == str(meta_b["side"])
#         and int(meta_a["clip"]) == int(meta_b["clip"])
#         and int(meta_a["start_idx"]) == int(meta_b["start_idx"])
#         and int(meta_a["end_idx"]) == int(meta_b["end_idx"])
#     )


# def dedup_signature(meta):
#     return (int(meta["vid"]), str(meta["side"]), int(meta["clip"]), int(meta["start_idx"]), int(meta["end_idx"]))


# def pad_or_trim(items, k, emb_dim, pad_meta_template):
#     items = list(items)
#     if len(items) >= k:
#         items = items[:k]
#     else:
#         pad_count = k - len(items)
#         zero_emb = np.zeros((emb_dim,), dtype=np.float32)
#         for _ in range(pad_count):
#             items.append({"emb": zero_emb.copy(), "meta": dict(pad_meta_template)})
#     embs = np.stack([x["emb"] for x in items], axis=0)
#     metas = [x["meta"] for x in items]
#     return embs, metas


# def encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx):
#     idxs = [path_to_idx[p] for p in chunk["frames"]]
#     frame_embs = frame_emb_mm[idxs].astype(np.float32)
#     frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)
#     stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)
#     return stage1_chunk_emb[0].numpy().astype(np.float32)


# def build_teacher_support_items(query_meta, content_candidates, k_teacher):
#     teacher_items, seen = [], set()
#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) != int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue
#         teacher_items.append(cand)
#         seen.add(sig)
#         if len(teacher_items) >= k_teacher:
#             break
#     return teacher_items


# def build_teacher_temporal_items(query_meta, temporal_candidates, k_teacher):
#     teacher_items, seen = [], set()
#     for cand in temporal_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) != int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue
#         teacher_items.append(cand)
#         seen.add(sig)
#         if len(teacher_items) >= k_teacher:
#             break
#     return teacher_items


# def build_live_entry(chunk, future_chunk, collection, chunk_encoder, frame_emb_mm, path_to_idx,
#                      search_k_content, search_k_temporal, k_sim, k_contrast, k_temporal, k_teacher=TEACHER_K):
#     query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     query_meta = extract_meta(chunk)
#     emb_dim = query_emb.shape[0]
#     pad_meta_template = {
#         "label": -1, "status": "PAD", "status_id": -1, "side": "PAD", "vid": -1,
#         "clip": -1, "t_center": -1.0, "t_width": -1.0, "start_idx": -1, "end_idx": -1,
#     }

#     content_candidates = query_collection(query_emb, collection, search_k_content)
#     used_content, filtered_candidates = set(), []
#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in used_content:
#             continue
#         filtered_candidates.append(cand)
#         used_content.add(sig)

#     sim_items = filtered_candidates[:k_sim]
#     remaining = filtered_candidates[k_sim:]
#     if len(filtered_candidates) >= 150:
#         contrast_pool = filtered_candidates[50:150]
#     else:
#         half = len(remaining) // 2
#         contrast_pool = remaining[half:]
#     if len(contrast_pool) < k_contrast:
#         contrast_pool = remaining
#     contrast_items = random.sample(contrast_pool, k_contrast) if len(contrast_pool) > k_contrast else contrast_pool[:k_contrast]

#     sim_embs, sim_meta = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)
#     contrast_embs, contrast_meta = pad_or_trim(contrast_items, k_contrast, emb_dim, pad_meta_template)

#     teacher_support_items = build_teacher_support_items(query_meta, content_candidates, k_teacher)
#     teacher_support_embs, teacher_support_meta = pad_or_trim(teacher_support_items, k_teacher, emb_dim, pad_meta_template)

#     temporal_candidates = query_collection(future_emb, collection, search_k_temporal)
#     temporal_items, seen_temporal = [], set()
#     for cand in temporal_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in seen_temporal:
#             continue
#         temporal_items.append(cand)
#         seen_temporal.add(sig)
#         if len(temporal_items) >= k_temporal:
#             break

#     temporal_embs, temporal_meta = pad_or_trim(temporal_items, k_temporal, emb_dim, pad_meta_template)
#     teacher_temporal_items = build_teacher_temporal_items(query_meta, temporal_candidates, k_teacher)
#     teacher_temporal_embs, teacher_temporal_meta = pad_or_trim(teacher_temporal_items, k_teacher, emb_dim, pad_meta_template)

#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "teacher_support_embs": teacher_support_embs,
#         "teacher_temporal_embs": teacher_temporal_embs,
#     }


# def _to_py_scalar(x):
#     if hasattr(x, "numpy"):
#         x = x.numpy()
#     if isinstance(x, np.ndarray):
#         if x.ndim == 0:
#             x = x.item()
#         else:
#             raise ValueError(f"Expected scalar, got array shape {x.shape}")
#     if isinstance(x, bytes):
#         x = x.decode("utf-8")
#     return x


# def make_chunk_key_from_meta(metadata, i, precision=6):
#     return (
#         int(_to_py_scalar(metadata["vid"][i])),
#         str(_to_py_scalar(metadata["side"][i])),
#         int(_to_py_scalar(metadata["clip"][i])),
#         int(_to_py_scalar(metadata["start_idx"][i])),
#         int(_to_py_scalar(metadata["end_idx"][i])),
#     )


# def fetch_live_batch(metadata, chunk_lookup, future_key_lookup, collection, chunk_encoder, frame_emb_mm, path_to_idx, return_teacher=False):
#     batch_size = metadata["vid"].shape[0]
#     query_embs, support_tokens, contrast_tokens, temporal_tokens = [], [], [], []
#     teacher_support_tokens, teacher_temporal_tokens = [], []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_chunk = chunk_lookup[future_key_lookup[key]]
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
#             k_teacher=TEACHER_K,
#         )
#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])
#         if return_teacher:
#             teacher_support_tokens.append(entry["teacher_support_embs"])
#             teacher_temporal_tokens.append(entry["teacher_temporal_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     if not return_teacher:
#         return query_embs, support_tokens, contrast_tokens, temporal_tokens

#     teacher_support_tokens = tf.convert_to_tensor(np.stack(teacher_support_tokens, axis=0), dtype=tf.float32)
#     teacher_temporal_tokens = tf.convert_to_tensor(np.stack(teacher_temporal_tokens, axis=0), dtype=tf.float32)
#     return query_embs, support_tokens, contrast_tokens, temporal_tokens, teacher_support_tokens, teacher_temporal_tokens


# scce_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


# def weighted_scce_loss(labels, logits, class_weights):
#     per_example_loss = scce_no_reduce(labels, logits)
#     weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
#     return tf.reduce_mean(per_example_loss * weights)


# def compute_class_weights(chunk_samples, power=0.5):
#     labels = np.array([int(c["status_id"]) for c in chunk_samples], dtype=np.int32)
#     counts = np.bincount(labels, minlength=3).astype(np.float32)
#     if np.any(counts == 0):
#         raise ValueError(f"Missing class in training data. Counts: {counts}")
#     max_count = counts.max()
#     weights = (max_count / counts) ** power
#     weights = weights / weights[0]
#     print("train class counts:", counts.astype(int).tolist())
#     print("class weights:", weights.tolist())
#     return tf.constant(weights, dtype=tf.float32)


# def support_mask_from_tokens(tokens):
#     token_norm = tf.reduce_sum(tf.abs(tokens), axis=-1)
#     return tf.cast(token_norm > 0.0, tf.float32)


# def grad_rms(g):
#     if g is None:
#         return 0.0
#     g = tf.cast(g, tf.float32)
#     return float(tf.sqrt(tf.reduce_mean(tf.square(g))).numpy())


# def mix_teacher_student_rep(student_rep, teacher_rep, mix_lambda):
#     mixed_rep = (1.0 - mix_lambda) * student_rep + mix_lambda * tf.stop_gradient(teacher_rep)
#     mixed_token = tf.expand_dims(mixed_rep, axis=1)
#     return mixed_token, mixed_rep


# def train_step(batch, ratt_head, support_selector, temporal_selector,
#                train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#                frame_emb_mm, path_to_idx, class_weights, mix_lambda):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     (query_embs, support_tokens, contrast_tokens, temporal_tokens,
#      teacher_support_tokens, teacher_temporal_tokens) = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=True,
#     )

#     support_mask = support_mask_from_tokens(support_tokens)
#     temporal_mask = support_mask_from_tokens(temporal_tokens)
#     zeros_contrast = tf.zeros_like(contrast_tokens)

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(query_embs)
#         tape.watch(support_tokens)
#         tape.watch(temporal_tokens)

#         student_support_token, student_support_rep, support_alpha, support_scores = support_selector(
#             query_embs=query_embs, support_tokens=support_tokens, support_mask=support_mask, training=True
#         )
#         student_temporal_token, student_temporal_rep, temporal_alpha, temporal_scores = temporal_selector(
#             query_embs=query_embs, support_tokens=temporal_tokens, support_mask=temporal_mask, training=True
#         )

#         teacher_support_rep = tf.reduce_mean(teacher_support_tokens, axis=1)
#         teacher_temporal_rep = tf.reduce_mean(teacher_temporal_tokens, axis=1)

#         mixed_support_token, mixed_support_rep = mix_teacher_student_rep(student_support_rep, teacher_support_rep, mix_lambda)
#         mixed_temporal_token, mixed_temporal_rep = mix_teacher_student_rep(student_temporal_rep, teacher_temporal_rep, mix_lambda)

#         class_logits, cls_out, aux = ratt_head(
#             chunk_embs=query_embs,
#             support_tokens=mixed_support_token,
#             contrast_tokens=zeros_contrast,
#             temporal_tokens=mixed_temporal_token,
#             training=True,
#         )

#         cls_loss = weighted_scce_loss(labels, class_logits, class_weights)
#         align_loss = tf.reduce_mean(tf.reduce_sum(tf.square(student_support_rep - tf.stop_gradient(teacher_support_rep)), axis=-1))
#         temp_align_loss = tf.reduce_mean(tf.reduce_sum(tf.square(student_temporal_rep - tf.stop_gradient(teacher_temporal_rep)), axis=-1))
#         loss = cls_loss + ALIGN_LOSS_WEIGHT * align_loss + TEMP_ALIGN_LOSS_WEIGHT * temp_align_loss
#         if ratt_head.losses:
#             loss += tf.add_n(ratt_head.losses)

#     train_vars = ratt_head.trainable_variables + support_selector.trainable_variables + temporal_selector.trainable_variables
#     grads = tape.gradient(loss, train_vars)
#     optimizer.apply_gradients(zip(grads, train_vars))

#     g_query = tape.gradient(loss, query_embs)
#     g_support = tape.gradient(loss, support_tokens)
#     g_temporal = tape.gradient(loss, temporal_tokens)

#     print(f"branch_grad_rms | query={grad_rms(g_query):.6f} support={grad_rms(g_support):.6f} temporal={grad_rms(g_temporal):.6f}")
#     print({"mix_lambda": float(mix_lambda), "cls_loss": float(cls_loss.numpy()), "align_loss": float(align_loss.numpy()), "temp_align_loss": float(temp_align_loss.numpy()), "loss": float(loss.numpy())})
#     del tape

#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)
#     probs = tf.nn.softmax(class_logits, axis=-1)

#     train_loss_metric.update_state(loss)
#     train_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()
#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print("support alpha stats:", "mix_lambda=", float(mix_lambda), "align_w=", ALIGN_LOSS_WEIGHT, "mean=", tf.reduce_mean(support_alpha).numpy(), "std=", tf.math.reduce_std(support_alpha).numpy())
#     print("temporal alpha stats:", "mix_lambda=", float(mix_lambda), "align_w=", TEMP_ALIGN_LOSS_WEIGHT, "mean=", tf.reduce_mean(temporal_alpha).numpy(), "std=", tf.math.reduce_std(temporal_alpha).numpy())

#     return {"loss": float(loss.numpy()), "acc": float(train_acc_metric.result().numpy()), "logits": class_logits, "probs": probs}


# def eval_step(batch, ratt_head, support_selector, temporal_selector,
#               val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#               frame_emb_mm, path_to_idx, class_weights):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=val_chunk_lookup,
#         future_key_lookup=val_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     support_mask = support_mask_from_tokens(support_tokens)
#     temporal_mask = support_mask_from_tokens(temporal_tokens)
#     zeros_contrast = tf.zeros_like(contrast_tokens)

#     student_support_token, student_support_rep, support_alpha, support_scores = support_selector(
#         query_embs=query_embs, support_tokens=support_tokens, support_mask=support_mask, training=False
#     )
#     student_temporal_token, student_temporal_rep, temporal_alpha, temporal_scores = temporal_selector(
#         query_embs=query_embs, support_tokens=temporal_tokens, support_mask=temporal_mask, training=False
#     )

#     class_logits, cls_out, aux = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_token,
#         contrast_tokens=zeros_contrast,
#         temporal_tokens=student_temporal_token,
#         training=False,
#     )

#     loss = weighted_scce_loss(labels, class_logits, class_weights)
#     if ratt_head.losses:
#         loss += tf.add_n(ratt_head.losses)

#     probs = tf.nn.softmax(class_logits, axis=-1)
#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)

#     val_loss_metric.update_state(loss)
#     val_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()
#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print("val support alpha stats:", "mean=", tf.reduce_mean(support_alpha).numpy(), "std=", tf.math.reduce_std(support_alpha).numpy())
#     print("val temporal alpha stats:", "mean=", tf.reduce_mean(temporal_alpha).numpy(), "std=", tf.math.reduce_std(temporal_alpha).numpy())

#     return {"loss": float(loss.numpy()), "acc": float(val_acc_metric.result().numpy()), "logits": class_logits, "probs": probs, "labels": labels}


# def run_train_epoch(train_ds, ratt_head, support_selector, temporal_selector,
#                     train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#                     frame_emb_mm, path_to_idx, class_weight, mix_lambda):
#     train_loss_metric.reset_state()
#     train_acc_metric.reset_state()
#     for step, batch in enumerate(train_ds):
#         out = train_step(batch, ratt_head, support_selector, temporal_selector,
#                          train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#                          frame_emb_mm, path_to_idx, class_weight, mix_lambda)
#         if step % 1 == 0:
#             print(f"[train] step={step} mix_lambda={mix_lambda:.4f} loss={out['loss']:.4f} acc={train_acc_metric.result().numpy():.4f}")
#     return float(train_loss_metric.result().numpy()), float(train_acc_metric.result().numpy())


# def run_val_epoch(val_ds, ratt_head, support_selector, temporal_selector,
#                   val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#                   frame_emb_mm, path_to_idx, class_weight):
#     val_loss_metric.reset_state()
#     val_acc_metric.reset_state()
#     for step, batch in enumerate(val_ds):
#         out = eval_step(batch, ratt_head, support_selector, temporal_selector,
#                         val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#                         frame_emb_mm, path_to_idx, class_weight)
#         if step % 1 == 0:
#             print(f"[val] step={step} loss={out['loss']:.4f} running acc={val_acc_metric.result().numpy():.4f}")
#     return float(val_loss_metric.result().numpy()), float(val_acc_metric.result().numpy())


# if __name__ == "__main__":
#     print("SEARCH_K_CONTENT", config.SEARCH_K_CONTENT)
#     print("SEARCH_K_TEMPORAL", config.SEARCH_K_TEMPORAL)
#     print("K_SIM", config.K_SIM)
#     print("K_CONTRAST", config.K_CONTRAST)
#     print("K_TEMPORAL", config.K_TEMPORAL)
#     print("FUTURE_CHUNK_STEP", config.FUTURE_CHUNK_STEP)
#     print("CHUNK_SIZE", config.CHUNK_SIZE)
#     print("CHROMADB_COLLECTION", config.CHROMADB_COLLECTION)
#     print("STAGE1_WEIGHTS", config.STAGE1_WEIGHTS)
#     print("RATT_WEIGHTS", config.RATT_WEIGHTS)
#     print("ALIGN_LOSS_WEIGHT", ALIGN_LOSS_WEIGHT)
#     print("TEMP_ALIGN_LOSS_WEIGHT", TEMP_ALIGN_LOSS_WEIGHT)
#     print("TEACHER_K", TEACHER_K)
#     print("MIX_LAMBDA_START", MIX_LAMBDA_START)
#     print("MIX_LAMBDA_END", MIX_LAMBDA_END)
#     print("MIX_SCHEDULE", MIX_SCHEDULE)

#     train_vids = config.TRAIN_VIDS
#     train_samples = load_samples(train_vids, stride=1)
#     train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE, chunk_stride=config.CHUNK_STRIDE)

#     test_vids = config.TEST_VIDS
#     test_samples = load_samples(test_vids, stride=1)
#     test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE, chunk_stride=config.CHUNK_STRIDE)

#     random.shuffle(train_samples)
#     random.shuffle(test_samples)

#     print(len(train_chunk_samples))
#     print(len(test_chunk_samples))

#     class_weight = compute_class_weights(train_chunk_samples)

#     print(f"Train chunks: {len(train_chunk_samples)}")
#     print(f"Val chunks:   {len(test_chunk_samples)}")

#     train_dataset = build_tf_dataset_chunks(train_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=True)
#     val_dataset = build_tf_dataset_chunks(test_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=False)

#     print(f"Train dataset: {train_dataset}")
#     print(f"Val dataset:   {val_dataset}")

#     client = PersistentClient(path="./chroma_store")
#     collection = client.get_or_create_collection(name=config.CHROMADB_COLLECTION, metadata={"hnsw:space": "cosine"})

#     chunk_encoder = ChunkEncoder(hidden_size=768, num_layers=4, num_heads=8, max_frames=config.CHUNK_SIZE)
#     dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
#     _ = chunk_encoder(dummy_frame_embs, training=False)

#     print("==== DEBUG STAGE2 ====")
#     print("cwd:", os.getcwd())
#     print("weights path:", config.STAGE1_WEIGHTS)
#     print("abs path:", os.path.abspath(config.STAGE1_WEIGHTS))
#     print("exists:", os.path.exists(config.STAGE1_WEIGHTS))
#     print("======================")

#     chunk_encoder.load_weights(config.STAGE1_WEIGHTS)
#     for i in range(chunk_encoder.num_layers):
#         block = getattr(chunk_encoder, f"transformer_block_{i}")
#         with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
#             weights = pickle.load(f)
#         block.set_weights(weights)

#     print("[STAGE1] Loaded chunk encoder weights")
#     chunk_encoder.trainable = False
#     print("[STAGE1] Chunk encoder frozen")

#     store_name = "train_val_frames_chunk8_stride2"
#     frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

#     train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
#     train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
#     val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
#     val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

#     batch = next(iter(train_dataset))

#     ratt_head = RATTHeadV2(hidden_size=768, num_heads=8, num_layers=config.NUM_LAYERS)
#     support_selector = SupportSelector(hidden_dim=1024, name="support_selector")
#     temporal_selector = SupportSelector(hidden_dim=1024, name="temporal_selector")

#     train_chunk_lookup = {make_chunk_key(c): c for c in train_chunk_samples}
#     train_future_key_lookup = build_future_key_lookup(train_chunk_samples, future_step=5)
#     val_chunk_lookup = {make_chunk_key(c): c for c in test_chunk_samples}
#     val_future_key_lookup = build_future_key_lookup(test_chunk_samples, future_step=5)

#     query_embs, support, contrast, temporal = fetch_live_batch(
#         metadata=batch[1],
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     _ = support_selector(query_embs, support, support_mask=support_mask_from_tokens(support), training=False)
#     _ = temporal_selector(query_embs, temporal, support_mask=support_mask_from_tokens(temporal), training=False)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

#     student_support_token, student_support_rep, support_alpha, _ = support_selector(
#         query_embs, support, support_mask=support_mask_from_tokens(support), training=False
#     )
#     student_temporal_token, student_temporal_rep, temporal_alpha, _ = temporal_selector(
#         query_embs, temporal, support_mask=support_mask_from_tokens(temporal), training=False
#     )

#     logits, _, _ = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_token,
#         contrast_tokens=tf.zeros_like(contrast),
#         temporal_tokens=student_temporal_token,
#         training=False,
#     )
#     print(logits)
#     print("logits:", logits.shape)

#     for epoch in range(config.EPOCHS):
#         print(f"\\n===== EPOCH {epoch+1}/{config.EPOCHS} =====")
#         mix_lambda = get_mix_lambda(epoch_idx=epoch, total_epochs=config.EPOCHS)
#         print(f"[epoch {epoch+1}] mix_lambda={mix_lambda:.4f}")

#         train_loss, train_acc = run_train_epoch(
#             train_ds=train_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             temporal_selector=temporal_selector,
#             train_chunk_lookup=train_chunk_lookup,
#             train_future_key_lookup=train_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#             mix_lambda=mix_lambda,
#         )

#         val_loss, val_acc = run_val_epoch(
#             val_ds=val_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             temporal_selector=temporal_selector,
#             val_chunk_lookup=val_chunk_lookup,
#             val_future_key_lookup=val_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#         )

#         print(f"[epoch {epoch+1}] mix_lambda={mix_lambda:.4f} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

#     ratt_head.save_weights(config.RATT_WEIGHTS)
#     print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")

#     os.makedirs("rag_weights", exist_ok=True)
#     for i in range(config.NUM_LAYERS):
#         block = getattr(ratt_head, f"transformer_block_{i}")
#         with open(f"rag_weights/{config.RUN_ID}_transformer_block_{i}.pkl", "wb") as f:
#             pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"[MAIN] saved transformer block {i} weights")








# from collections import defaultdict
# import os
# import random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from chromadb import PersistentClient
# from dataset import load_samples, build_chunks, build_tf_dataset_chunks, oversample_chunk_samples
# import config_stage2 as config
# import pickle
# from models.chunk_encoder import ChunkEncoder
# from models.ratt_v2 import RATTHeadV2
# import gc

# tf.keras.backend.clear_session()
# gc.collect()

# SEED = 12
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# try:
#     tf.config.experimental.enable_op_determinism()
# except Exception:
#     pass

# ALIGN_LOSS_WEIGHT = 0.3
# TEMP_ALIGN_LOSS_WEIGHT = 0.3
# CONTRAST_ALIGN_LOSS_WEIGHT = 0.3

# TEACHER_K = config.K_SIM
# MIX_LAMBDA_START = 0.5
# MIX_LAMBDA_END = 0.0
# MIX_SCHEDULE = "cosine"


# def get_mix_lambda(epoch_idx, total_epochs):
#     if total_epochs <= 1:
#         return MIX_LAMBDA_END
#     t = epoch_idx / float(total_epochs - 1)
#     t = min(max(t, 0.0), 1.0)
#     if MIX_SCHEDULE == "cosine":
#         alpha = 0.5 * (1.0 + np.cos(np.pi * t))
#     else:
#         alpha = 1.0 - t
#     return float(MIX_LAMBDA_END + (MIX_LAMBDA_START - MIX_LAMBDA_END) * alpha)


# # class BranchSelector(tf.keras.Model):
# #     def __init__(self, hidden_dim=256, name="branch_selector"):
# #         super().__init__(name=name)
# #         self.d1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
# #         self.d2 = tf.keras.layers.Dense(256, activation="relu")
# #         self.d3 = tf.keras.layers.Dense(1)

# #     def call(self, query_embs, branch_tokens, branch_mask=None, training=False):
# #         q = tf.expand_dims(query_embs, axis=1)
# #         q = tf.repeat(q, tf.shape(branch_tokens)[1], axis=1)
# #         x = tf.concat([q, branch_tokens, q - branch_tokens, q * branch_tokens], axis=-1)
# #         h = self.d1(x, training=training)
# #         h = self.d2(h, training=training)
# #         scores = self.d3(h, training=training)[..., 0]
# #         if branch_mask is not None:
# #             scores = tf.where(branch_mask > 0, scores, tf.constant(-1e9, dtype=scores.dtype))
# #         alpha = tf.nn.softmax(scores, axis=-1)
# #         pooled_rep = tf.reduce_sum(branch_tokens * alpha[..., None], axis=1)
# #         pooled_token = tf.expand_dims(pooled_rep, axis=1)
# #         return pooled_token, pooled_rep, alpha, scores


# class BranchSelector(tf.keras.Model):
#     def __init__(self, hidden_dim=256, name="branch_selector"):
#         super().__init__(name=name)
#         self.d1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
#         self.d2 = tf.keras.layers.Dense(256, activation="relu")
#         self.d3 = tf.keras.layers.Dense(1)

#     def call(self, query_embs, branch_tokens, branch_mask=None, training=False):
#         q = tf.expand_dims(query_embs, axis=1)                       # (B, 1, D)
#         q = tf.repeat(q, tf.shape(branch_tokens)[1], axis=1)        # (B, K, D)

#         x = tf.concat([q, branch_tokens, q - branch_tokens, q * branch_tokens], axis=-1)
#         h = self.d1(x, training=training)
#         h = self.d2(h, training=training)
#         scores = self.d3(h, training=training)[..., 0]              # (B, K)

#         if branch_mask is not None:
#             scores = tf.where(branch_mask > 0, scores, tf.constant(-1e9, dtype=scores.dtype))

#         alpha = tf.nn.softmax(scores, axis=-1)                      # (B, K)

#         # keep ALL tokens, just reweight them instead of pooling
#         weighted_tokens = branch_tokens * alpha[..., None]          # (B, K, D)

#         # optional summary only for logging / aux losses
#         summary_rep = tf.reduce_sum(weighted_tokens, axis=1)        # (B, D)

#         return weighted_tokens, summary_rep, alpha, scores

# def mix_teacher_student_tokens(student_tokens, teacher_tokens, mix_lambda):
#     return (1.0 - mix_lambda) * student_tokens + mix_lambda * tf.stop_gradient(teacher_tokens)

# def make_chunk_key(chunk):
#     return (
#         int(chunk["vid"]),
#         str(chunk["side"]),
#         int(chunk["clip"]),
#         int(chunk["start_idx"]),
#         int(chunk["end_idx"]),
#     )


# def build_future_key_lookup(all_chunks, future_step=5):
#     grouped = defaultdict(list)
#     for chunk in all_chunks:
#         grouped[(int(chunk["vid"]), int(chunk["clip"]))].append(chunk)

#     future_key_lookup = {}
#     for _, group in grouped.items():
#         group_sorted = sorted(group, key=lambda c: int(c["start_idx"]))
#         last_idx = len(group_sorted) - 1
#         for idx, chunk in enumerate(group_sorted):
#             cur_key = make_chunk_key(chunk)
#             fut_idx = min(idx + future_step, last_idx)
#             fut_key = make_chunk_key(group_sorted[fut_idx])
#             future_key_lookup[cur_key] = fut_key
#     return future_key_lookup


# CACHE_DIR = "./frame_cache_vit"


# def get_store_paths(store_name):
#     return {
#         "emb": os.path.join(CACHE_DIR, f"{store_name}_emb.dat"),
#         "paths": os.path.join(CACHE_DIR, f"{store_name}_paths.npy"),
#         "meta": os.path.join(CACHE_DIR, f"{store_name}_meta.npz"),
#     }


# def load_frame_store(store_name="train_val_frames"):
#     paths = get_store_paths(store_name)
#     meta = np.load(paths["meta"])

#     n_frames = int(meta["n_frames"])
#     emb_dim = int(meta["emb_dim"])

#     emb_mm = np.memmap(paths["emb"], dtype="float32", mode="r", shape=(n_frames, emb_dim))
#     frame_paths = np.load(paths["paths"], allow_pickle=True).tolist()
#     path_to_idx = {p: i for i, p in enumerate(frame_paths)}

#     print(f"[frame cache] loaded store '{store_name}'")
#     print(f"[frame cache] shape = ({n_frames}, {emb_dim})")
#     return emb_mm, frame_paths, path_to_idx


# def query_collection(query_emb, collection, n_results):
#     results = collection.query(
#         query_embeddings=[query_emb.tolist()],
#         n_results=n_results,
#         include=["embeddings", "metadatas"]
#     )

#     raw_embs = results["embeddings"][0]
#     raw_meta = results["metadatas"][0]

#     out = []
#     for emb, meta in zip(raw_embs, raw_meta):
#         out.append({
#             "emb": np.asarray(emb, dtype=np.float32),
#             "meta": {
#                 "label": int(meta["label"]),
#                 "status": str(meta["status"]),
#                 "status_id": int(meta["status_id"]),
#                 "side": str(meta["side"]),
#                 "vid": int(meta["vid_num"]),
#                 "clip": int(meta["clip_num"]),
#                 "t_center": float(meta["t_center"]),
#                 "t_width": float(meta["t_width"]),
#                 "start_idx": int(meta["start_idx"]),
#                 "end_idx": int(meta["end_idx"]),
#                 "class_logit": float(meta["class_logit"])
#             },
#         })
#     return out


# def extract_meta(chunk):
#     return {
#         "label": int(chunk["label"]),
#         "status": str(chunk["status"]),
#         "status_id": int(chunk["status_id"]),
#         "side": str(chunk["side"]),
#         "vid": int(chunk["vid"]),
#         "clip": int(chunk["clip"]),
#         "t_center": float(chunk["t_center"]),
#         "t_width": float(chunk["t_width"]),
#         "start_idx": int(chunk["start_idx"]),
#         "end_idx": int(chunk["end_idx"]),
#     }


# def same_chunk_meta(meta_a, meta_b):
#     return (
#         int(meta_a["vid"]) == int(meta_b["vid"])
#         and str(meta_a["side"]) == str(meta_b["side"])
#         and int(meta_a["clip"]) == int(meta_b["clip"])
#         and int(meta_a["start_idx"]) == int(meta_b["start_idx"])
#         and int(meta_a["end_idx"]) == int(meta_b["end_idx"])
#     )


# def dedup_signature(meta):
#     return (
#         int(meta["vid"]),
#         str(meta["side"]),
#         int(meta["clip"]),
#         int(meta["start_idx"]),
#         int(meta["end_idx"]),
#     )


# def pad_or_trim(items, k, emb_dim, pad_meta_template):
#     items = list(items)
#     if len(items) >= k:
#         items = items[:k]
#     else:
#         zero_emb = np.zeros((emb_dim,), dtype=np.float32)
#         for _ in range(k - len(items)):
#             items.append({"emb": zero_emb.copy(), "meta": dict(pad_meta_template)})

#     embs = np.stack([x["emb"] for x in items], axis=0)
#     metas = [x["meta"] for x in items]
#     return embs, metas


# def encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx):
#     idxs = [path_to_idx[p] for p in chunk["frames"]]
#     frame_embs = frame_emb_mm[idxs].astype(np.float32)
#     frame_embs = tf.convert_to_tensor(frame_embs[None, :, :], dtype=tf.float32)
#     stage1_chunk_emb, _ = chunk_encoder(frame_embs, training=False)
#     return stage1_chunk_emb[0].numpy().astype(np.float32)


# def build_teacher_support_items(query_meta, content_candidates, k_teacher):
#     teacher_items, seen = [], set()
#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) != int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue
#         teacher_items.append(cand)
#         seen.add(sig)
#         if len(teacher_items) >= k_teacher:
#             break
#     return teacher_items


# def build_teacher_temporal_items(query_meta, temporal_candidates, k_teacher):
#     teacher_items, seen = [], set()
#     for cand in temporal_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) != int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue
#         teacher_items.append(cand)
#         seen.add(sig)
#         if len(teacher_items) >= k_teacher:
#             break
#     return teacher_items


# def build_teacher_contrast_items(query_meta, content_candidates, k_teacher):
#     teacher_items, seen = [], set()
#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if int(cand_meta["status_id"]) == int(query_meta["status_id"]):
#             continue
#         if sig in seen:
#             continue
#         teacher_items.append(cand)
#         seen.add(sig)
#         if len(teacher_items) >= k_teacher:
#             break
#     return teacher_items


# def build_live_entry(
#     chunk, future_chunk, collection, chunk_encoder, frame_emb_mm, path_to_idx,
#     search_k_content, search_k_temporal, k_sim, k_contrast, k_temporal, k_teacher=TEACHER_K
# ):
#     query_emb = encode_chunk(chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     future_emb = encode_chunk(future_chunk, chunk_encoder, frame_emb_mm, path_to_idx)
#     query_meta = extract_meta(chunk)
#     emb_dim = query_emb.shape[0]

#     pad_meta_template = {
#         "label": -1, "status": "PAD", "status_id": -1, "side": "PAD", "vid": -1,
#         "clip": -1, "t_center": -1.0, "t_width": -1.0, "start_idx": -1, "end_idx": -1,
#     }

#     content_candidates = query_collection(query_emb, collection, search_k_content)
#     used_content, filtered_candidates = set(), []

#     for cand in content_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in used_content:
#             continue
#         filtered_candidates.append(cand)
#         used_content.add(sig)

#     sim_items = filtered_candidates[:k_sim]
#     remaining = filtered_candidates[k_sim:]

#     if len(filtered_candidates) >= 150:
#         contrast_pool = filtered_candidates[50:150]
#     else:
#         half = len(remaining) // 2
#         contrast_pool = remaining[half:]

#     if len(contrast_pool) < k_contrast:
#         contrast_pool = remaining

#     contrast_items = random.sample(contrast_pool, k_contrast) if len(contrast_pool) > k_contrast else contrast_pool[:k_contrast]

#     sim_embs, _ = pad_or_trim(sim_items, k_sim, emb_dim, pad_meta_template)
#     contrast_embs, _ = pad_or_trim(contrast_items, k_contrast, emb_dim, pad_meta_template)

#     teacher_support_items = build_teacher_support_items(query_meta, content_candidates, k_teacher)
#     teacher_support_embs, _ = pad_or_trim(teacher_support_items, k_teacher, emb_dim, pad_meta_template)

#     teacher_contrast_items = build_teacher_contrast_items(query_meta, content_candidates, k_teacher)
#     teacher_contrast_embs, _ = pad_or_trim(teacher_contrast_items, k_teacher, emb_dim, pad_meta_template)

#     temporal_candidates = query_collection(future_emb, collection, search_k_temporal)
#     temporal_items, seen_temporal = [], set()

#     for cand in temporal_candidates:
#         cand_meta = cand["meta"]
#         sig = dedup_signature(cand_meta)
#         if same_chunk_meta(query_meta, cand_meta):
#             continue
#         if cand_meta["side"] != query_meta["side"]:
#             continue
#         if sig in seen_temporal:
#             continue
#         temporal_items.append(cand)
#         seen_temporal.add(sig)
#         if len(temporal_items) >= k_temporal:
#             break

#     temporal_embs, _ = pad_or_trim(temporal_items, k_temporal, emb_dim, pad_meta_template)
#     teacher_temporal_items = build_teacher_temporal_items(query_meta, temporal_candidates, k_teacher)
#     teacher_temporal_embs, _ = pad_or_trim(teacher_temporal_items, k_teacher, emb_dim, pad_meta_template)

#     return {
#         "query_emb": query_emb,
#         "sim_embs": sim_embs,
#         "contrast_embs": contrast_embs,
#         "temporal_embs": temporal_embs,
#         "teacher_support_embs": teacher_support_embs,
#         "teacher_temporal_embs": teacher_temporal_embs,
#         "teacher_contrast_embs": teacher_contrast_embs,
#     }


# def _to_py_scalar(x):
#     if hasattr(x, "numpy"):
#         x = x.numpy()
#     if isinstance(x, np.ndarray):
#         if x.ndim == 0:
#             x = x.item()
#         else:
#             raise ValueError(f"Expected scalar, got array shape {x.shape}")
#     if isinstance(x, bytes):
#         x = x.decode("utf-8")
#     return x


# def make_chunk_key_from_meta(metadata, i):
#     return (
#         int(_to_py_scalar(metadata["vid"][i])),
#         str(_to_py_scalar(metadata["side"][i])),
#         int(_to_py_scalar(metadata["clip"][i])),
#         int(_to_py_scalar(metadata["start_idx"][i])),
#         int(_to_py_scalar(metadata["end_idx"][i])),
#     )


# def fetch_live_batch(metadata, chunk_lookup, future_key_lookup, collection, chunk_encoder, frame_emb_mm, path_to_idx, return_teacher=False):
#     batch_size = metadata["vid"].shape[0]
#     query_embs, support_tokens, contrast_tokens, temporal_tokens = [], [], [], []
#     teacher_support_tokens, teacher_temporal_tokens, teacher_contrast_tokens = [], [], []

#     for i in range(batch_size):
#         key = make_chunk_key_from_meta(metadata, i)
#         chunk = chunk_lookup[key]
#         future_chunk = chunk_lookup[future_key_lookup[key]]

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
#             k_teacher=TEACHER_K,
#         )

#         query_embs.append(entry["query_emb"])
#         support_tokens.append(entry["sim_embs"])
#         contrast_tokens.append(entry["contrast_embs"])
#         temporal_tokens.append(entry["temporal_embs"])

#         if return_teacher:
#             teacher_support_tokens.append(entry["teacher_support_embs"])
#             teacher_temporal_tokens.append(entry["teacher_temporal_embs"])
#             teacher_contrast_tokens.append(entry["teacher_contrast_embs"])

#     query_embs = tf.convert_to_tensor(np.stack(query_embs, axis=0), dtype=tf.float32)
#     support_tokens = tf.convert_to_tensor(np.stack(support_tokens, axis=0), dtype=tf.float32)
#     contrast_tokens = tf.convert_to_tensor(np.stack(contrast_tokens, axis=0), dtype=tf.float32)
#     temporal_tokens = tf.convert_to_tensor(np.stack(temporal_tokens, axis=0), dtype=tf.float32)

#     if not return_teacher:
#         return query_embs, support_tokens, contrast_tokens, temporal_tokens

#     teacher_support_tokens = tf.convert_to_tensor(np.stack(teacher_support_tokens, axis=0), dtype=tf.float32)
#     teacher_temporal_tokens = tf.convert_to_tensor(np.stack(teacher_temporal_tokens, axis=0), dtype=tf.float32)
#     teacher_contrast_tokens = tf.convert_to_tensor(np.stack(teacher_contrast_tokens, axis=0), dtype=tf.float32)

#     return (
#         query_embs,
#         support_tokens,
#         contrast_tokens,
#         temporal_tokens,
#         teacher_support_tokens,
#         teacher_temporal_tokens,
#         teacher_contrast_tokens,
#     )


# def token_mask(tokens):
#     token_norm = tf.reduce_sum(tf.abs(tokens), axis=-1)
#     return tf.cast(token_norm > 0.0, tf.float32)


# scce_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


# def weighted_scce_loss(labels, logits, class_weights):
#     per_example_loss = scce_no_reduce(labels, logits)
#     weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
#     return tf.reduce_mean(per_example_loss * weights)


# def compute_class_weights(chunk_samples, power=0.5):
#     labels = np.array([int(c["status_id"]) for c in chunk_samples], dtype=np.int32)
#     counts = np.bincount(labels, minlength=3).astype(np.float32)
#     if np.any(counts == 0):
#         raise ValueError(f"Missing class in training data. Counts: {counts}")
#     max_count = counts.max()
#     weights = (max_count / counts) ** power
#     weights = weights / weights[0]
#     print("train class counts:", counts.astype(int).tolist())
#     print("class weights:", weights.tolist())
#     return tf.constant(weights, dtype=tf.float32)


# def grad_rms(g):
#     if g is None:
#         return 0.0
#     g = tf.cast(g, tf.float32)
#     return float(tf.sqrt(tf.reduce_mean(tf.square(g))).numpy())


# def mix_teacher_student_rep(student_rep, teacher_rep, mix_lambda):
#     mixed_rep = (1.0 - mix_lambda) * student_rep + mix_lambda * tf.stop_gradient(teacher_rep)
#     mixed_token = tf.expand_dims(mixed_rep, axis=1)
#     return mixed_token, mixed_rep


# def train_step(
#     batch, ratt_head, support_selector, temporal_selector, contrast_selector,
#     train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#     frame_emb_mm, path_to_idx, class_weights, mix_lambda
# ):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     (
#         query_embs, support_tokens, contrast_tokens, temporal_tokens,
#         teacher_support_tokens, teacher_temporal_tokens, teacher_contrast_tokens
#     ) = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=True,
#     )

#     support_mask = token_mask(support_tokens)
#     temporal_mask = token_mask(temporal_tokens)
#     contrast_mask = token_mask(contrast_tokens)

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(query_embs)
#         tape.watch(support_tokens)
#         tape.watch(temporal_tokens)
#         tape.watch(contrast_tokens)

#         # student_support_token, student_support_rep, support_alpha, _ = support_selector(
#         #     query_embs=query_embs, branch_tokens=support_tokens, branch_mask=support_mask, training=True
#         # )
#         # student_temporal_token, student_temporal_rep, temporal_alpha, _ = temporal_selector(
#         #     query_embs=query_embs, branch_tokens=temporal_tokens, branch_mask=temporal_mask, training=True
#         # )
#         # student_contrast_token, student_contrast_rep, contrast_alpha, _ = contrast_selector(
#         #     query_embs=query_embs, branch_tokens=contrast_tokens, branch_mask=contrast_mask, training=True
#         # )

#         # teacher_support_rep = tf.reduce_mean(teacher_support_tokens, axis=1)
#         # teacher_temporal_rep = tf.reduce_mean(teacher_temporal_tokens, axis=1)
#         # teacher_contrast_rep = tf.reduce_mean(teacher_contrast_tokens, axis=1)

#         # mixed_support_token, _ = mix_teacher_student_rep(student_support_rep, teacher_support_rep, mix_lambda)
#         # mixed_temporal_token, _ = mix_teacher_student_rep(student_temporal_rep, teacher_temporal_rep, mix_lambda)
#         # mixed_contrast_token, _ = mix_teacher_student_rep(student_contrast_rep, teacher_contrast_rep, mix_lambda)

#         student_support_tokens, student_support_rep, support_alpha, _ = support_selector(
#             query_embs=query_embs, branch_tokens=support_tokens, branch_mask=support_mask, training=True
#         )
#         student_temporal_tokens, student_temporal_rep, temporal_alpha, _ = temporal_selector(
#             query_embs=query_embs, branch_tokens=temporal_tokens, branch_mask=temporal_mask, training=True
#         )
#         student_contrast_tokens, student_contrast_rep, contrast_alpha, _ = contrast_selector(
#             query_embs=query_embs, branch_tokens=contrast_tokens, branch_mask=contrast_mask, training=True
#         )

#         teacher_support_rep = tf.reduce_mean(teacher_support_tokens, axis=1)
#         teacher_temporal_rep = tf.reduce_mean(teacher_temporal_tokens, axis=1)
#         teacher_contrast_rep = tf.reduce_mean(teacher_contrast_tokens, axis=1)

#         mixed_support_token = mix_teacher_student_tokens(student_support_tokens, teacher_support_tokens, mix_lambda)
#         mixed_temporal_token = mix_teacher_student_tokens(student_temporal_tokens, teacher_temporal_tokens, mix_lambda)
#         mixed_contrast_token = mix_teacher_student_tokens(student_contrast_tokens, teacher_contrast_tokens, mix_lambda)

#         class_logits, cls_out, aux = ratt_head(
#             chunk_embs=query_embs,
#             support_tokens=mixed_support_token,
#             contrast_tokens=mixed_contrast_token,
#             temporal_tokens=mixed_temporal_token,
#             training=True,
#         )

#         cls_loss = weighted_scce_loss(labels, class_logits, class_weights)
#         support_align_loss = tf.reduce_mean(tf.reduce_sum(tf.square(student_support_rep - tf.stop_gradient(teacher_support_rep)), axis=-1))
#         temporal_align_loss = tf.reduce_mean(tf.reduce_sum(tf.square(student_temporal_rep - tf.stop_gradient(teacher_temporal_rep)), axis=-1))
#         contrast_align_loss = tf.reduce_mean(tf.reduce_sum(tf.square(student_contrast_rep - tf.stop_gradient(teacher_contrast_rep)), axis=-1))

#         loss = (
#             cls_loss
#             + ALIGN_LOSS_WEIGHT * support_align_loss
#             + TEMP_ALIGN_LOSS_WEIGHT * temporal_align_loss
#             + CONTRAST_ALIGN_LOSS_WEIGHT * contrast_align_loss
#         )

#         if ratt_head.losses:
#             loss += tf.add_n(ratt_head.losses)

#     train_vars = (
#         ratt_head.trainable_variables
#         + support_selector.trainable_variables
#         + temporal_selector.trainable_variables
#         + contrast_selector.trainable_variables
#     )
#     grads = tape.gradient(loss, train_vars)
#     optimizer.apply_gradients(zip(grads, train_vars))

#     g_query = tape.gradient(loss, query_embs)
#     g_support = tape.gradient(loss, support_tokens)
#     g_temporal = tape.gradient(loss, temporal_tokens)
#     g_contrast = tape.gradient(loss, contrast_tokens)

#     print(
#         f"branch_grad_rms | "
#         f"query={grad_rms(g_query):.6f} "
#         f"support={grad_rms(g_support):.6f} "
#         f"contrast={grad_rms(g_contrast):.6f} "
#         f"temporal={grad_rms(g_temporal):.6f}"
#     )
#     print({
#         "mix_lambda": float(mix_lambda),
#         "cls_loss": float(cls_loss.numpy()),
#         "support_align_loss": float(support_align_loss.numpy()),
#         "contrast_align_loss": float(contrast_align_loss.numpy()),
#         "temporal_align_loss": float(temporal_align_loss.numpy()),
#         "loss": float(loss.numpy()),
#     })

#     del tape

#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)
#     probs = tf.nn.softmax(class_logits, axis=-1)

#     train_loss_metric.update_state(loss)
#     train_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()
#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print("support alpha stats:", "mean=", tf.reduce_mean(support_alpha).numpy(), "std=", tf.math.reduce_std(support_alpha).numpy())
#     print("contrast alpha stats:", "mean=", tf.reduce_mean(contrast_alpha).numpy(), "std=", tf.math.reduce_std(contrast_alpha).numpy())
#     print("temporal alpha stats:", "mean=", tf.reduce_mean(temporal_alpha).numpy(), "std=", tf.math.reduce_std(temporal_alpha).numpy())

#     return {"loss": float(loss.numpy()), "acc": float(train_acc_metric.result().numpy()), "logits": class_logits, "probs": probs}


# def eval_step(
#     batch, ratt_head, support_selector, temporal_selector, contrast_selector,
#     val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#     frame_emb_mm, path_to_idx, class_weights
# ):
#     metadata = batch[1]
#     labels = tf.cast(metadata["status_id"], tf.int32)
#     labels = tf.reshape(labels, (-1,))

#     query_embs, support_tokens, contrast_tokens, temporal_tokens = fetch_live_batch(
#         metadata=metadata,
#         chunk_lookup=val_chunk_lookup,
#         future_key_lookup=val_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     support_mask = token_mask(support_tokens)
#     temporal_mask = token_mask(temporal_tokens)
#     contrast_mask = token_mask(contrast_tokens)

#     # student_support_token, _, support_alpha, _ = support_selector(
#     #     query_embs=query_embs, branch_tokens=support_tokens, branch_mask=support_mask, training=False
#     # )
#     # student_temporal_token, _, temporal_alpha, _ = temporal_selector(
#     #     query_embs=query_embs, branch_tokens=temporal_tokens, branch_mask=temporal_mask, training=False
#     # )
#     # student_contrast_token, _, contrast_alpha, _ = contrast_selector(
#     #     query_embs=query_embs, branch_tokens=contrast_tokens, branch_mask=contrast_mask, training=False
#     # )

#     student_support_tokens, _, support_alpha, _ = support_selector(
#         query_embs=query_embs, branch_tokens=support_tokens, branch_mask=support_mask, training=False
#     )
#     student_temporal_tokens, _, temporal_alpha, _ = temporal_selector(
#         query_embs=query_embs, branch_tokens=temporal_tokens, branch_mask=temporal_mask, training=False
#     )
#     student_contrast_tokens, _, contrast_alpha, _ = contrast_selector(
#         query_embs=query_embs, branch_tokens=contrast_tokens, branch_mask=contrast_mask, training=False
#     )

#     class_logits, cls_out, aux = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_tokens,
#         contrast_tokens=student_contrast_tokens,
#         temporal_tokens=student_temporal_tokens,
#         training=False,
#     )

#     loss = weighted_scce_loss(labels, class_logits, class_weights)
#     if ratt_head.losses:
#         loss += tf.add_n(ratt_head.losses)

#     probs = tf.nn.softmax(class_logits, axis=-1)
#     preds = tf.argmax(class_logits, axis=-1, output_type=tf.int32)

#     val_loss_metric.update_state(loss)
#     val_acc_metric.update_state(labels, class_logits)

#     temp = pd.DataFrame()
#     batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
#     temp["labels"] = labels.numpy()
#     temp["preds"] = preds.numpy()
#     print(temp)
#     print(class_logits)
#     print(f"batch acc={batch_acc:.6f}")

#     print("val support alpha stats:", "mean=", tf.reduce_mean(support_alpha).numpy(), "std=", tf.math.reduce_std(support_alpha).numpy())
#     print("val contrast alpha stats:", "mean=", tf.reduce_mean(contrast_alpha).numpy(), "std=", tf.math.reduce_std(contrast_alpha).numpy())
#     print("val temporal alpha stats:", "mean=", tf.reduce_mean(temporal_alpha).numpy(), "std=", tf.math.reduce_std(temporal_alpha).numpy())

#     return {"loss": float(loss.numpy()), "acc": float(val_acc_metric.result().numpy()), "logits": class_logits, "probs": probs, "labels": labels}


# def run_train_epoch(
#     train_ds, ratt_head, support_selector, temporal_selector, contrast_selector,
#     train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#     frame_emb_mm, path_to_idx, class_weight, mix_lambda
# ):
#     train_loss_metric.reset_state()
#     train_acc_metric.reset_state()
#     for step, batch in enumerate(train_ds):
#         out = train_step(
#             batch, ratt_head, support_selector, temporal_selector, contrast_selector,
#             train_chunk_lookup, train_future_key_lookup, collection, chunk_encoder,
#             frame_emb_mm, path_to_idx, class_weight, mix_lambda
#         )
#         if step % 1 == 0:
#             print(f"[train] step={step} mix_lambda={mix_lambda:.4f} loss={out['loss']:.4f} acc={train_acc_metric.result().numpy():.4f}")
#     return float(train_loss_metric.result().numpy()), float(train_acc_metric.result().numpy())


# def run_val_epoch(
#     val_ds, ratt_head, support_selector, temporal_selector, contrast_selector,
#     val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#     frame_emb_mm, path_to_idx, class_weight
# ):
#     val_loss_metric.reset_state()
#     val_acc_metric.reset_state()
#     for step, batch in enumerate(val_ds):
#         out = eval_step(
#             batch, ratt_head, support_selector, temporal_selector, contrast_selector,
#             val_chunk_lookup, val_future_key_lookup, collection, chunk_encoder,
#             frame_emb_mm, path_to_idx, class_weight
#         )
#         if step % 1 == 0:
#             print(f"[val] step={step} loss={out['loss']:.4f} running acc={val_acc_metric.result().numpy():.4f}")
#     return float(val_loss_metric.result().numpy()), float(val_acc_metric.result().numpy())


# if __name__ == "__main__":
#     print("SEARCH_K_CONTENT", config.SEARCH_K_CONTENT)
#     print("SEARCH_K_TEMPORAL", config.SEARCH_K_TEMPORAL)
#     print("K_SIM", config.K_SIM)
#     print("K_CONTRAST", config.K_CONTRAST)
#     print("K_TEMPORAL", config.K_TEMPORAL)
#     print("FUTURE_CHUNK_STEP", config.FUTURE_CHUNK_STEP)
#     print("CHUNK_SIZE", config.CHUNK_SIZE)
#     print("CHROMADB_COLLECTION", config.CHROMADB_COLLECTION)
#     print("STAGE1_WEIGHTS", config.STAGE1_WEIGHTS)
#     print("RATT_WEIGHTS", config.RATT_WEIGHTS)
#     print("ALIGN_LOSS_WEIGHT", ALIGN_LOSS_WEIGHT)
#     print("TEMP_ALIGN_LOSS_WEIGHT", TEMP_ALIGN_LOSS_WEIGHT)
#     print("CONTRAST_ALIGN_LOSS_WEIGHT", CONTRAST_ALIGN_LOSS_WEIGHT)
#     print("TEACHER_K", TEACHER_K)
#     print("MIX_LAMBDA_START", MIX_LAMBDA_START)
#     print("MIX_LAMBDA_END", MIX_LAMBDA_END)
#     print("MIX_SCHEDULE", MIX_SCHEDULE)

#     train_vids = config.TRAIN_VIDS
#     train_samples = load_samples(train_vids, stride=1)
#     train_chunk_samples = build_chunks(train_samples, chunk_size=config.CHUNK_SIZE, chunk_stride=config.CHUNK_STRIDE)

#     test_vids = config.TEST_VIDS
#     test_samples = load_samples(test_vids, stride=1)
#     test_chunk_samples = build_chunks(test_samples, chunk_size=config.CHUNK_SIZE, chunk_stride=config.CHUNK_STRIDE)

#     random.shuffle(train_samples)
#     random.shuffle(test_samples)

#     print(len(train_chunk_samples))
#     print(len(test_chunk_samples))

#     class_weight = compute_class_weights(train_chunk_samples)

#     print(f"Train chunks: {len(train_chunk_samples)}")
#     print(f"Val chunks:   {len(test_chunk_samples)}")

#     # train_chunk_samples_balanced = oversample_chunk_samples(train_chunk_samples, target="max")
#     train_chunk_samples_balanced = oversample_chunk_samples(train_chunk_samples, target=0.3)

#     # train_dataset = build_tf_dataset_chunks(train_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=True)
#     train_dataset = build_tf_dataset_chunks(train_chunk_samples_balanced, batch_size=config.CHUNK_BATCH_SIZE, training=True)
#     val_dataset = build_tf_dataset_chunks(test_chunk_samples, batch_size=config.CHUNK_BATCH_SIZE, training=False)

#     print(f"Train dataset: {train_dataset}")
#     print(f"Val dataset:   {val_dataset}")

#     client = PersistentClient(path="./chroma_store")
#     collection = client.get_or_create_collection(name=config.CHROMADB_COLLECTION, metadata={"hnsw:space": "cosine"})

#     chunk_encoder = ChunkEncoder(hidden_size=768, num_layers=1, num_heads=4, max_frames=config.CHUNK_SIZE)
#     dummy_frame_embs = tf.zeros((1, config.CHUNK_SIZE, 768), dtype=tf.float32)
#     _ = chunk_encoder(dummy_frame_embs, training=False)

#     print("==== DEBUG STAGE2 ====")
#     print("cwd:", os.getcwd())
#     print("weights path:", config.STAGE1_WEIGHTS)
#     print("abs path:", os.path.abspath(config.STAGE1_WEIGHTS))
#     print("exists:", os.path.exists(config.STAGE1_WEIGHTS))
#     print("======================")

#     chunk_encoder.load_weights(config.STAGE1_WEIGHTS)
#     for i in range(chunk_encoder.num_layers):
#         block = getattr(chunk_encoder, f"transformer_block_{i}")
#         with open(f"stage1_block_weights/chunk_encoder_block_{i}.pkl", "rb") as f:
#             weights = pickle.load(f)
#         block.set_weights(weights)

#     print("[STAGE1] Loaded chunk encoder weights")
#     chunk_encoder.trainable = False
#     print("[STAGE1] Chunk encoder frozen")

#     store_name = "train_val_frames_chunk8_stride2"
#     frame_emb_mm, frame_paths, path_to_idx = load_frame_store(store_name)

#     train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
#     train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
#     val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
#     val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

#     batch = next(iter(train_dataset))

#     ratt_head = RATTHeadV2(hidden_size=768, num_heads=8, num_layers=config.NUM_LAYERS)
#     support_selector = BranchSelector(hidden_dim=1024, name="support_selector")
#     temporal_selector = BranchSelector(hidden_dim=1024, name="temporal_selector")
#     contrast_selector = BranchSelector(hidden_dim=1024, name="contrast_selector")

#     train_chunk_lookup = {make_chunk_key(c): c for c in train_chunk_samples}
#     train_future_key_lookup = build_future_key_lookup(train_chunk_samples, future_step=5)
#     val_chunk_lookup = {make_chunk_key(c): c for c in test_chunk_samples}
#     val_future_key_lookup = build_future_key_lookup(test_chunk_samples, future_step=5)

#     query_embs, support, contrast, temporal = fetch_live_batch(
#         metadata=batch[1],
#         chunk_lookup=train_chunk_lookup,
#         future_key_lookup=train_future_key_lookup,
#         collection=collection,
#         chunk_encoder=chunk_encoder,
#         frame_emb_mm=frame_emb_mm,
#         path_to_idx=path_to_idx,
#         return_teacher=False,
#     )

#     _ = support_selector(query_embs, support, branch_mask=token_mask(support), training=False)
#     _ = temporal_selector(query_embs, temporal, branch_mask=token_mask(temporal), training=False)
#     _ = contrast_selector(query_embs, contrast, branch_mask=token_mask(contrast), training=False)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

#     # student_support_token, _, _, _ = support_selector(query_embs, support, branch_mask=token_mask(support), training=False)
#     # student_temporal_token, _, _, _ = temporal_selector(query_embs, temporal, branch_mask=token_mask(temporal), training=False)
#     # student_contrast_token, _, _, _ = contrast_selector(query_embs, contrast, branch_mask=token_mask(contrast), training=False)

#     student_support_token, _, _, _ = support_selector(query_embs, support, branch_mask=token_mask(support), training=False)
#     student_temporal_token, _, _, _ = temporal_selector(query_embs, temporal, branch_mask=token_mask(temporal), training=False)
#     student_contrast_token, _, _, _ = contrast_selector(query_embs, contrast, branch_mask=token_mask(contrast), training=False)
    
#     logits, _, _ = ratt_head(
#         chunk_embs=query_embs,
#         support_tokens=student_support_token,
#         contrast_tokens=student_contrast_token,
#         temporal_tokens=student_temporal_token,
#         training=False,
#     )
#     print(logits)
#     print("logits:", logits.shape)

#     for epoch in range(config.EPOCHS):
#         print(f"\\n===== EPOCH {epoch+1}/{config.EPOCHS} =====")
#         mix_lambda = get_mix_lambda(epoch_idx=epoch, total_epochs=config.EPOCHS)
#         print(f"[epoch {epoch+1}] mix_lambda={mix_lambda:.4f}")

#         train_loss, train_acc = run_train_epoch(
#             train_ds=train_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             temporal_selector=temporal_selector,
#             contrast_selector=contrast_selector,
#             train_chunk_lookup=train_chunk_lookup,
#             train_future_key_lookup=train_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#             mix_lambda=mix_lambda,
#         )

#         val_loss, val_acc = run_val_epoch(
#             val_ds=val_dataset,
#             ratt_head=ratt_head,
#             support_selector=support_selector,
#             temporal_selector=temporal_selector,
#             contrast_selector=contrast_selector,
#             val_chunk_lookup=val_chunk_lookup,
#             val_future_key_lookup=val_future_key_lookup,
#             collection=collection,
#             chunk_encoder=chunk_encoder,
#             frame_emb_mm=frame_emb_mm,
#             path_to_idx=path_to_idx,
#             class_weight=class_weight,
#         )

#         print(f"[epoch {epoch+1}] mix_lambda={mix_lambda:.4f} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

#     ratt_head.save_weights(config.RATT_WEIGHTS)
#     print(f"[MAIN] saved weights to {config.RATT_WEIGHTS}")

#     os.makedirs("rag_weights", exist_ok=True)
#     for i in range(config.NUM_LAYERS):
#         block = getattr(ratt_head, f"transformer_block_{i}")
#         with open(f"rag_weights/{config.RUN_ID}_transformer_block_{i}.pkl", "wb") as f:
#             pickle.dump(block.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"[MAIN] saved transformer block {i} weights")








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
                        "status": str(meta["status"]),
                        "status_id": int(meta["status_id"]),
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
        "status": "PAD",
        "status_id": -1,
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
            cand_meta["status_id"] == query_meta["status_id"]
            and sig not in used_content
            and len(sim_items) < k_sim
        ):
            sim_items.append(cand)
            used_content.add(sig)
            continue

        if (
            cand_meta["status_id"] != query_meta["status_id"]
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
        # key = (7, 'left', 6, 24, 35) 
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
    class_weights
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
    class_weight
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
            class_weights=class_weight
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

    store_name = 'train_val_frames_chunk12_stride4'
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
            class_weight=class_weight
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

