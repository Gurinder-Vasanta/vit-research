import os
import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tf_keras
from collections import defaultdict

from chromadb import PersistentClient
from models.ratt_head import RATTHead
from models.projection_head import ProjectionHead
import config_chunks_cached as config

from transformers import ViTModel, ViTImageProcessor
import torch
import time

# -----------------------
# DEVICE + VIT
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_processor.do_rescale = False
hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
hf_vit.eval()

# -----------------------
# HELPERS
# -----------------------
def hf_vit_embed_batch(frames_np):
    frames_np = frames_np.astype(np.float32)
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = hf_processor(images=frames_list, return_tensors="pt").to(device)
        out = hf_vit(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)

def z_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return x
    return (x - x.mean()) / (x.std() + 1e-6)

def coarse_time_bin(t_center):
    return int(t_center // config.DELTA_T_NORM)

# -----------------------
# DATASET BUILDING
# -----------------------
def load_and_preprocess_image(path, img_size=(432, 768)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)
    return img

def parse_chunk(chunk, img_size=(432,768)):
    frames = tf.map_fn(
        lambda p: load_and_preprocess_image(p, img_size),
        chunk["frames"],
        fn_output_signature=tf.float32
    )
    return frames, chunk

def build_tf_dataset_chunks(chunk_samples, batch_size, num_workers=8):
    def gen():
        for s in chunk_samples:
            yield s

    output_signature = {
        "frames": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "side": tf.TensorSpec(shape=(), dtype=tf.string),
        "vid": tf.TensorSpec(shape=(), dtype=tf.int32),
        "clip": tf.TensorSpec(shape=(), dtype=tf.int32),
        "t_center": tf.TensorSpec(shape=(), dtype=tf.float32),
        "t_width": tf.TensorSpec(shape=(), dtype=tf.float32),
    }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.map(parse_chunk, num_parallel_calls=num_workers)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------
# RETRIEVAL CACHE
# -----------------------
def embed_rep_chunk(rep):
    frames = []
    for fp in rep["frames"]:
        img = tf.keras.utils.load_img(fp, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(img)
        frames.append(img)
    frames = np.stack(frames, axis=0)
    embs = hf_vit_embed_batch(frames)
    mean = embs.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)

def build_retrieval_cache(collection, chunks, C):
    cache = {}
    bins = {}

    for c in chunks:
        key = (c["side"], coarse_time_bin(c["t_center"]))
        if key not in bins:
            bins[key] = c

    for (side, bin_id), rep in bins.items():
        start = time.perf_counter()
        anchor = embed_rep_chunk(rep)
        res = collection.query(
            query_embeddings=[anchor.tolist()],
            n_results=C,
            where={"side": side},
            include=["embeddings", "metadatas"]
        )
        cache[(side, bin_id)] = {
            "embeddings": np.asarray(res["embeddings"][0], np.float32),
            "vid": np.array([m["vid_num"] for m in res["metadatas"][0]]),
            "t_center": np.array([m["t_center"] for m in res["metadatas"][0]])
        }

        embs = np.asarray(res["embeddings"][0], dtype=np.float32)
        metas = res["metadatas"][0]
        print(
            f"[CACHE] ({side}, {bin_id}) "
            f"size={len(embs)} "
            f"time={time.perf_counter() - start:.2f}s"
            f"rep={metas}"
        )
    return cache

def get_retrieved(metadata, cache):
    retrieved = []
    B = len(metadata["vid"])
    for i in range(B):
        side = metadata["side"][i].numpy().decode()
        t_center = float(metadata["t_center"][i])
        t_width  = float(metadata["t_width"][i].numpy())
        vid = int(metadata["vid"][i])

        pool = cache[(side, coarse_time_bin(t_center))]
        # mask = pool["vid"] != vid
        mask = (
            (pool["vid"] != vid) &
            (np.abs(pool["t_center"] - t_center) <= t_width / 2)
        )
        embs = pool["embeddings"][mask][:config.TOP_K]

        if len(embs) < config.TOP_K:
            pad = np.zeros((config.TOP_K - len(embs), embs.shape[1]), np.float32)
            embs = np.vstack([embs, pad])

        retrieved.append(embs)

    # retrieved = tf.convert_to_tensor(np.stack(retrieved), tf.float32)
    # return tf.nn.l2_normalize(tf.stop_gradient(retrieved), axis=2)
    retrieved = tf.nn.l2_normalize(
        tf.stop_gradient(tf.convert_to_tensor(np.stack(retrieved), tf.float32)),
        axis=2
    )
    return retrieved

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def load_samples(train_vids, stride = 1, max_clips = 20):
    already_labelled = pd.read_csv("clips_label.csv")
    
    samples = []

    print(train_vids)
    for vid in train_vids:
        # clip_root = f"/home/.../clips_finalized_{vid}"

        print(vid)
        # clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}'
        clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_{vid}_smart'
        clips = sorted(os.listdir(clip_root),key=comparator) 
        # print(clips)
        clips = clips[30:40] # get 10 completely new clips  max_clips:max_clips+10

        for clip in clips:
            clip_path = os.path.join(clip_root, clip)
            print(clip_path)
            frames = sorted(os.listdir(clip_path),key=comparator) 
            
            # find label
            # label_row = already_labelled[
            #     already_labelled["clip_path"] == clip_path
            # ]
            # print(len(np.array(label_row)[0]))
            # if label_row.empty or pd.isna(label_row['label'].iloc[0]):
            #     continue
            # clip_label = int(label_row["label"].iloc[0])

            num_frames = len(frames)
            stride_counter = 0
            for i, f in enumerate(frames, start=1):
                fpath = os.path.join(clip_path, f)
                
                temp_tnorm = i / num_frames
                # if(temp_tnorm < 0.7):
                #     continue

                stride_counter += 1

                if(stride_counter == stride):
                    samples.append({
                        "pth": fpath,
                        "side": clip.split("_")[3],
                        "t_norm": i / num_frames,
                        "clip_num": int(clip.split("_")[2]),
                        "vid_num": int(f.split("_")[0][3:]),
                        # "label": clip_label
                    })

                    stride_counter = 0
    # input(np.array(samples))
    return samples

def build_chunks(frame_samples, chunk_size=60):
    """
    frame_samples: list of dictionaries (one per frame)
    chunk_size: number of frames to group into one chunk

    Returns: list of chunk dictionaries
    """

    # --- Group frames by clip ---
    clips = {}
    for s in frame_samples:
        key = (s["vid_num"], s["clip_num"])
        if key not in clips:
            clips[key] = []
        clips[key].append(s)

    # --- Sort frames within each clip ---
    for key in clips:
        clips[key].sort(key=lambda x: x["t_norm"])

    # --- Build chunks ---
    chunk_samples = []

    for (vid, clip), frames in clips.items():
        total_frames = len(frames)
        #label = frames[0]["label"]     # all frames share the clip label
        side = frames[0]["side"]       # same within a clip

        # slide with step = chunk_size (no overlap for now)
        for i in range(0, total_frames, chunk_size):
            sub = frames[i : i + chunk_size]
            if len(sub) < chunk_size:
                # OPTIONAL: skip incomplete chunk
                continue

            # frame paths
            frame_paths = [f["pth"] for f in sub]

            # compute temporal window
            t_vals = [f["t_norm"] for f in sub]
            t_center = float(sum(t_vals) / len(t_vals))
            t_width = float(max(t_vals) - min(t_vals))  # ~60 frames worth

            chunk_samples.append({
                "frames": frame_paths,    # list of image paths (length chunk_size)
                #"label": label,
                "side": side,
                "vid": vid,
                "clip": clip,
                "t_center": t_center,
                "t_width": t_width, #max(t_width, 0.4)
            })

    return chunk_samples

def build_tf_dataset_chunks(chunk_samples, batch_size, img_size=(432,768), num_workers=16):
    def gen():
        for sample in chunk_samples:
            yield sample

    output_signature = {
        "frames": tf.TensorSpec(shape=(None,), dtype=tf.string),  # list of frame paths
        # "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        "side":  tf.TensorSpec(shape=(), dtype=tf.string),
        "vid":   tf.TensorSpec(shape=(), dtype=tf.int32),
        "clip":  tf.TensorSpec(shape=(), dtype=tf.int32),
        "t_center": tf.TensorSpec(shape=(), dtype=tf.float32),
        "t_width":  tf.TensorSpec(shape=(), dtype=tf.float32),
    }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # shuffle dataset order
    # ds = ds.shuffle(2048, reshuffle_each_iteration=True) # 2048 was len(chunk_samples)

    # map to parsed tensors (loads images, builds metadata dict)
    ds = ds.map(lambda chunk: parse_chunk(chunk, img_size),
                num_parallel_calls=num_workers)

    # batching & prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    # ---- load labels ----
    # labels_df = pd.read_csv("clips_label.csv")
    # LABELS = {}

    # for _, row in labels_df.iterrows():
    #     if pd.isna(row.label):
    #         continue  # skip unlabeled clips
    #     LABELS[row.clip_path.rstrip("/")] = int(row.label)

    # ---- load chunks (YOUR EXISTING METHOD) ----
    # from dataset import load_samples, build_chunks
    samples = load_samples(config.VIDS_TO_USE, stride=1)
    chunks = build_chunks(samples, chunk_size=12)

    dataset = build_tf_dataset_chunks(chunks, batch_size=config.CHUNK_BATCH_SIZE)

    # ---- models ----
    ratt_head = RATTHead(
        hidden_size=768,
        num_queries=config.NUM_QUERIES,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS
    )
    _ = ratt_head(tf.zeros((1,768)), tf.zeros((1,10,768)), training=False)
    ratt_head.load_weights(config.RATT_WEIGHTS)

    proj_head = ProjectionHead(2304, 768*8, 768)
    _ = proj_head(tf.zeros((1,2304)))
    proj_head.load_weights(config.PROJ_WEIGHTS)

    # ---- retrieval ----
    client = PersistentClient(path="./chroma_store")
    collection = client.get_collection(config.CHROMADB_COLLECTION)
    retrieval_cache = build_retrieval_cache(collection, chunks, config.SEARCH_K)

    # ---- extract logits ----
    clip_logits = defaultdict(list)

    for frames, metadata in dataset:
        B, T = frames.shape[0], frames.shape[1]

        frame_embs = tf.numpy_function(
            hf_vit_embed_batch,
            [tf.reshape(frames, (-1, 432, 768, 3))],
            tf.float32
        )
        frame_embs = tf.reshape(frame_embs, (B, T, 768))

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
        retrieved = get_retrieved(metadata,retrieval_cache)
        # ---- new retriever end -----

        logits, fused_cls, attn_scores = ratt_head(chunk_embs, retrieved, disable_cls=False, training=False)


        # deltas = frame_embs[:,1:] - frame_embs[:,:-1]
        # raw = tf.concat([
        #     tf.reduce_mean(frame_embs,1),
        #     tf.reduce_mean(deltas,1),
        #     tf.math.reduce_std(deltas,1)
        # ], axis=1)
        # raw = tf.nn.l2_normalize(raw, axis=1)

        # chunk_embs = tf.nn.l2_normalize(proj_head(raw, training=False), axis=1)
        # retrieved = get_retrieved(metadata, retrieval_cache)

        # logits, _, _ = ratt_head(chunk_embs, retrieved, training=False)
        # logits = logits.numpy().reshape(-1)

        pprint.pprint(metadata)
        print(logits)
        for i in range(B):
            clip_dir = os.path.dirname(metadata["frames"][i][0].numpy().decode())
            clip_logits[clip_dir].append(float(logits[i]))
            # if clip_dir in LABELS:
            #     clip_logits[clip_dir].append(float(logits[i]))

    # ---- normalize + save ----
    rows = []
    for clip, seq in clip_logits.items():
        rows.append({
            "clip_path": clip,
            # "label": LABELS[clip],
            "z_sequence": z_normalize(seq).tolist(),
            "raw_sequence": seq
        })

    df = pd.DataFrame(rows)
    df.to_json("test/new_clip_z_sequences_fixed.json", orient="records", indent=2)
    print("Saved new_clip_z_sequences_fixed.json")
