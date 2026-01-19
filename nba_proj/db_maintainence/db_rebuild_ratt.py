import pprint
import chromadb
from chromadb import PersistentClient

# REMOVE tensorflow ViT
# from official.vision.modeling.backbones import vit
import tensorflow as tf, tf_keras

import cv2
import numpy as np
import os

# NEW: PyTorch + transformers imports
import torch
from transformers import ViTModel, ViTImageProcessor
from multiprocessing import Pool
import functools
import time
# import config_exp as config
import config_ratt
import dataset
from models.projection_head import ProjectionHead
 
# ------------------------------
# GLOBAL CONFIG
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

HIDDEN = 768
ENRICH_DIM = 768
SIDE_DIM = 1
TOTAL_DIM = 2 * HIDDEN + ENRICH_DIM + SIDE_DIM 
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# LOAD PRETRAINED GOOGLE VIT
# ------------------------------
# Automatically handles resizing + center cropping.
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()

projector = ProjectionHead(input_dim = 2304, hidden_dim=768, proj_dim=768)
projector.build((None, 2304))
projector.load_weights(config_ratt.PROJ_WEIGHTS)

_ = projector(tf.zeros((1, 2304), dtype=tf.float32))
projector.call = tf.function(projector.call)

# time it took: 421.1687158672139

# ------------------------------
# HELPERS
# ------------------------------
def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3:])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def get_clip_num(name):
    return int(name.split('_')[2])

def get_side(name):
    return name.split('_')[3]


# ------------------------------
# IMAGE LOADING
# ------------------------------
def im_to_array(frame_path):
    """
    Return resized numpy image as (432,768,3).
    Google ViT processor will handle final 224x224 transform internally.
    """
    im = cv2.imread(frame_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
    # return cv2.resize(im, (768, 432), interpolation=cv2.INTER_AREA)

def worker_prepare_frame(args):
    frame_path, local_idx, total_frames, clip_num = args

    img = im_to_array(frame_path)

    # frame_idx the global frame number from filename, but we don't use it for t_norm
    fname = os.path.basename(frame_path)
    global_frame_idx = int(fname.split("_")[2].split('.')[0])

    clip_folder = os.path.basename(os.path.dirname(frame_path))
    side = get_side(clip_folder)

    # Correct t_norm:
    t_norm = local_idx / total_frames   # local_idx is 1..N

    return (img, t_norm, side, local_idx, fname)

def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, 432, 768, 3) uint8 or float32
    Returns (N, 768) numpy embeddings (L2 normalized)
    """
    frames_np = frames_np.astype(np.float32)
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = processor(images=frames_list, return_tensors="pt").to(device)
        out = vit_model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (N,768)
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)

# ------------------------------
# ENRICHMENT FUNCTIONS (unchanged)
# ------------------------------
def temporal_encoding(t_norm):
    # higher, nonlinear, randomized phase encoding
    freqs = np.linspace(5, 300, ENRICH_DIM)       # FAST oscillation
    phases = np.random.uniform(0, 2*np.pi, ENRICH_DIM)

    # nonlinear time warp
    t = t_norm ** 1.5

    return np.sin(2 * np.pi * freqs * t + phases)
    # freqs = np.linspace(5, 300, ENRICH_DIM)
    # return np.sin(2 * np.pi * freqs * t_norm)

def side_mask(side_str):
    return np.ones(SIDE_DIM) if side_str == "left" else -np.ones(SIDE_DIM)

def frame_index_encoding(idx, total_frames):
    t = idx / total_frames
    freqs = np.linspace(1, 16, ENRICH_DIM)
    return np.cos(2 * np.pi * freqs * t)


# ------------------------------
# FIXED PROJECTION MATRIX
# ------------------------------



# ------------------------------
# MAIN ENRICH EMBEDDING FN
# ------------------------------
   
# ------------------------------
# CHROMA CLIENT
# ------------------------------
client = PersistentClient(path="./chroma_store")
# chromadb hnsw metadata keys; search this up and go to the ai overview
ratt_db = client.get_or_create_collection(
    name = 'ratt_db',
    # name = 'rich_embeddings_7_vids_bs4_cls_rag_temp_12_epochs_rebuild',
    # name='rich_embeddings_7_vids_bs4_cls_rag_12_epochs_rebuild',
    metadata={"hnsw:space":"cosine",
              }
    )


# ------------------------------
# PROCESS VIDEOS (unchanged)
# ------------------------------
vids = config_ratt.VIDS_TO_USE

def rebuild_db():
    ratt_db.delete(where={"vid_num": {"$ne": 'vid0'}})
    samples = dataset.load_samples(vids,stride=1,max_clips=config_ratt.NUM_CLIPS_PER_VID)
    chunk_samples = dataset.build_chunks(samples, chunk_size=12)
    # pprint.pprint(chunk_samples[0])
    chunked_ds = dataset.build_tf_dataset_chunks(chunk_samples, batch_size=1)

    b_embeddings = []
    b_ids = []
    b_metadatas = []

    upsert_size = 512
    # input(len(chunked_ds))
    start = time.perf_counter()
    for frames_batch, metadata_batch, labels_batch in chunked_ds:
        B = tf.shape(frames_batch)[0]
        T = tf.shape(frames_batch)[1]
        # pprint.pprint(metadata_batch)
        # pprint.pprint(labels_batch)
        # input('stop')
        # print(
        #     tf.reduce_min(frames_batch),
        #     tf.reduce_max(frames_batch),
        #     tf.reduce_mean(frames_batch)
        # )
        # input('stop')
        # input(frames_batch)
        frames_np = tf.numpy_function(
            hf_vit_embed_batch,
            [tf.reshape(frames_batch, (-1, 432, 768, 3))],
            tf.float32
        )
        frame_embs = tf.reshape(frames_np, (B, T, 768))
        deltas = frame_embs[:, 1:, :] - frame_embs[:, :-1, :]
        
        mean = tf.reduce_mean(frame_embs, axis=1)
        mean_deltas = tf.reduce_mean(deltas, axis=1)
        std_deltas = tf.math.reduce_std(deltas, axis=1)

        raw_chunk = tf.concat([mean, mean_deltas, std_deltas], axis=-1)
        raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=1)

        projected = projector(tf.convert_to_tensor(raw_chunk, dtype=tf.float32)).numpy()

        b_embeddings.append(projected)

        cur_chunk_id = f"vid{int(metadata_batch['vid'][0])}_clip_{int(metadata_batch['clip'][0])}_t_{float(metadata_batch['t_center'][0]):.3f}"
        print(f'upserting id: {cur_chunk_id}')
        b_ids.append(cur_chunk_id)
        meta = {
            "vid_num": int(metadata_batch["vid"][0]),
            "clip_num": int(metadata_batch["clip"][0]),
            "side": metadata_batch["side"][0].numpy().decode(),
            "t_center": float(metadata_batch["t_center"][0]),
            "t_width": float(metadata_batch["t_width"][0]),
        }

        b_metadatas.append(meta)


        if(len(b_embeddings) == upsert_size): 
            embeddings = np.concatenate(b_embeddings, axis=0)
            ids = b_ids
            metadatas = b_metadatas
            
            ratt_db.upsert(
                    embeddings = np.concatenate(b_embeddings, axis=0), 
                    ids = b_ids, 
                    metadatas = b_metadatas
                )
            print('upserted batch')
            b_embeddings = []
            b_ids = []
            b_metadatas = []

    if(len(b_embeddings) != 0):
            print(len(b_embeddings))
            embeddings = np.concatenate(b_embeddings, axis=0)
            ids = b_ids
            metadatas = b_metadatas
            
            ratt_db.upsert(
                    embeddings = np.concatenate(b_embeddings, axis=0), 
                    ids = b_ids, 
                    metadatas = b_metadatas
                )
            print('upserted leftover batch')
    end = time.perf_counter()
    print(f'time it took: {end - start}')
