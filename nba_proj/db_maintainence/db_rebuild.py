import numpy as np
from chromadb import PersistentClient
import tensorflow as tf
import cv2
import os

from models.projection_head import ProjectionHead
import torch
from transformers import ViTModel, ViTImageProcessor


# ------------------------------
# CONFIG
# ------------------------------
HIDDEN = 768
ENRICH_DIM = 768
SIDE_DIM = 1
INPUT_DIM = 768 + 768 + 1 + 768   # 2305

TEMP_FREQS = np.linspace(5, 300, ENRICH_DIM)
TEMP_PHASES = np.random.uniform(0, 2*np.pi, ENRICH_DIM)

FRAME_FREQS = np.linspace(1, 16, ENRICH_DIM)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# LOAD ViT
# ------------------------------
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model  = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()


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

def im_to_array(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


# ------------------------------
# ENCODING FUNCTIONS (numpy-only)
# ------------------------------
def temporal_encoding(t_norm):
    t = t_norm ** 1.5
    arr = np.sin(2*np.pi*TEMP_FREQS*t + TEMP_PHASES)
    return arr / (np.linalg.norm(arr)+1e-8)

def side_mask(side_str):
    arr = np.array([1.0]) if side_str == "left" else np.array([-1.0])
    return arr

def frame_index_encoding(idx, total):
    t = idx / total
    arr = np.cos(2*np.pi*FRAME_FREQS*t)
    return arr / (np.linalg.norm(arr)+1e-8)

def vit_embed(img_np):
    inputs = processor(images=img_np, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vit_model(**inputs)
    cls = out.last_hidden_state[:,0,:].cpu().numpy()[0]
    return cls / (np.linalg.norm(cls)+1e-8)


# ------------------------------
# ENRICH (numpy-only)
# ------------------------------
def enrich_for_training(vit_base, t_norm, side, frame_idx, total_frames):
    e0 = vit_base
    e1 = temporal_encoding(t_norm)
    e2 = side_mask(side)
    e3 = frame_index_encoding(frame_idx, total_frames)

    return np.concatenate([
        0.4 * e0,
        0.15 * e1,
        0.35 * e2,
        0.10 * e3
    ], axis=0).astype(np.float32)


# ------------------------------
# LOAD PROJECTOR
# ------------------------------
projector = ProjectionHead(input_dim=INPUT_DIM)
projector.build((None, INPUT_DIM))
projector.load_weights("projection_head.h5")

projector.call = tf.function(projector.call)

# ------------------------------
# CHROMA SETUP
# ------------------------------
client = PersistentClient(path="./chroma_store")
ragdb = client.get_or_create_collection(
    name="ragdb_p32_rich_embeddings",
    metadata={"hnsw:space": "l2"}
)

ragdb.delete(where={})  # wipe DB


# ------------------------------
# REBUILD LOOP
# ------------------------------
vids = ["vid2", "vid4"]
batch_cap = 128

for vid in vids:
    root = f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}"
    clips = sorted(os.listdir(root), key=comparator)

    for clip in clips:
        frames = sorted(os.listdir(os.path.join(root,clip)), key=comparator)
        total_frames = len(frames)

        batch_embs, batch_ids, batch_meta = [], [], []

        for i, fname in enumerate(frames, start=1):
            path = os.path.join(root, clip, fname)

            base = vit_embed(im_to_array(path))
            enriched = enrich_for_training(
                base,
                t_norm=i/total_frames,
                side=get_side(clip),
                frame_idx=i,
                total_frames=total_frames
            )

            # project into learned 768-dim space
            emb = projector(enriched[None,:]).numpy()[0]

            batch_embs.append(emb)
            batch_ids.append(fname)
            batch_meta.append({
                "side": get_side(clip),
                "t_norm": i/total_frames,
                "clip_num": get_clip_num(clip),
                "vid_num": int(fname.split("_")[0][3:])
            })

            if len(batch_embs) == batch_cap:
                ragdb.upsert(embeddings=batch_embs, ids=batch_ids, metadatas=batch_meta)
                batch_embs, batch_ids, batch_meta = [], [], []

        if len(batch_embs) > 0:
            ragdb.upsert(embeddings=batch_embs, ids=batch_ids, metadatas=batch_meta)
