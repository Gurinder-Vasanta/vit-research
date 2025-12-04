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

 
# ------------------------------
# GLOBAL CONFIG
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

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


# ------------------------------
# REPLACE TF VIT WITH PYTORCH EMBEDDING FUNCTION
# ------------------------------

# def vit_embed_pytorch(img_np):
#     """
#     img_np: raw numpy image (H, W, 3) in RGB.
#             DO NOT resize or crop beforehand.
#             ViTImageProcessor will do this correctly.

#     Returns:
#         768-dim L2-normalized patch-mean embedding (numpy array)
#     """

#     # Convert to torch input through ViT processor
#     inputs = processor(
#         images=img_np,
#         return_tensors="pt"
#     ).to(device)

#     # Forward pass (no gradients)
#     with torch.no_grad():
#         out = vit_model(**inputs)

#     # out.last_hidden_state: (1, num_tokens, 768)
#     # token 0 = CLS, skip it
#     tokens = out.last_hidden_state[:, 1:, :]      # shape: (1, N, 768)

#     # Take mean over all patch tokens
#     patch_mean = tokens.mean(dim=1)               # shape: (1, 768)

#     # Convert to numpy
#     emb = patch_mean.cpu().numpy()[0]

#     # Normalize
#     emb = emb / (np.linalg.norm(emb) + 1e-8)

#     return emb

def vit_embed_pytorch(img_np):
    """
    img_np: raw numpy image (H,W,3), already resized to 432x768
    Returns: 768-dim CLS embedding (numpy)
    """
    inputs = processor(images=img_np, return_tensors="pt").to(device)

    with torch.no_grad():
        out = vit_model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (1, 768)

    return cls.cpu().numpy()[0]

    #     patches = out.last_hidden_state[:, 1:, :]   # remove CLS
    #     avg_emb = patches.mean(dim=1)
    # return (avg_emb.cpu().numpy()[0])

    #     cls = out.last_hidden_state[:, 0, :]  # (1, 768)

    # return cls.cpu().numpy()[0]


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
P = np.random.normal(0, 1/np.sqrt(TOTAL_DIM), (TOTAL_DIM, HIDDEN))


# ------------------------------
# MAIN ENRICH EMBEDDING FN
# ------------------------------
def enrich_embeddings(frames, t_norms, sides, frame_indices):

    # NEW: replace TF model.predict()
    base_list = [vit_embed_pytorch(img) for img in frames]
    base = np.stack(base_list, axis=0)

    enriched = []

    for i in range(len(frames)):

        e0 = base[i]
        e1 = temporal_encoding(t_norms[i])
        e2 = side_mask(sides[i])
        e3 = frame_index_encoding(frame_indices[i], max(frame_indices))

        # normalize
        e0 = e0 / (np.linalg.norm(e0)+1e-8)
        e1 = e1 / (np.linalg.norm(e1)+1e-8)
        e2 = e2 / (np.linalg.norm(e2)+1e-8)
        e3 = e3 / (np.linalg.norm(e3)+1e-8)

        # Weighting
        # w0, w1, w2, w3 = 0.35, 0.35, 0.25, 0.05 <-- this worked the best so far (iter 1)
        # w0, w1, w2, w3 = 0.35, 0.25, 0.35, 0.05 <-- this worked even better (iter 2)
        # w0, w1, w2, w3 = 0.35, 0.20, 0.4, 0.05
        w0, w1, w2, w3 = 0.4, 0.15, 0.35, 0.10  #<-- this worked even better (iter 3)

        # w0, w1, w2, w3 = 0.4, 0.05, 0.4, 0.15 <-- this was bad
        concat = np.concatenate([
            w0 * e0,
            w1 * e1,
            w2 * e2,
            w3 * e3
        ])

        proj = concat @ P
        proj = proj / (np.linalg.norm(proj) + 1e-8)

        enriched.append(proj)

    return np.array(enriched, dtype=np.float32)


# ------------------------------
# CHROMA CLIENT
# ------------------------------
client = PersistentClient(path="./chroma_store")
ragdb = client.get_or_create_collection(
    name="ragdb_p32_rich_embeddings",
    metadata={"hnsw:space":"l2"}
)


# ------------------------------
# PROCESS VIDEOS (unchanged)
# ------------------------------
vids = ["vid2",'vid4']
batch_cap = 128
# nvidia-cudnn-cu12 9.3.0.75
for vid in vids:
    all_clips_path = f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}"
    clips = sorted(os.listdir(all_clips_path), key=comparator)
    # clips = clips[0:3]
    # print(clips)

    for clip in clips:
        print("cur clip:", clip)

        frames = sorted(os.listdir(os.path.join(all_clips_path, clip)), key=comparator)
        total_frames = len(frames)
        clip_num = get_clip_num(clip)

        # batches
        batch_imgs, batch_ids, batch_meta = [], [], []
        batch_tnorm, batch_side, batch_frame_idx = [], [], []

        for f_i, fname in enumerate(frames, start=1):

            full_path = os.path.join(all_clips_path, clip, fname)
            batch_imgs.append(im_to_array(full_path))
            batch_ids.append(fname)

            batch_meta.append({
                "side": get_side(clip),
                "t_norm": f_i / total_frames,
                "clip_num": clip_num,
                "vid_num": int(fname.split("_")[0][3:])
            })

            batch_tnorm.append(f_i / total_frames)
            batch_side.append(get_side(clip))
            batch_frame_idx.append(f_i)

            if len(batch_imgs) == batch_cap:

                embeddings = enrich_embeddings(
                    batch_imgs, batch_tnorm, batch_side, batch_frame_idx
                )

                ragdb.upsert(
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_meta
                )

                # reset
                batch_imgs = []
                batch_ids = []
                batch_meta = []

                batch_tnorm = []
                batch_side = []
                batch_frame_idx = []

        # leftover
        if len(batch_imgs) > 0:
            embeddings = enrich_embeddings(
                batch_imgs, batch_tnorm, batch_side, batch_frame_idx
            )

            ragdb.upsert(
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_meta
            )