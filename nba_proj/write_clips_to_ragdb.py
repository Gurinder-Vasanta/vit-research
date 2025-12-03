import chromadb
from chromadb import PersistentClient

from official.vision.modeling.backbones import vit
import tensorflow as tf, tf_keras
import cv2
import numpy as np
import os

# ------------------------------
# GLOBAL CONFIG
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

HIDDEN = 768
ENRICH_DIM = 768
TOTAL_DIM = HIDDEN * 4  # 4 deterministic components
np.random.seed(42)

# Fixed projection matrix (mathematical, NOT learned)
P = np.random.normal(0, 1/np.sqrt(TOTAL_DIM), (TOTAL_DIM, HIDDEN))


# ------------------------------
# LOAD FROZEN ViT
# ------------------------------
layers = tf_keras.layers

model = vit.VisionTransformer(
    input_specs=layers.InputSpec(shape=[None, 432, 768, 3]),
    patch_size=32,
    num_layers=12,
    num_heads=12,
    hidden_size=HIDDEN,
    mlp_dim=3072
)

model.load_weights("vit_random_weights.h5")


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

def get_frame_num(name):
    return int(name.split('_')[2].split('.')[0])

def get_side(name):
    return name.split('_')[3]


# ------------------------------
# IMAGE LOADING
# ------------------------------
def im_to_array(frame_path):
    im = cv2.imread(frame_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    target_size = (HIDDEN, 432)
    temp_frame = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return temp_frame


# ------------------------------
# DETERMINISTIC ENRICHMENT FUNCTIONS
# ------------------------------

def temporal_encoding(t_norm):
    freqs = np.linspace(1, 32, ENRICH_DIM)
    return np.sin(2 * np.pi * freqs * t_norm)

def side_mask(side_str):
    return np.ones(ENRICH_DIM) if side_str == "left" else -np.ones(ENRICH_DIM)

def frame_index_encoding(idx, total_frames):
    t = idx / total_frames
    freqs = np.linspace(1, 16, ENRICH_DIM)
    return np.cos(2 * np.pi * freqs * t)

# def clip_position_encoding(clip_norm):
#     freqs = np.linspace(1, 8, ENRICH_DIM)
#     return np.sin(2 * np.pi * freqs * clip_norm)


# ------------------------------
# MAIN ENRICH EMBEDDING FUNCTION
# ------------------------------

def enrich_embeddings(frames, t_norms, sides, frame_indices):
    """
    frames: list of image arrays (B, H, W, 3)
    t_norms: per-frame relative position in clip
    sides: 'left' or 'right'
    frame_indices: frame index inside clip
    clip_norms: clip index normalized
    """

    batch = np.array(frames)
    output = model.predict(batch, batch_size=128, verbose=1)
    base = output["pre_logits"].reshape(len(frames), HIDDEN)

    enriched = []

    for i in range(len(frames)):
        e0 = base[i]                            # (768)
        e1 = temporal_encoding(t_norms[i])      # (768)
        e2 = side_mask(sides[i])                # (768)
        e3 = frame_index_encoding(frame_indices[i], max(frame_indices))  # (768)
        # e4 = clip_position_encoding(clip_norms[i])  # (768)
        e0 = e0 / (np.linalg.norm(e0) + 1e-8)
        e1 = e1 / (np.linalg.norm(e1) + 1e-8)
        e2 = e2 / (np.linalg.norm(e2) + 1e-8)
        e3 = e3 / (np.linalg.norm(e3) + 1e-8)

        # Weighting (critical fix)
        w0, w1, w2, w3 = 0.25, 0.45, 0.20, 0.10

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
    metadata={"hnsw:space": "l2"}
)

# ------------------------------
# PROCESS VIDEOS
# ------------------------------
# vids = ["vid2", "vid4"]
vids=['vid2']
batch_cap = 128

for vid in vids:
    all_clips_path = f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}"
    clips = sorted(os.listdir(all_clips_path), key=comparator)
    clips = clips[0:3]
    input(clips)
    for clip in clips:
        print("cur clip:", clip)

        frames = sorted(os.listdir(os.path.join(all_clips_path, clip)), key=comparator)
        total_frames = len(frames)
        clip_num = get_clip_num(clip)
        # clip_norm = clip_num / 200  # assume max clip number 200 (change if needed)

        # batch buffers
        batch_imgs = []
        batch_ids = []
        batch_meta = []
        batch_tnorm = []
        batch_side = []
        batch_frame_idx = []
        # batch_clip_norm = []

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
            # batch_clip_norm.append(clip_norm)

            if len(batch_imgs) == batch_cap:
                embeddings = enrich_embeddings(
                    batch_imgs, batch_tnorm, batch_side,
                    batch_frame_idx
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
                # batch_clip_norm = []

        # leftover batch
        if len(batch_imgs) > 0:
            embeddings = enrich_embeddings(
                batch_imgs, batch_tnorm, batch_side,
                batch_frame_idx
            )

            ragdb.upsert(
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_meta
            )
