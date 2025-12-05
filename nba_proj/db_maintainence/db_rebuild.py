import numpy as np
import tensorflow as tf
import cv2, os
from multiprocessing import cpu_count

from chromadb import PersistentClient
from models.projection_head import ProjectionHead

import torch
from transformers import ViTModel, ViTImageProcessor


# --------------------------
# DEVICE + SEED
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# LOAD ViT (raw embeddings)
# --------------------------
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.eval()


# --------------------------
# HELPERS
# --------------------------
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
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def vit_embed(img_np):
    """Returns a 768-dim normalized CLS embedding."""
    inputs = processor(images=img_np, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vit_model(**inputs)

    cls = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return cls / (np.linalg.norm(cls) + 1e-8)


# --------------------------
# LOAD TRAINED PROJECTOR
# --------------------------
projector = ProjectionHead(input_dim = 768, hidden_dim=768, proj_dim=768)
projector.build((None, 768))
projector.load_weights("projection_head.weights.h5")

projector.call = tf.function(projector.call)

client = PersistentClient(path="./chroma_store")

ragdb = client.get_or_create_collection(
    name="ragdb_p32_rich_embeddings",     # NEW, cleaner name
    metadata={"hnsw:space": "l2"}
)

# wipe old DB
# ragdb.delete(where={})

def rebuild_db():
    ragdb.delete(where={"vid_num": {"$ne": 'vid0'}})
    # client.delete_collection(name=f'ragdb_p32_rich_embeddings')

    # ragdb = client.get_or_create_collection(
    #     name="ragdb_p32_rich_embeddings",     # NEW, cleaner name
    #     metadata={"hnsw:space": "l2"}
    # )
    
    vids = ["vid2", "vid4"]
    batch_cap = 128

    for vid in vids:
        root = f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}"
        clips = sorted(os.listdir(root), key=comparator)
        clips = clips[0:5]
        for clip in clips:
            print("rebuilding cur clip:", clip)
            frames = sorted(os.listdir(os.path.join(root, clip)), key=comparator)

            base_list = []
            ids = []
            metas = []

            # ---------------------------------------
            # 1. Collect raw ViT base embeddings
            # ---------------------------------------
            count = 0
            for fname in frames:
                path = os.path.join(root, clip, fname)
                img = im_to_array(path)

                base = vit_embed(img)     # 768-dim
                base_list.append(base)

            #  batch_meta.append({
            #     "side": get_side(clip),
            #     "t_norm": f_i / total_frames,
            #     "clip_num": clip_num,
            #     "vid_num": int(fname.split("_")[0][3:])
            # })

                ids.append(fname)
                metas.append({
                    "clip_num": get_clip_num(clip),
                    "side": get_side(clip),
                    "vid_num": vid,
                    't_norm': count/len(frames)
                })

                count += 1

            base_arr = np.stack(base_list, axis=0)

            # ---------------------------------------
            # 2. Project using trained projection head
            # ---------------------------------------
            projected = projector(base_arr).numpy()

            # ---------------------------------------
            # 3. Write to DB in batches
            # ---------------------------------------
            start = 0
            n = len(projected)
            while start < n:
                end = min(start + batch_cap, n)

                ragdb.upsert(
                    embeddings=projected[start:end],
                    ids=ids[start:end],
                    metadatas=metas[start:end]
                )

                start = end


# import numpy as np
# import tensorflow as tf
# import cv2, os
# from multiprocessing import Pool, cpu_count

# from chromadb import PersistentClient
# from models.projection_head import ProjectionHead

# import torch
# from transformers import ViTModel, ViTImageProcessor

# np.random.seed(1234)
# # ------------------------------
# # CONFIG
# # ------------------------------
# HIDDEN = 768
# ENRICH_DIM = 768
# SIDE_DIM = 1
# INPUT_DIM = 768 + 768 + 1 + 768   # 2305

# TEMP_FREQS = np.linspace(5, 300, ENRICH_DIM)
# TEMP_PHASES = np.random.uniform(0, 2*np.pi, ENRICH_DIM)

# FRAME_FREQS = np.linspace(1, 16, ENRICH_DIM)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
# np.random.seed(42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------------
# # LOAD ViT
# # ------------------------------
# processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# vit_model  = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
# vit_model.eval()


# # ------------------------------
# # HELPERS
# # ------------------------------
# def comparator(fname):
#     splitted = fname.split('_')
#     vid_num = int(splitted[0][3:])
#     frame_num = int(splitted[2].split('.')[0])
#     return (vid_num, frame_num)

# def get_clip_num(name):
#     return int(name.split('_')[2])

# def get_side(name):
#     return name.split('_')[3]

# def im_to_array(path):
#     im = cv2.imread(path)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     return im


# # ------------------------------
# # ENCODING FUNCTIONS (numpy-only)
# # ------------------------------
# def temporal_encoding(t_norm):
#     t = t_norm ** 1.5
#     arr = np.sin(2*np.pi*TEMP_FREQS*t + TEMP_PHASES)
#     return arr / (np.linalg.norm(arr)+1e-8)

# def side_mask(side_str):
#     arr = np.array([1.0]) if side_str == "left" else np.array([-1.0])
#     return arr

# def frame_index_encoding(idx, total):
#     t = idx / total
#     arr = np.cos(2*np.pi*FRAME_FREQS*t)
#     return arr / (np.linalg.norm(arr)+1e-8)

# def vit_embed(img_np):
#     inputs = processor(images=img_np, return_tensors="pt").to(device)
#     with torch.no_grad():
#         out = vit_model(**inputs)
#     cls = out.last_hidden_state[:,0,:].cpu().numpy()[0]
#     return cls / (np.linalg.norm(cls)+1e-8)


# # =============================
# # CPU-ONLY WORKER FUNCTION
# # =============================
# def cpu_enrich_worker(args):
#     base, t_norm, side, frame_idx, total_frames = args
    
#     # All numpy ops, safe inside multiprocessing
#     e1 = temporal_encoding(t_norm)
#     e2 = side_mask(side)
#     e3 = frame_index_encoding(frame_idx, total_frames)

#     enriched = np.concatenate([
#         0.4 * base,
#         0.15 * e1,
#         0.35 * e2,
#         0.10 * e3
#     ], axis=0)

#     # normalize
#     enriched = enriched.astype(np.float32)
#     enriched = enriched / (np.linalg.norm(enriched) + 1e-8)
#     return enriched

# # # ------------------------------
# # # ENRICH (numpy-only)
# # # ------------------------------
# # def enrich_for_training(vit_base, t_norm, side, frame_idx, total_frames):
# #     e0 = vit_base
# #     e1 = temporal_encoding(t_norm)
# #     e2 = side_mask(side)
# #     e3 = frame_index_encoding(frame_idx, total_frames)

# #     return np.concatenate([
# #         0.4 * e0,
# #         0.15 * e1,
# #         0.35 * e2,
# #         0.10 * e3
# #     ], axis=0).astype(np.float32)


# # ------------------------------
# # LOAD PROJECTOR
# # ------------------------------
# projector = ProjectionHead(input_dim=INPUT_DIM)
# projector.build((None, INPUT_DIM))
# projector.load_weights("projection_head.h5")

# # projector.call = tf.function(projector.call)

# # ------------------------------
# # CHROMA SETUP
# # ------------------------------
# client = PersistentClient(path="./chroma_store")
# ragdb = client.get_or_create_collection(
#     name="ragdb_p32_rich_embeddings",
#     metadata={"hnsw:space": "l2"}
# )

# ragdb.delete(where={})  # wipe DB


# # ------------------------------
# # REBUILD LOOP
# # ------------------------------
# def rebuild_db():
#     vids = ["vid2", "vid4"]
#     batch_cap = 128
#     MAX_WORKERS = min(cpu_count(), 32)

#     for vid in vids:
#         root = f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}"
#         clips = sorted(os.listdir(root), key=comparator)

#         for clip in clips:
#             frames = sorted(os.listdir(os.path.join(root, clip)), key=comparator)
#             total_frames = len(frames)

#             # --- Precompute base embeddings + metadata ---
#             jobs = []
#             ids = []
#             metas = []

#             for i, fname in enumerate(frames, start=1):
#                 path = os.path.join(root, clip, fname)
#                 img = im_to_array(path)
#                 base = vit_embed(img)

#                 jobs.append((base, i/total_frames, get_side(clip), i, total_frames))

#                 ids.append(fname)
#                 metas.append({
#                     "side": get_side(clip),
#                     "t_norm": i/total_frames,
#                     "clip_num": get_clip_num(clip),
#                     "vid_num": int(fname.split("_")[0][3:])
#                 })

#             # --- Parallel CPU enrichment ---
#             with Pool(processes=MAX_WORKERS) as pool:
#                 enriched = pool.map(cpu_enrich_worker, jobs)

#             enriched = np.stack(enriched, axis=0)

#             # --- GPU projection + DB batch writes ---
#             start = 0
#             n = len(enriched)

#             while start < n:
#                 end = min(start + batch_cap, n)

#                 proj_batch = projector(enriched[start:end]).numpy()
#                 id_batch = ids[start:end]
#                 meta_batch = metas[start:end]

#                 ragdb.upsert(
#                     embeddings=proj_batch,
#                     ids=id_batch,
#                     metadatas=meta_batch
#                 )

#                 start = end
