import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tf_keras
import shutil
from itertools import groupby

from official.vision.modeling.backbones import vit
import hmm   # your existing HMM module


#############################################
# CONFIG
#############################################

IMG_SIZE = (768, 432)
EMBED_DIM = 768
EPOCHS = 3000
LR = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#############################################
# 1. LOAD VIT BACKBONE (same as yours)
#############################################

def build_vit():
    layers = tf_keras.layers

    model = vit.VisionTransformer(
        input_specs=layers.InputSpec(shape=[None,432,768,3]),
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_size=EMBED_DIM,
        mlp_dim=3072
    )

    model.load_weights("vit_random_weights.h5")
    return model


#############################################
# 2. EXTRACT EMBEDDINGS
#############################################

# def extract_embeddings(vit_model, image_paths):
#     embeds = []
#     count = 0
#     for p in image_paths:
#         im = cv2.imread(p)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         im = cv2.resize(im, IMG_SIZE)

#         out = vit_model.predict(np.array([im]), verbose=0)
#         emb = out["pre_logits"].reshape(-1)

#         embeds.append(emb)
#         count +=1 
#         print(f'{count}/{len(image_paths)}')

#     return np.stack(embeds)  # (N, 768)

def extract_embeddings(vit_model, image_paths, batch_size=1024):
    all_embeds = []

    num_batches = int(len(image_paths)/batch_size)+1
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        batch_imgs = []

        for p in batch_paths:
            im = cv2.imread(p)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (768, 432))
            batch_imgs.append(im)


        batch_imgs = np.array(batch_imgs)

        out = vit_model.predict(batch_imgs, verbose=0)
        emb = out["pre_logits"].reshape(len(batch_imgs), -1)
        # emb = out["pre_logits"]      # (B, 768)

        all_embeds.append(emb)
        print(f'{(i/batch_size)+1}/{num_batches}')

    return np.vstack(all_embeds)



#############################################
# 3. BUILD LABELS FROM CSV
#############################################

def build_labels(frame_names, csv_path):
    """
    Returns:
        y : shape (N,)
            -1 for unlabeled frames (ignored by loss)
             0 left
             1 right
             2 none
    """

    mapping = {"left":0, "right":1, "none":2}

    # default = ignore
    y = np.full(len(frame_names), -1, dtype=int)

    name_to_idx = {f:i for i,f in enumerate(frame_names)}

    with open(csv_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            for side in ["left", "right", "none"]:
                s = row[f"{side}_start"]
                e = row[f"{side}_end"]

                if not s or not e:
                    continue

                vid = s.split("_")[0]
                start = int(s.split("_")[1])
                end   = int(e.split("_")[1])

                for i in range(start, end+1):
                    fname = f"{vid}_frame_{i}.jpg"

                    if fname in name_to_idx:
                        y[name_to_idx[fname]] = mapping[side]

    return y



# def build_labels(frame_names, csv_path):
#     labels = {f: "none" for f in frame_names}

#     with open(csv_path) as f:
#         reader = csv.DictReader(f)

#         for row in reader:
#             for side in ["left", "right", "none"]:
#                 s = row[f"{side}_start"]
#                 e = row[f"{side}_end"]

#                 if not s or not e:
#                     continue

#                 vid = s.split("_")[0]
#                 start = int(s.split("_")[1])
#                 end   = int(e.split("_")[1])

#                 for i in range(start, end+1):
#                     fname = f"{vid}_{i}.jpg"
#                     labels[fname] = side

#     mapping = {"left":0, "right":1, "none":2}
#     return np.array([mapping[f] if isinstance(f,int) else mapping[labels[f]] for f in frame_names])


#############################################
# 4. TEMPORAL CNN HEAD
#############################################

# class TemporalHead(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv1d(EMBED_DIM, 128, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.Conv1d(128, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(64, 3, kernel_size=1)
#         )

#     def forward(self, x):  # (B,T,768)
#         x = x.transpose(1,2)
#         x = self.net(x)
#         return x.transpose(1,2)
class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(EMBED_DIM, 256, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, 3, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.net(x)
        return x.transpose(1,2)



#############################################
# 5. TRAIN
#############################################

# def train_model(E, y):
#     model = TemporalHead().to(DEVICE)

#     X = torch.tensor(E, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     Y = torch.tensor(y, dtype=torch.long).unsqueeze(0).to(DEVICE)

#     opt = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.CrossEntropyLoss()

#     for epoch in range(EPOCHS):
#         opt.zero_grad()

#         logits = model(X)
#         loss = loss_fn(logits.view(-1,3), Y.view(-1))

#         loss.backward()
#         opt.step()

#         print(f"epoch {epoch} loss {loss.item():.4f}")

#     return model

def train_model(E, y):
    model = TemporalHead().to(DEVICE)
    
    X = torch.tensor(E, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    Y = torch.tensor(y, dtype=torch.long).unsqueeze(0).to(DEVICE)

    print(X)
    print(Y)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(EPOCHS):
        opt.zero_grad()

        logits = model(X)
        loss = loss_fn(logits.view(-1,3), Y.view(-1))

        loss.backward()
        opt.step()

        print(f"epoch {epoch} loss {loss.item():.4f}")

    return model



#############################################
# 6. INFERENCE
#############################################

def predict_probs(model, E):
    model.eval()

    X = torch.tensor(E, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)

    return probs.squeeze(0).cpu().numpy()



def copy_clip(frames, src_dir, out_root, clip_id, label, vid):
    clip_dir = os.path.join(out_root, f"{vid}_clip_{clip_id}_{label}")
    os.makedirs(clip_dir, exist_ok=True)

    for f in frames:
        shutil.copy(
            os.path.join(src_dir, f),
            os.path.join(clip_dir, f)
        )


def save_clips_from_sequence(decoded, frame_names, src_dir,
                             out_root="clips_output",
                             min_len=75,
                             pad=50,
                             vid='vid0'):
    """
    decoded: list[str]  ('left','right','none')
    frame_names: filenames in order
    min_len: ignore tiny streaks (noise)
    pad: extend clip by +/- pad frames
    """

    os.makedirs(out_root, exist_ok=True)

    clip_id = 0
    N = len(decoded)

    i = 0
    while i < N:

        cur = decoded[i]
        start = i

        while i < N and decoded[i] == cur:
            i += 1

        end = i - 1
        length = end - start + 1

        if cur != "none" and length >= min_len:

            clip_id += 1

            s = max(0, start - pad)
            e = min(N-1, end + pad)

            frames = frame_names[s:e+1]

            copy_clip(frames, src_dir, out_root, clip_id, cur,vid)

    print(f"Saved {clip_id} clips.")


#############################################
# 7. MAIN PIPELINE
#############################################
def frame_key(fname):
    # vid1_12345.jpg → 12345
    # print(fname)
    return int(fname.split('_')[2].split('.')[0])

def run(video_dir, manual_csv, vid):

    # sort frames
    frame_names = sorted(os.listdir(video_dir),key=frame_key)

    image_paths = [os.path.join(video_dir, f) for f in frame_names]

    print(image_paths[0:4])
    print("Loading ViT...")
    vit_model = build_vit()

    # image_paths = image_paths[0:100000]
    # frame_names = frame_names[0:100000]
    print("Extracting embeddings...")
    E = extract_embeddings(vit_model, image_paths)

    print("Building labels...")
    y = build_labels(frame_names, manual_csv)
    # model = train_model(E, y)
    # y = build_labels(frame_names, manual_csv)
    # y, mask = build_labels(frame_names, manual_csv)
    # print(mask)
    # print("E shape:", E.shape)
    # print("E[mask] shape:", E[mask].shape)
    # model = train_model(E[mask], y[mask])

    print("unique labels:", np.unique(y))
    print("labeled count:", (y!=-1).sum())
    print("class counts:", np.bincount(y[y!=-1]))

    if os.path.exists(f"data/unseen_test_images/frame_nns/temporal_head_{vid}.pt"):
        print("Loading saved model...")
        model = TemporalHead().to(DEVICE)
        model.load_state_dict(torch.load(f"data/unseen_test_images/frame_nns/temporal_head_{vid}.pt", map_location=DEVICE))
        model.eval()
    else:
        print("Training temporal head...")
        model = train_model(E, y)
        torch.save(model.state_dict(), f"data/unseen_test_images/frame_nns/temporal_head_{vid}.pt")

    # print("Training temporal head...")
    # model = train_model(E, y)

    # torch.save(model.state_dict(), f"/home/vasantgc/venv/nba_proj/data/unseen_test_images/frame_nns/temporal_head_{vid}.pt")
    print("Predicting...")
    probs = predict_probs(model, E)

    print(probs)
    print("mean probs:", probs.mean(axis=0))
    print("unique argmax:", np.unique(np.argmax(probs, axis=1)))
    print("Running HMM smoothing...")
    # print(len(probs))
    hmm_matrix = hmm.hmm(len(probs)+2)

    for p in probs:
        hmm_matrix.add_col_to_lattice({
            'left': float(p[0]),
            'right': float(p[1]),
            'none': float(p[2])
        })
        # print(hmm_matrix.dp)

    decoded = hmm_matrix.decode_sequence()
    print(decoded)
    save_clips_from_sequence(
        decoded,
        frame_names,
        src_dir=video_dir,
        out_root=f"data/unseen_test_images/smarter_clips/clips_hmm_smooth_{vid}_smart",
        min_len=100,   # your old threshold
        pad=0,        # your old +100 frame extension trick
        vid=vid
    )
    print("Done.")
    return decoded


#############################################

if __name__ == "__main__":
    vid='vid7'
    decoded = run(
        video_dir=f"data/unseen_test_images/smarter_ims/ims_{vid}",
        manual_csv="/home/vasantgc/venv/nba_proj/data/manual_intervals.csv",
        vid=vid
    )

    lengths = [(k, len(list(g))) for k,g in groupby(decoded)]
    for b in lengths: 
        print(b)
    # print(lengths[:20])
