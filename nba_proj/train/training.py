import os
import pandas as pd
import numpy as np
import tensorflow as tf, tf_keras
from chromadb import PersistentClient

from models.vit_backbone import VisionTransformer
from models.rag_head import RAGHead
from retrieval.frame_retriever import FrameRetriever
from dataset import build_tf_dataset


layers = tf_keras.layers
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# usage: python -m train.training

# -------------------------
# 1. Load labels + build samples
# -------------------------
def load_samples(train_vids):
    already_labelled = pd.read_csv("clips_label.csv")
    
    samples = []

    for vid in train_vids:
        # clip_root = f"/home/.../clips_finalized_{vid}"

        clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}'
        clips = sorted(os.listdir(clip_root))

        for clip in clips:
            clip_path = os.path.join(clip_root, clip)
            frames = sorted(os.listdir(clip_path))

            # find label
            label_row = already_labelled[
                already_labelled["clip_path"] == clip_path
            ]
            # print(len(np.array(label_row)[0]))
            if label_row.empty or pd.isna(label_row['label'].iloc[0]):
                continue
            clip_label = int(label_row["label"])

            num_frames = len(frames)
            for i, f in enumerate(frames, start=1):
                fpath = os.path.join(clip_path, f)

                samples.append({
                    "pth": fpath,
                    "side": clip.split("_")[3],
                    "t_norm": i / num_frames,
                    "clip_num": int(clip.split("_")[2]),
                    "vid_num": int(f.split("_")[0][3:]),
                    "label": clip_label
                })

    return samples


# -------------------------
# 2. Train Step
# -------------------------
# @tf.function
def train_step(vit, rag_head, retriever, optimizer, loss_fn,
               frames, metadata, labels):

    with tf.GradientTape() as tape:
        vit_out = vit(frames, training=True)
        cls_embeddings = vit_out["pre_logits"]  # (B, 768)

        # Python retrieval step
        # cls_np = cls_embeddings.numpy()
        # retrieved_np = retriever(cls_embeddings, metadata_batch)
        # retrieved_embeddings = tf.convert_to_tensor(retrieved_np, dtype=tf.float32)
        retrieved_embeddings = retriever(cls_embeddings, metadata_batch)

        logits, _ = rag_head(cls_embeddings, retrieved_embeddings, training=True)
        loss = loss_fn(labels, logits)

    train_vars = vit.trainable_variables + rag_head.trainable_variables
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))

    return loss


# -------------------------
# 3. Main
# -------------------------
if __name__ == "__main__":

    train_vids = ["vid2",'vid4']
    samples = load_samples(train_vids)

    # print("Unique labels:", set(s["label"] for s in samples))
    # input('stop')

    # build dataset
    train_dataset = build_tf_dataset(samples, batch_size=4, num_workers=32)

    # input(len(train_dataset))
    # count = 0
    # for _ in train_dataset: 
    #     count += 1
    #     if(count == 500):break
    # input(count)
    # build backbone
    vit = VisionTransformer(
        input_specs=layers.InputSpec(shape=[None, 432, 768, 3]),
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        mlp_dim=3072,
    )

    # retriever
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name="ragdb_p32_embeddings",
        metadata={"hnsw:space": "l2"}
    )
    retriever = FrameRetriever(collection, top_k=50, search_k=100)

    # rag head
    rag_head = RAGHead(hidden_size=768, num_queries=4)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # training loop
    for frames_batch, metadata_batch, labels_batch in train_dataset:
        loss = train_step(
            vit, rag_head, retriever,
            optimizer, bce,
            frames_batch, metadata_batch, labels_batch
        )
        print("loss:", float(loss))