import os
import pandas as pd
import numpy as np
import tensorflow as tf, tf_keras
from chromadb import PersistentClient

from models.vit_backbone import VisionTransformer
from models.rag_head import RAGHead
from retrieval.frame_retriever import FrameRetriever
from dataset import build_tf_dataset
import random

layers = tf_keras.layers
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# usage: python -m train.training

def compute_accuracy(labels, logits):
    preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    return tf.reduce_mean(tf.cast(preds == labels, tf.float32))


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
        retrieved_embeddings = retriever(cls_embeddings, metadata)

        # print(np.mean(np.var(retrieved_embeddings, axis=1))) variance is around 0.05
        # input('stop')
        logits, _ = rag_head(cls_embeddings, retrieved_embeddings, training=True)
        loss = loss_fn(labels, logits)

    train_vars = vit.trainable_variables + rag_head.trainable_variables
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))

    acc = compute_accuracy(labels, logits)
    print("loss:", loss.numpy(), "acc:", acc.numpy())

    return loss

def evaluate(val_ds, vit, rag_head, retriever):
    losses = []
    accs = []

    # input(val_ds)
    for frames, metadata, labels in val_ds:
        vit_out = vit(frames, training=False)
        cls = vit_out["pre_logits"]
        retrieved = retriever(cls, metadata)

        logits, _ = rag_head(cls, retrieved, training=False)

        loss = bce(labels, logits)
        # preds = (tf.sigmoid(logits) > 0.5)
        acc = compute_accuracy(labels,logits)

        losses.append(loss.numpy())
        accs.append(acc.numpy())
    
    print("validation loss:", np.mean(losses), "validation acc:", np.mean(accs))
    # return np.mean(losses), np.mean(accs)

# -------------------------
# 3. Main
# -------------------------
if __name__ == "__main__":

    train_vids = ["vid2",'vid4']
    samples = load_samples(train_vids)
    # input(np.array(samples))
    random.shuffle(samples)

    train_samples = samples[0:600]
    validation_samples = samples[600:700]

    # input(np.array(samples))
    # samples = 

    # print("Unique labels:", set(s["label"] for s in samples))
    # input('stop')

    # build dataset
    # train_dataset = build_tf_dataset(samples, batch_size=4, num_workers=32)
    train_dataset = build_tf_dataset(train_samples,batch_size=4, num_workers=24)
    valid_dataset = build_tf_dataset(validation_samples,batch_size=4, num_workers=8)
    
    # print(train_dataset)
    # print(valid_dataset)
    # input('stop')
    # input(len(train_dataset))
    # input(len(samples)) --> is 50100 (only vid2 and vid4)
    # count = 0
    # for _ in train_dataset: 
    #     count += 1
    #     print(count)
    # #     if(count == 500):break
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

    acc_count = 0
    acc_frames = []
    acc_metadata = {"vid": [], "clip": [], "side": [], "t_norm": []}
    acc_labels = []

    valid_counter = 0
    # training loop
    for frames_batch, metadata_batch, labels_batch in train_dataset:
        
        # print(frames_batch) shape=(4, 432, 768, 3)
        # print(metadata_batch)
        # print(labels_batch)
        # input('stop')

# {'vid': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([2, 2, 2, 2], dtype=int32)>, 
#  'clip': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([100, 100, 100, 100], dtype=int32)>, 
#  'side': <tf.Tensor: shape=(4,), dtype=string, numpy=array([b'left', b'left', b'left', b'left'], dtype=object)>, 
#  't_norm': <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.0020284 , 0.0040568 , 0.00608519, 0.00811359], dtype=float32)>}

        acc_frames.append(frames_batch)
        acc_metadata["vid"].append(metadata_batch["vid"])
        acc_metadata["clip"].append(metadata_batch["clip"])
        acc_metadata["side"].append(metadata_batch["side"])
        acc_metadata["t_norm"].append(metadata_batch["t_norm"])
        acc_labels.append(labels_batch)

        acc_count += 1
        valid_counter += 1

        if(acc_count == 5):
            big_frames = tf.concat(acc_frames, axis=0)

            big_metadata = {
                "vid": tf.concat(acc_metadata["vid"], axis=0),
                "clip": tf.concat(acc_metadata["clip"], axis=0),
                "side": tf.concat(acc_metadata["side"], axis=0),
                "t_norm": tf.concat(acc_metadata["t_norm"], axis=0)
            }

            big_labels = tf.concat(acc_labels, axis=0)

            loss = train_step(
                vit, rag_head, retriever,
                optimizer, bce,
                big_frames, big_metadata, big_labels
            )

            # if(valid_counter == 1):
            evaluate(valid_dataset,vit,rag_head,retriever)
                # valid_counter = 0
            # print("loss:", float(loss))

            acc_frames = []
            acc_metadata = {"vid": [], "clip": [], "side": [], "t_norm": []}
            acc_labels = []
            acc_count = 0

            # for v_frames, v_metadata, v_labels in valid_dataset
        
        
            # print(big_metadata)
            # input('stop')

        # loss = train_step(
        #     vit, rag_head, retriever,
        #     optimizer, bce,
        #     frames_batch, metadata_batch, labels_batch
        # )
        # print("loss:", float(loss))


# code for checkpointing: 
#         ckpt = tf.train.Checkpoint(
#     vit=vit,
#     rag_head=rag_head,
#     optimizer=optimizer
# )
# manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=5)

# if batch_i % 10 == 0:
#     manager.save()

#     ckpt.restore(manager.latest_checkpoint)