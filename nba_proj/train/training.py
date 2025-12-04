import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from chromadb import PersistentClient

from dataset import load_samples, build_chunks, build_tf_dataset_chunks
from models.vit_backbone import VisionTransformer
from models.rag_head import RAGHead
from models.projection_head import ProjectionHead
from retrieval.frame_retriever import FrameRetriever

layers = tf_keras.layers
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"


# ---------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------
def compute_accuracy(labels, logits):
    preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    return tf.reduce_mean(tf.cast(preds == labels, tf.float32))

# ---------------------------------------------------------
# TRAIN STEP WITH GRADIENT ACCUMULATION
# ---------------------------------------------------------

class Accumulator:
    def __init__(self, model, accum_steps):
        self.accum_steps = accum_steps
        self.step = 0
        self.gradients = [tf.zeros_like(v) for v in model.trainable_variables]

    def accumulate(self, grads):
        self.gradients = [g_old + g_new for g_old, g_new in zip(self.gradients, grads)]
        self.step += 1

    def apply(self, optimizer, model):
        if self.step == self.accum_steps:
            avg_grads = [g / self.accum_steps for g in self.gradients]
            optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))
            self.gradients = [tf.zeros_like(v) for v in model.trainable_variables]
            self.step = 0


def train_step(vit, rag_head, proj_head, retriever, optimizer, loss_fn,
               frames, metadata, labels,
               accum):
    
    B = tf.shape(frames)[0]
    T = tf.shape(frames)[1]

    # ---- VI T (frozen) ----
    frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))

    vit_out = vit(frames_reshaped, training=False)
    frame_embs_flat = vit_out["pre_logits"]

    frame_embs = tf.reshape(frame_embs_flat, (B, T, 768))

    # chunk_embs = tf.reduce_max(frame_embs, axis=1)
    # chunk_embs = tf.stop_gradient(chunk_embs)

    # retrieved_np = retriever(chunk_embs, metadata)

    # Pool raw chunk embeddings
    raw_chunk = tf.reduce_max(frame_embs, axis=1)  # (B, 768)
    raw_chunk = tf.stop_gradient(raw_chunk)

    # Learnable projection
    chunk_embs = proj_head(raw_chunk)  # (B, 768)

    # Retrieval now expects projection-space embeddings
    retrieved_np = retriever(chunk_embs, metadata)
    # retrieved_embs = tf.convert_to_tensor(retrieved_np, dtype=tf.float32)
    # retrieved_embs = tf.nn.l2_normalize(retrieved_embs, axis=2)


    retrieved = tf.convert_to_tensor(retrieved_np, dtype=tf.float32)
    retrieved = tf.stop_gradient(retrieved)

    # ---- Train only RAG head ----
    with tf.GradientTape() as tape:
        logits, _ = rag_head(chunk_embs, retrieved, training=True)
        loss = loss_fn(labels, logits)

    grads = tape.gradient(loss, rag_head.trainable_variables + proj_head.trainable_variables)

    # accumulate
    accum.accumulate(grads)

    # apply when full accumulation reached
    accum.apply(optimizer, rag_head)

    acc = compute_accuracy(labels, logits)

    # print("loss:", float(loss), "acc:", float(acc))
    return float(loss), float(acc)

# # ---------------------------------------------------------
# # TRAIN STEP (ViT frozen, RAG-head trainable)
# # ---------------------------------------------------------
# def train_step(vit, rag_head, retriever, optimizer, loss_fn,
#                frames, metadata, labels):

#     # frames: (B, 60, 432, 768, 3)
#     B = tf.shape(frames)[0]
#     T = tf.shape(frames)[1]  # 60

#     # input(frames)
#     # reshape so ViT sees individual frames
#     frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))  # (B*T, 432,768,3)
#     # input(frames_reshaped)
#     # ---- ViT forward ----
#     with tf.GradientTape() as tape_feat:
#         vit_out = vit(frames_reshaped, training=False)
#         frame_embs_flat = vit_out["pre_logits"]   # (B*T, 768)

#     # reshape back
#     frame_embs = tf.reshape(frame_embs_flat, (B, T, 768))  # (B,60,768)

#     # ---- chunk pooling ----
#     chunk_embs = tf.reduce_max(frame_embs, axis=1)  # (B, 768) #TODO: replace with max
#     chunk_embs = tf.stop_gradient(chunk_embs)

#     # input(chunk_embs)
#     # ---- retrieval ----
#     retrieved_np = retriever(chunk_embs, metadata)
#     retrieved_embs = tf.convert_to_tensor(retrieved_np, dtype=tf.float32)
#     retrieved_embs = tf.stop_gradient(retrieved_embs)

#     # ---- RAG head training ----
#     with tf.GradientTape() as tape_rag:
#         logits, _ = rag_head(chunk_embs, retrieved_embs, training=True)
#         loss = loss_fn(labels, logits)

#     grads = tape_rag.gradient(loss, rag_head.trainable_variables)
#     optimizer.apply_gradients(zip(grads, rag_head.trainable_variables))

#     # print()
#     # print('train logits')
#     # print(logits)
#     # print()
    
#     acc = compute_accuracy(labels, logits)
#     print("loss:", float(loss), "acc:", float(acc))

#     return loss,acc



# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
def evaluate(val_ds, vit, rag_head, proj_head, retriever, loss_fn):
    losses = []
    accs = []

    for frames, metadata, labels in val_ds:
        # frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))
        # vit_out = vit(frames_reshaped, training=False)
        # cls = vit_out["pre_logits"]
        # cls = tf.nn.l2_normalize(cls, axis=1)

        B = tf.shape(frames)[0]
        T = tf.shape(frames)[1]  # 60

        # input(frames)
        # reshape so ViT sees individual frames
        frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))  # (B*T, 432,768,3)
        # input(frames_reshaped)
       
        vit_out = vit(frames_reshaped, training=False)
        frame_embs_flat = vit_out["pre_logits"]   # (B*T, 768)
        frame_embs_flat = tf.nn.l2_normalize(frame_embs_flat, axis=1)

        # reshape back
        frame_embs = tf.reshape(frame_embs_flat, (B, T, 768))  # (B,60,768)

        # ---- chunk pooling ----
        # chunk_embs = tf.reduce_max(frame_embs, axis=1)  # (B, 768) #TODO: reduce to max
        raw_chunk = tf.reduce_max(frame_embs, axis=1)
        chunk_embs = proj_head(raw_chunk)
        chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)

        # print("frames:", frames.shape)
        # print("metadata['vid']:", metadata["vid"].shape)
        # print("metadata:", metadata)
        # input("pause")

        # input(metadata)        
        retrieved_np = retriever(chunk_embs, metadata)
        retrieved = tf.nn.l2_normalize(
            tf.convert_to_tensor(retrieved_np, dtype=tf.float32),
            axis=2
        )

        logits, _ = rag_head(chunk_embs, retrieved, training=False)
        # print()
        # print('eval logits')
        # print(logits)
        # print()

        # --- retrieved similarity ---
        # retrieved shape: (B, top_k, 768)
        r1 = retrieved[0]
        r2 = retrieved[1]

        # REAL cosine similarity (correct)
        cos_cls = -tf.keras.losses.cosine_similarity(chunk_embs[0], chunk_embs[1])
        print()
        # print("CLS true cosine similarity:", float(cos_cls))

        # Retrieved similarity (REAL)
        cos_retr = -tf.reduce_mean(tf.keras.losses.cosine_similarity(r1, r2))
        print("Retrieved true cosine similarity:", float(cos_retr))

        # # --- combined feature similarity (optional) ---
        z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)

        # Combined similarity (REAL)
        cos_comb = -tf.keras.losses.cosine_similarity(z1, z2)
        # print("Combined true cosine similarity:", float(cos_comb))

        # cos_cls = 1 - tf.keras.losses.cosine_similarity(chunk_embs[0], chunk_embs[1])
        # print("CLS cosine similarity:", float(cos_cls))

       
        # # average similarity over top-k retrieved vectors
        # cos_retr = tf.reduce_mean(1 - tf.keras.losses.cosine_similarity(r1, r2))
        # print("Retrieved cosine similarity:", float(cos_retr))

        # # --- combined feature similarity (optional) ---
        # z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        # z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)
        # cos_comb = 1 - tf.keras.losses.cosine_similarity(z1, z2)
        # print("Combined feature similarity:", float(cos_comb))
        # print("--------------------------------------------------")
        
        loss = loss_fn(labels, logits)
        acc = compute_accuracy(labels, logits)

        print(f"val batch loss: {loss:.4f}, val batch acc: {acc:.4f}")
        print()

        losses.append(loss.numpy())
        accs.append(acc.numpy())

    print(f"VAL loss: {np.mean(losses):.4f}, VAL acc: {np.mean(accs):.4f}")
    print("----------------------------------------------------")


# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = ["vid2", "vid4"]
    samples = load_samples(train_vids,stride=2)
    chunk_samples = build_chunks(samples, chunk_size=12)

    random.shuffle(chunk_samples)

    # split 95/5
    n = len(chunk_samples)

    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    train_chunks = chunk_samples[0:128]
    val_chunks = chunk_samples[150:161]
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")

    # ---------------------------------------------
    # 2. Build TF datasets
    # ---------------------------------------------
    train_dataset = build_tf_dataset_chunks(train_chunks, batch_size=2)
    val_dataset   = build_tf_dataset_chunks(val_chunks,   batch_size=2)

    # ---------------------------------------------
    # 3. Build models
    # ---------------------------------------------
    vit = VisionTransformer(
        input_specs=layers.InputSpec(shape=[None, 432, 768, 3]),
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        mlp_dim=3072,
    )

    # freeze ViT
    # vit.trainable = False

    # RAG head (trainable)
    rag_head = RAGHead(hidden_size=768, num_queries=4)

    proj_head = ProjectionHead(hidden_dim=768, proj_dim=768)
    # Retrieval DB
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name="ragdb_p32_rich_embeddings",
        metadata={"hnsw:space": "l2"}
    )
    retriever = FrameRetriever(collection, top_k=10, search_k=200)

    dummy_chunk = tf.zeros((1, 768))
    dummy_retrieved = tf.zeros((1, 10, 768))
    _ = rag_head(dummy_chunk, dummy_retrieved, training=False)

    # ---------------------------------------------
    # 4. Optimizer / loss
    # ---------------------------------------------
    optimizer = tf.keras.optimizers.Adam(1e-6)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # ---------------------------------------------
    # 5. Training loop
    # ---------------------------------------------
    EPOCHS = 10

    accum_steps = 8  # effective batch = physical batch * accum_steps
    accum = Accumulator(rag_head, accum_steps)

    for epoch in range(EPOCHS):
        print(f"\n================= EPOCH {epoch+1} =================")

        losses = []
        accs = []
        batch_counter = 0
        for frames_batch, metadata_batch, labels_batch in train_dataset:
            curloss, curacc = train_step(
                vit, rag_head, proj_head, retriever,
                optimizer, bce,
                frames_batch, metadata_batch, labels_batch,
                accum
            )

            batch_counter += 1
            losses.append(curloss)
            accs.append(curacc)
            if(batch_counter % 5 == 0):
                print(f"EPOCH {epoch+1} BATCH {batch_counter} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch+1} TRAIN acc: {np.mean(accs):.4f}")
        print(f"EPOCH {epoch+1} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch+1} TRAIN acc: {np.mean(accs):.4f}")

        # validation at end of every epoch
        evaluate(val_dataset, vit, rag_head, proj_head, retriever, bce)
