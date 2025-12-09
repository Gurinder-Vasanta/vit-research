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
from db_maintainence.db_rebuild import rebuild_db

from transformers import ViTModel, ViTImageProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_processor.do_rescale = False
hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
hf_vit.eval()

layers = tf_keras.layers
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

np.random.seed(1234)

def hf_vit_embed_batch(frames_np):
    """
    frames_np: (N, 432, 768, 3) uint8 or float32
    Returns (N, 768) numpy embeddings (L2 normalized)
    """
    frames_np = frames_np.astype(np.float32)
    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    with torch.no_grad():
        inputs = hf_processor(images=frames_list, return_tensors="pt").to(device)
        out = hf_vit(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # (N,768)
        cls = cls.cpu().numpy()
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        return cls.astype(np.float32)
    
def simple_retrieval_contrastive_loss(q, retrieved):
    """
    q:          (B, 768) projections of chunk embeddings
    retrieved:  (B, K, 768) projections of retrieved embeddings
    """

    B = tf.shape(q)[0]

    # mean retrieved embedding for each example
    r_mean = tf.reduce_mean(retrieved, axis=1)  # (B, 768)

    # positive pull = 1 - cosine(q_i, r_mean_i)
    pos_sim = tf.reduce_sum(q * r_mean, axis=-1) #/ 0.1
    pull = 1.0 - pos_sim   # (B,)

    # negative push:
    # shift r_mean by 1 position to create "other" neighborhoods
    r_other = tf.roll(r_mean, shift=1, axis=0)

    neg_sim = tf.reduce_sum(q * r_other, axis=-1) #/ 0.1
    push = neg_sim         # (B,)

    # final loss: pull + push
    loss = pull + push
    return tf.reduce_mean(loss)

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
    def __init__(self, rag_head, proj_head, accum_steps):
        self.accum_steps = accum_steps
        self.step = 0
        self.vars = rag_head.trainable_variables + proj_head.trainable_variables
        self.gradients = [tf.zeros_like(v) for v in self.vars]

    def accumulate(self, grads):
        self.gradients = [g_old + g_new for g_old, g_new in zip(self.gradients, grads)]
        self.step += 1

    def apply(self, optimizer):
        if self.step == self.accum_steps:
            avg_grads = [g / self.accum_steps for g in self.gradients]
            optimizer.apply_gradients(zip(avg_grads, self.vars))
            self.gradients = [tf.zeros_like(v) for v in self.vars]
            self.step = 0


def train_step(rag_head, proj_head, retriever, optimizer, loss_fn,
               frames, metadata, labels, accum):

    B = tf.shape(frames)[0]
    T = tf.shape(frames)[1]

    frames_np = tf.numpy_function(
        hf_vit_embed_batch,
        [tf.reshape(frames, (-1, 432, 768, 3))],
        tf.float32
    )
    frame_embs = tf.reshape(frames_np, (B, T, 768))

    # # ----- ViT forward -----
    # frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))
    # vit_out = vit(frames_reshaped, training=False)
    # frame_embs_flat = vit_out["pre_logits"]
    # frame_embs_flat = tf.nn.l2_normalize(frame_embs_flat, axis=1)
    # frame_embs = tf.reshape(frame_embs_flat, (B, T, 768))

    # ----- Raw chunk pool -----
    raw_chunk = tf.reduce_mean(frame_embs, axis=1)
    raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=1) 
    # raw_chunk = tf.stop_gradient(raw_chunk)

    # ----- Forward projection (learnable) -----
    with tf.GradientTape() as tape:
        chunk_embs = proj_head(raw_chunk, training=True)
        chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)

        # ----- Retrieval (stop gradient) -----
        retrieved_np = retriever(chunk_embs, metadata)
        retrieved = tf.nn.l2_normalize(
            tf.stop_gradient(tf.convert_to_tensor(retrieved_np, tf.float32)),
            axis=2
        )


        logits, _ = rag_head(chunk_embs, retrieved, training=True)
        loss_cls = loss_fn(labels, logits)
        # loss = loss_fn(labels, logits)
        # ----- NEW: simple retrieval-aware contrast -----
        loss_contrast = simple_retrieval_contrastive_loss(chunk_embs, retrieved)

        # combine them
        loss = loss_cls + 0.1 * loss_contrast        # λ = 0.1 to start

    # Get grads for BOTH heads
    grads = tape.gradient(loss,
            rag_head.trainable_variables + proj_head.trainable_variables)

    # optimizer.apply_gradients(zip(grads, rag_head.trainable_variables + proj_head.trainable_variables))
    accum.accumulate(grads)
    accum.apply(optimizer)

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
def evaluate(val_ds, rag_head, proj_head, retriever, loss_fn):
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

        frames_np = tf.numpy_function(
            hf_vit_embed_batch,
            [tf.reshape(frames, (-1, 432, 768, 3))],
            tf.float32
        )
        frame_embs = tf.reshape(frames_np, (B, T, 768))

        # # reshape so ViT sees individual frames
        # frames_reshaped = tf.reshape(frames, (-1, 432, 768, 3))  # (B*T, 432,768,3)
        # # input(frames_reshaped)
       
        # vit_out = vit(frames_reshaped, training=False)
        # frame_embs_flat = vit_out["pre_logits"]   # (B*T, 768)
        # frame_embs_flat = tf.nn.l2_normalize(frame_embs_flat, axis=1)

        # # reshape back
        # frame_embs = tf.reshape(frame_embs_flat, (B, T, 768))  # (B,60,768)

        # ---- chunk pooling ----
        # chunk_embs = tf.reduce_max(frame_embs, axis=1)  # (B, 768) #TODO: reduce to max
        raw_chunk = tf.reduce_mean(frame_embs, axis=1)
        raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=-1)
        chunk_embs = proj_head(raw_chunk, training=False)
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
        print("Combined true cosine similarity:", float(cos_comb))

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
    samples = load_samples(train_vids,stride=1)
    chunk_samples = build_chunks(samples, chunk_size=12)

    random.shuffle(chunk_samples)

    # split 95/5
    n = len(chunk_samples)
    print(n)
    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    train_chunks = chunk_samples[0:500] #was 128
    val_chunks = chunk_samples[600:625]
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
    # vit = VisionTransformer(
    #     input_specs=layers.InputSpec(shape=[None, 432, 768, 3]),
    #     patch_size=32,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_size=768,
    #     mlp_dim=3072,
    # )

    # # freeze ViT
    # vit.trainable = False

    # RAG head (trainable)
    rag_head = RAGHead(hidden_size=768, num_queries=4)

    proj_head = ProjectionHead(input_dim=768, hidden_dim=768*4, proj_dim=768)

    # proj_head.load_weights("projection_head.weights.h5")
    
    # Retrieval DB
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name="ragdb_p32_rich_embeddings",
        metadata={"hnsw:space": "cosine"}
    )

    
    retriever = FrameRetriever(collection, top_k=25, search_k=200)

    dummy_chunk = tf.zeros((1, 768))
    dummy_retrieved = tf.zeros((1, 10, 768))
    _ = rag_head(dummy_chunk, dummy_retrieved, training=False)
    rag_head.load_weights('rag_head.weights.h5')

    dummy = tf.zeros((1, 768), dtype=tf.float32)
    _ = proj_head(dummy)   # builds variables
    proj_head.load_weights("projection_head.weights.h5")

    # ---------------------------------------------
    # 4. Optimizer / loss
    # ---------------------------------------------
    optimizer = tf.keras.optimizers.Adam(1e-5)
    # optimizer = torch.optim.AdamW([
    #     {"params": hf_vit.encoder.layer[-1].parameters(), "lr": 1e-6},
    #     {"params": hf_vit.layernorm.parameters(), "lr": 1e-6},
    #     {"params": rag_head.parameters(), "lr": 1e-5},
    #     {"params": projector.parameters(), "lr": 1e-5},
    # ])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # ---------------------------------------------
    # 5. Training loop
    # ---------------------------------------------
    EPOCHS = 48

    # lower lr to 1e-5 and make contrastive coefficient 0.1 at epoch 24 (rebuild every 6)
    accum_steps = 16  # effective batch = physical batch * accum_steps
    accum = Accumulator(rag_head, proj_head, accum_steps)

    for epoch in range(EPOCHS):
        if(epoch+1 < 25):
            continue
        print(f"\n================= EPOCH {epoch+1} =================")
        print('collection count in training ')
        print(collection.count())
        losses = []
        accs = []
        batch_counter = 0
        for frames_batch, metadata_batch, labels_batch in train_dataset:
            curloss, curacc = train_step(
                rag_head, proj_head, retriever,
                optimizer, bce,
                frames_batch, metadata_batch, labels_batch,
                accum
            )

            batch_counter += 1
            losses.append(curloss)
            accs.append(curacc)
            if(batch_counter % 10 == 0):
                print(f"EPOCH {epoch+1} BATCH {batch_counter} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch+1} TRAIN acc: {np.mean(accs):.4f}")
        print(f"EPOCH {epoch+1} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch+1} TRAIN acc: {np.mean(accs):.4f}")
        proj_head.save_weights("projection_head.weights.h5")
        rag_head.save_weights('rag_head.weights.h5')
        # validation at end of every epoch
        evaluate(val_dataset, rag_head, proj_head, retriever, bce)
        # rebuild_db()
        if((epoch + 1) % 6 == 0 and (epoch+1) >= 6): 
            rebuild_db()
            

