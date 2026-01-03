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
import config

from transformers import ViTModel, ViTImageProcessor
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

# another idea: 
# we can prolly use the manually labelled intervals 
# you can actually have it so that the ragdb collection 
# has all of the manual interval clips (most likely add a piece of metadata called 'manual')
# and then you can throw in the auto labelled ones as well (with mdata 'auto')
# this allows you to: 
# a: have a solid number of clips from each of the videos (even clips from vid5 can be used)
# b: actually use the manually labelled clips for more than just autolabelling 
# c: allow you to keep a set of embeddings that you know are correct so that 
# maybe when you rebuild, you only rebuild things with the tag of 'auto' (or something like that)

# usage: python -m train.training
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_processor.do_rescale = False
hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
hf_vit.eval()

layers = tf_keras.layers


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
# def compute_accuracy(labels, logits):
#     preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
#     return tf.reduce_mean(tf.cast(preds == labels, tf.float32))

def compute_accuracy(labels, logits):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)   # (B,)
    logits = tf.reshape(logits, [-1])                      # (B,)
    preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    # print('raw sigmoid ')
    # print(tf.sigmoid(logits))
    # print('preds')
    # print(preds)
    return tf.reduce_mean(tf.cast(preds == labels, tf.float32))

def find_best_f1(labels, probs):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_f1, best_t



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
               frames, metadata, labels, accum,contrastive_coefficient):

    B = tf.shape(frames)[0]
    T = tf.shape(frames)[1]

    frames_np = tf.numpy_function(
        hf_vit_embed_batch,
        [tf.reshape(frames, (-1, 432, 768, 3))],
        tf.float32
    )
    frame_embs = tf.reshape(frames_np, (B, T, 768))

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
        loss = loss_cls + contrastive_coefficient * loss_contrast        # λ = 0.1 to start

    # Get grads for BOTH heads
    grads = tape.gradient(loss,
            rag_head.trainable_variables + proj_head.trainable_variables)

    # optimizer.apply_gradients(zip(grads, rag_head.trainable_variables + proj_head.trainable_variables))
    accum.accumulate(grads)
    accum.apply(optimizer)

    # print(labels)
    # print(logits)
    # input('stop')
    acc = compute_accuracy(labels, logits)
    # true_acc = compute_true_accuracy(labels, logits)
    # print(acc)
    # print(true_acc)
    # input('stop')
    # print("loss:", float(loss), "acc:", float(acc))
    return float(loss), float(acc)

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
def evaluate(val_ds, rag_head, proj_head, retriever, loss_fn):
    losses = []
    accs = []

    comb_sims = []
    retr_sims = []
    
    c1s = []
    c2s = []

    all_val_logits = []
    all_val_labels = []

    for frames, metadata, labels in val_ds:

        B = tf.shape(frames)[0]
        T = tf.shape(frames)[1]  # 60

        # input(frames)

        frames_np = tf.numpy_function(
            hf_vit_embed_batch,
            [tf.reshape(frames, (-1, 432, 768, 3))],
            tf.float32
        )
        frame_embs = tf.reshape(frames_np, (B, T, 768))

        # ---- chunk pooling ----
        # chunk_embs = tf.reduce_max(frame_embs, axis=1)  # (B, 768) #TODO: reduce to max
        raw_chunk = tf.reduce_mean(frame_embs, axis=1)
        raw_chunk = tf.nn.l2_normalize(raw_chunk, axis=-1)
        chunk_embs = proj_head(raw_chunk, training=False)
        chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)
      
        retrieved_np = retriever(chunk_embs, metadata)
        retrieved = tf.nn.l2_normalize(
            tf.convert_to_tensor(retrieved_np, dtype=tf.float32),
            axis=2
        )

        logits, _ = rag_head(chunk_embs, retrieved, training=False)

        all_val_logits.append(logits.numpy())
        all_val_labels.append(labels.numpy())

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
        retr_sims.append(cos_retr)

        # # --- combined feature similarity (optional) ---
        z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)

        # Combined similarity (REAL)
        cos_comb = -tf.keras.losses.cosine_similarity(z1, z2)
        print("Combined true cosine similarity:", float(cos_comb))
        comb_sims.append(cos_comb)

        q1 = chunk_embs[0]
        q2 = chunk_embs[1]

        cos_qr1 = -tf.reduce_mean(
            tf.keras.losses.cosine_similarity(q1[None, :], r1)
        )
        cos_qr2 = -tf.reduce_mean(
            tf.keras.losses.cosine_similarity(q2[None, :], r2)
        )

        print(f'C1 retrieval purity: {cos_qr1}, C2 retrieval purity: {cos_qr2}')
        c1s.append(cos_qr1)
        c2s.append(cos_qr2)
        # val_probs = 1 / (1 + np.exp(-logits))  # sigmoid
        # roc_auc = roc_auc_score(labels, val_probs)
        # # print("ROC-AUC:", roc_auc)

        # val_preds = (val_probs > 0.5).astype(int)
        # f1_default = f1_score(labels, val_preds)
        # # print("F1 @ 0.5 threshold:", f1_default)

        # best_f1, best_threshold = find_best_f1(labels, val_probs)
        # # print("Best F1:", best_f1)
        # # print("Best threshold:", best_threshold)

        # print(f'ROC-AUC: {roc_auc} F1 @ 0.5 threshold: {f1_default}')

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

    all_val_logits = np.concatenate(all_val_logits, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    val_probs = 1 / (1 + np.exp(-all_val_logits))  # sigmoid
    # roc_auc = roc_auc_score(all_val_labels, val_probs)
    # print("ROC-AUC:", roc_auc)

    val_preds = (val_probs > 0.5).astype(int)
    # f1_default = f1_score(all_val_labels, val_preds)
    # print("F1 @ 0.5 threshold:", f1_default)

    # best_f1, best_threshold = find_best_f1(all_val_labels, val_probs)
    # print("Best F1:", best_f1)
    # print("Best threshold:", best_threshold)

    # print(f'ROC-AUC: {roc_auc} F1 @ 0.5 threshold: {f1_default}')
    # print(f'Best F1: {best_f1} Best threshold: {best_threshold}')
    print(f"VAL loss: {np.mean(losses):.4f}, VAL acc: {np.mean(accs):.4f}")
    print(f'MEAN comb sim:  {np.mean(comb_sims):.4f}, MEAN retr sim: {np.mean(retr_sims):.4f}')
    print(f'STDEV comb sim:  {np.std(comb_sims):.4f}, STDEV retr sim: {np.std(retr_sims):.4f}')
    print(f'MEAN c1s:  {np.mean(c1s):.4f}, MEAN c2s: {np.mean(c2s):.4f}')
    print("----------------------------------------------------")


# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = config.VIDS_TO_USE
    samples = load_samples(train_vids,stride=1)
    chunk_samples = build_chunks(samples, chunk_size=12)

    random.shuffle(chunk_samples)

    # split 95/5
    n = len(chunk_samples)
    print(n)
    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    train_chunks = chunk_samples[config.START_CHUNK_TRAIN:config.END_CHUNK_TRAIN] #was 2000
    val_chunks = chunk_samples[config.START_CHUNK_VALID:config.END_CHUNK_VALID]
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")

    # ---------------------------------------------
    # 2. Build TF datasets
    # ---------------------------------------------
    train_dataset = build_tf_dataset_chunks(train_chunks, batch_size=config.CHUNK_BATCH_SIZE)
    val_dataset   = build_tf_dataset_chunks(val_chunks,   batch_size=config.CHUNK_BATCH_SIZE)

    # ---------------------------------------------
    # 3. Build models
    # ---------------------------------------------
    
    # RAG head (trainable)
    rag_head = RAGHead(hidden_size=768, num_queries=config.NUM_QUERIES, num_layers=config.NUM_LAYERS,num_heads=config.NUM_HEADS)

    proj_head = ProjectionHead(input_dim=768, hidden_dim=768*8, proj_dim=768)

    # proj_head.load_weights("projection_head.weights.h5")
    
    # Retrieval DB
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(
        name=config.CHROMADB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    
    retriever = FrameRetriever(collection, top_k=config.TOP_K, search_k=config.SEARCH_K)

    dummy_chunk = tf.zeros((1, 768))
    dummy_retrieved = tf.zeros((1, 10, 768))
    _ = rag_head(dummy_chunk, dummy_retrieved, training=False)
    # rag_head.save_weights(config.RAG_WEIGHTS)
    # rag_head.load_weights('rag_head_5vid_new_v2.weights.h5')

    dummy = tf.zeros((1, 768), dtype=tf.float32)
    _ = proj_head(dummy)   # builds variables
    # proj_head.save_weights(config.PROJ_WEIGHTS)
    # proj_head.load_weights("projection_head_5vid_new_v2.weights.h5")

    # ---------------------------------------------
    # 4. Optimizer / loss
    # ---------------------------------------------
    optimizer = tf.keras.optimizers.Adam(config.PHASE_1_LEARNING_RATE)
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
    EPOCHS = config.EPOCHS

    # lower lr to 1e-5 and make contrastive coefficient 0.1 at epoch 24 (rebuild every 6)
    # stopped at epoch 39 (so if you were to start this run again, start it at epoch 40)
    # results look great, val loss (old treshold > 0.5 loss) is 0.2171, train loss is 0.2649, train acc is 0.75
    accum_steps = config.ACCUM_BATCH_SIZE  # effective batch = physical batch * accum_steps
    accum = Accumulator(rag_head, proj_head, accum_steps)

    for epoch in range(1,EPOCHS+1):
        # if(epoch < 5):
        #     continue
        print(f"\n================= EPOCH {epoch} =================")
        print('collection count in training ')
        print(collection.count())
        losses = []
        accs = []
        batch_counter = 0
        
        if(epoch >= int(EPOCHS/2)+1): 
            optimizer.learning_rate.assign(config.PHASE_2_LEARNING_RATE)
            contrastive_coefficient = config.PHASE_2_CONTRASTIVE_LOSS
        else: 
            optimizer.learning_rate.assign(config.PHASE_1_LEARNING_RATE)
            contrastive_coefficient = config.PHASE_1_CONTRASTIVE_LOSS

        for frames_batch, metadata_batch, labels_batch in train_dataset:
            curloss, curacc = train_step(
                rag_head, proj_head, retriever,
                optimizer, bce,
                frames_batch, metadata_batch, labels_batch,
                accum,contrastive_coefficient
            )

            batch_counter += 1
            losses.append(curloss)
            accs.append(curacc)
            if(batch_counter % config.PRINT_EVERY == 0):
                print(f"EPOCH {epoch} BATCH {batch_counter} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f}")
        print(f"EPOCH {epoch} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f}")
        proj_head.save_weights(config.PROJ_WEIGHTS)
        rag_head.save_weights(config.RAG_WEIGHTS)
        # validation at end of every epoch
        evaluate(val_dataset, rag_head, proj_head, retriever, bce)
        # rebuild_db()
        if((epoch) % config.REBUILD_EVERY == 0 and (epoch) >= config.REBUILD_EVERY): 
            rebuild_db()
# epoch 1
# VAL loss: 0.5056, VAL acc: 0.7833
# MEAN comb sim:  0.9192, MEAN retr sim: 0.8704
# STDEV comb sim:  0.0236, STDEV retr sim: 0.0118

# epoch 2
# EPOCH 2 TRAIN loss: 0.7300, EPOCH 2 TRAIN acc: 0.6420
# VAL loss: 0.4981, VAL acc: 0.7333
# MEAN comb sim:  0.9031, MEAN retr sim: 0.8661
# STDEV comb sim:  0.0272, STDEV retr sim: 0.0138

# epoch 3
# EPOCH 3 TRAIN loss: 0.6700, EPOCH 3 TRAIN acc: 0.6560
# VAL loss: 0.4388, VAL acc: 0.7500
# MEAN comb sim:  0.8805, MEAN retr sim: 0.8610
# STDEV comb sim:  0.0314, STDEV retr sim: 0.0139

# epoch 4
# EPOCH 4 TRAIN loss: 0.6168, EPOCH 4 TRAIN acc: 0.6685
# VAL loss: 0.4173, VAL acc: 0.6833
# MEAN comb sim:  0.8418, MEAN retr sim: 0.8578
# STDEV comb sim:  0.0357, STDEV retr sim: 0.0174

# -- rebuild --
# epoch 5
# EPOCH 5 TRAIN loss: 0.6000, EPOCH 5 TRAIN acc: 0.6640
# VAL loss: 0.3526, VAL acc: 0.7833
# MEAN comb sim:  0.8209, MEAN retr sim: 0.8672
# STDEV comb sim:  0.0470, STDEV retr sim: 0.0127
   
# epoch 6
# EPOCH 6 TRAIN loss: 0.5397, EPOCH 6 TRAIN acc: 0.6860
# VAL loss: 0.4481, VAL acc: 0.6333
# MEAN comb sim:  0.7879, MEAN retr sim: 0.8691
# STDEV comb sim:  0.0523, STDEV retr sim: 0.0143

# epoch 7
# EPOCH 7 TRAIN loss: 0.4603, EPOCH 7 TRAIN acc: 0.6925
# VAL loss: 0.3258, VAL acc: 0.7833
# MEAN comb sim:  0.7655, MEAN retr sim: 0.8632
# STDEV comb sim:  0.0512, STDEV retr sim: 0.0137

# epoch 8
# EPOCH 8 TRAIN loss: 0.4164, EPOCH 8 TRAIN acc: 0.7130
# VAL loss: 0.3739, VAL acc: 0.6500
# MEAN comb sim:  0.7353, MEAN retr sim: 0.8631
# STDEV comb sim:  0.0496, STDEV retr sim: 0.0189

# -- rebuild
# epoch 9
# EPOCH 9 TRAIN loss: 0.3547, EPOCH 9 TRAIN acc: 0.7145
# VAL loss: 0.3561, VAL acc: 0.7000
# MEAN comb sim:  0.7328, MEAN retr sim: 0.8619
# STDEV comb sim:  0.0616, STDEV retr sim: 0.0160

# epoch 10
# EPOCH 10 TRAIN loss: 0.3108, EPOCH 10 TRAIN acc: 0.7420
# VAL loss: 0.3930, VAL acc: 0.7000
# MEAN comb sim:  0.7244, MEAN retr sim: 0.8587
# STDEV comb sim:  0.0441, STDEV retr sim: 0.0160

# epoch 11
# EPOCH 11 TRAIN loss: 0.2703, EPOCH 11 TRAIN acc: 0.7490
# VAL loss: 0.3444, VAL acc: 0.7333
# MEAN comb sim:  0.7119, MEAN retr sim: 0.8608
# STDEV comb sim:  0.0614, STDEV retr sim: 0.0125

# epoch 12
# EPOCH 12 TRAIN loss: 0.2409, EPOCH 12 TRAIN acc: 0.7700
# VAL loss: 0.2713, VAL acc: 0.7833
# MEAN comb sim:  0.6946, MEAN retr sim: 0.8619
# STDEV comb sim:  0.0575, STDEV retr sim: 0.0185

# -- stopped so had to restart (so maybe just let it do 24 full epochs instead of 12)

# epoch 13
# EPOCH 13 TRAIN loss: 0.6352, EPOCH 13 TRAIN acc: 0.6615
# VAL loss: 0.3631, VAL acc: 0.7667
# MEAN comb sim:  0.7522, MEAN retr sim: 0.8507
# STDEV comb sim:  0.0560, STDEV retr sim: 0.0141

# epoch 14
# EPOCH 14 TRAIN loss: 0.4770, EPOCH 14 TRAIN acc: 0.7085
# VAL loss: 0.3472, VAL acc: 0.7333
# MEAN comb sim:  0.7486, MEAN retr sim: 0.8471
# STDEV comb sim:  0.0635, STDEV retr sim: 0.0215

# epoch 15
# EPOCH 15 TRAIN loss: 0.3908, EPOCH 15 TRAIN acc: 0.7260
# VAL loss: 0.2857, VAL acc: 0.7500
# MEAN comb sim:  0.7496, MEAN retr sim: 0.8429
# STDEV comb sim:  0.0646, STDEV retr sim: 0.0245

# epoch 16
# EPOCH 16 TRAIN loss: 0.3259, EPOCH 16 TRAIN acc: 0.7405
# VAL loss: 0.2711, VAL acc: 0.7833
# MEAN comb sim:  0.7320, MEAN retr sim: 0.8384
# STDEV comb sim:  0.0545, STDEV retr sim: 0.0201

# -- rebuild
# epoch 17
# EPOCH 17 TRAIN loss: 0.3797, EPOCH 17 TRAIN acc: 0.7215
# VAL loss: 0.3659, VAL acc: 0.7167
# MEAN comb sim:  0.7241, MEAN retr sim: 0.8201
# STDEV comb sim:  0.0561, STDEV retr sim: 0.0258

# full 24 epoch run: 
# epoch 1
# EPOCH 1 TRAIN loss: 0.8210, EPOCH 1 TRAIN acc: 0.6075
# VAL loss: 0.4956, VAL acc: 0.8000
# MEAN comb sim:  0.9287, MEAN retr sim: 0.8286
# STDEV comb sim:  0.0262, STDEV retr sim: 0.0207
# MEAN c1s:  0.0038, MEAN c2s: 0.0100

# epoch 2
# EPOCH 2 TRAIN loss: 0.7263, EPOCH 2 TRAIN acc: 0.6330
# VAL loss: 0.4748, VAL acc: 0.8000
# MEAN comb sim:  0.9229, MEAN retr sim: 0.8288
# STDEV comb sim:  0.0299, STDEV retr sim: 0.0240
# MEAN c1s:  -0.0042, MEAN c2s: -0.0040

# epoch 3
# EPOCH 3 TRAIN loss: 0.7018, EPOCH 3 TRAIN acc: 0.6295
# VAL loss: 0.4571, VAL acc: 0.7833
# MEAN comb sim:  0.8982, MEAN retr sim: 0.8142
# STDEV comb sim:  0.0280, STDEV retr sim: 0.0203
# MEAN c1s:  -0.0223, MEAN c2s: -0.0268

# epoch 4
# EPOCH 4 TRAIN loss: 0.6459, EPOCH 4 TRAIN acc: 0.6470
# VAL loss: 0.4283, VAL acc: 0.7500
# MEAN comb sim:  0.8726, MEAN retr sim: 0.8073
# STDEV comb sim:  0.0365, STDEV retr sim: 0.0173
# MEAN c1s:  -0.0239, MEAN c2s: -0.0299

# --rebuild
# (had to start again, so new train/valid data)

# epoch 5
# EPOCH 5 TRAIN loss: 0.7675, EPOCH 5 TRAIN acc: 0.6110
# VAL loss: 0.6590, VAL acc: 0.5167
# MEAN comb sim:  0.9346, MEAN retr sim: 0.8688
# STDEV comb sim:  0.0232, STDEV retr sim: 0.0148
# MEAN c1s:  0.0217, MEAN c2s: 0.0212

# epoch 6
# EPOCH 6 TRAIN loss: 0.7310, EPOCH 6 TRAIN acc: 0.6215
# VAL loss: 0.5388, VAL acc: 0.7000
# MEAN comb sim:  0.9224, MEAN retr sim: 0.8665
# STDEV comb sim:  0.0253, STDEV retr sim: 0.0093
# MEAN c1s:  0.0102, MEAN c2s: 0.0103

# epoch 7
# EPOCH 7 TRAIN loss: 0.6680, EPOCH 7 TRAIN acc: 0.6435
# VAL loss: 0.4816, VAL acc: 0.7667
# MEAN comb sim:  0.9104, MEAN retr sim: 0.8675
# STDEV comb sim:  0.0274, STDEV retr sim: 0.0095
# MEAN c1s:  -0.0007, MEAN c2s: 0.0050

# epoch 8
# EPOCH 8 TRAIN loss: 0.6292, EPOCH 8 TRAIN acc: 0.6610
# VAL loss: 0.4634, VAL acc: 0.6167
# MEAN comb sim:  0.8994, MEAN retr sim: 0.8702
# STDEV comb sim:  0.0279, STDEV retr sim: 0.0085
# MEAN c1s:  -0.0075, MEAN c2s: -0.0054

# --rebuild

# epoch 9
# EPOCH 9 TRAIN loss: 0.6716, EPOCH 9 TRAIN acc: 0.6460
# VAL loss: 0.4870, VAL acc: 0.6833
# MEAN comb sim:  0.8768, MEAN retr sim: 0.8632
# STDEV comb sim:  0.0361, STDEV retr sim: 0.0134
# MEAN c1s:  0.0096, MEAN c2s: 0.0074

# epoch 10
# EPOCH 10 TRAIN loss: 0.5865, EPOCH 10 TRAIN acc: 0.6565
# VAL loss: 0.4515, VAL acc: 0.6333
# MEAN comb sim:  0.8375, MEAN retr sim: 0.8573
# STDEV comb sim:  0.0479, STDEV retr sim: 0.0147
# MEAN c1s:  -0.0199, MEAN c2s: -0.0183

# epoch 11
# EPOCH 11 TRAIN loss: 0.5016, EPOCH 11 TRAIN acc: 0.6890
# VAL loss: 0.4637, VAL acc: 0.6500
# MEAN comb sim:  0.8102, MEAN retr sim: 0.8558
# STDEV comb sim:  0.0533, STDEV retr sim: 0.0136
# MEAN c1s:  -0.0342, MEAN c2s: -0.0306

# epoch 12
# EPOCH 12 TRAIN loss: 0.4894, EPOCH 12 TRAIN acc: 0.6895
# VAL loss: 0.3384, VAL acc: 0.7000
# MEAN comb sim:  0.7814, MEAN retr sim: 0.8523
# STDEV comb sim:  0.0585, STDEV retr sim: 0.0133
# MEAN c1s:  -0.0337, MEAN c2s: -0.0342

# --rebuild 
# epoch 13
# EPOCH 13 TRAIN loss: 0.4275, EPOCH 13 TRAIN acc: 0.7005
# VAL loss: 0.2573, VAL acc: 0.7833
# MEAN comb sim:  0.7473, MEAN retr sim: 0.8566
# STDEV comb sim:  0.0559, STDEV retr sim: 0.0155
# MEAN c1s:  -0.0356, MEAN c2s: -0.0276

# epoch 14
# EPOCH 14 TRAIN loss: 0.3749, EPOCH 14 TRAIN acc: 0.7115
# VAL loss: 0.2982, VAL acc: 0.8000
# MEAN comb sim:  0.7438, MEAN retr sim: 0.8586
# STDEV comb sim:  0.0519, STDEV retr sim: 0.0120
# MEAN c1s:  -0.0377, MEAN c2s: -0.0388

# epoch 15
# EPOCH 15 TRAIN loss: 0.3383, EPOCH 15 TRAIN acc: 0.7300
# VAL loss: 0.3032, VAL acc: 0.6333
# MEAN comb sim:  0.7350, MEAN retr sim: 0.8601
# STDEV comb sim:  0.0648, STDEV retr sim: 0.0158
# MEAN c1s:  -0.0615, MEAN c2s: -0.0445

# epoch 16
# EPOCH 16 TRAIN loss: 0.2993, EPOCH 16 TRAIN acc: 0.7410
# VAL loss: 0.2366, VAL acc: 0.7333
# MEAN comb sim:  0.7214, MEAN retr sim: 0.8571
# STDEV comb sim:  0.0595, STDEV retr sim: 0.0137
# MEAN c1s:  -0.0475, MEAN c2s: -0.0451

# --rebuild
# epoch 17
# EPOCH 17 TRAIN loss: 0.2648, EPOCH 17 TRAIN acc: 0.7495
# VAL loss: 0.1846, VAL acc: 0.7667
# MEAN comb sim:  0.7112, MEAN retr sim: 0.8580
# STDEV comb sim:  0.0643, STDEV retr sim: 0.0131
# MEAN c1s:  -0.0544, MEAN c2s: -0.0502

# epoch 18
# EPOCH 18 TRAIN loss: 0.2234, EPOCH 18 TRAIN acc: 0.7600
# VAL loss: 0.3642, VAL acc: 0.7000
# MEAN comb sim:  0.6821, MEAN retr sim: 0.8563
# STDEV comb sim:  0.0500, STDEV retr sim: 0.0133
# MEAN c1s:  -0.0601, MEAN c2s: -0.0604

# epoch 19
# EPOCH 19 TRAIN loss: 0.2033, EPOCH 19 TRAIN acc: 0.7510
# VAL loss: 0.3097, VAL acc: 0.6667
# MEAN comb sim:  0.6865, MEAN retr sim: 0.8570
# STDEV comb sim:  0.0568, STDEV retr sim: 0.0149
# MEAN c1s:  -0.0448, MEAN c2s: -0.0495

# epoch 20
# EPOCH 20 TRAIN loss: 0.1994, EPOCH 20 TRAIN acc: 0.7590
# VAL loss: 0.9180, VAL acc: 0.6000
# MEAN comb sim:  0.6830, MEAN retr sim: 0.8621
# STDEV comb sim:  0.0776, STDEV retr sim: 0.0156
# MEAN c1s:  -0.0550, MEAN c2s: -0.0560

# --rebuild
# epoch 21
# EPOCH 21 TRAIN loss: 0.2256, EPOCH 21 TRAIN acc: 0.7445
# VAL loss: 0.4926, VAL acc: 0.7333
# MEAN comb sim:  0.6687, MEAN retr sim: 0.8539
# STDEV comb sim:  0.0573, STDEV retr sim: 0.0177
# MEAN c1s:  -0.0465, MEAN c2s: -0.0557

# epoch 22
# EPOCH 22 TRAIN loss: 0.1634, EPOCH 22 TRAIN acc: 0.7665
# VAL loss: 0.2174, VAL acc: 0.7333
# MEAN comb sim:  0.6854, MEAN retr sim: 0.8566
# STDEV comb sim:  0.0799, STDEV retr sim: 0.0209
# MEAN c1s:  -0.0528, MEAN c2s: -0.0605

# EPOCH 23 TRAIN loss: 0.1468, EPOCH 23 TRAIN acc: 0.7705
# VAL loss: 0.3027, VAL acc: 0.7333
# MEAN comb sim:  0.6610, MEAN retr sim: 0.8541
# STDEV comb sim:  0.0600, STDEV retr sim: 0.0148
# MEAN c1s:  -0.0533, MEAN c2s: -0.0527

# EPOCH 24 TRAIN loss: 0.1380, EPOCH 24 TRAIN acc: 0.7710
# VAL loss: 0.3102, VAL acc: 0.6333
# MEAN comb sim:  0.6641, MEAN retr sim: 0.8570
# STDEV comb sim:  0.0556, STDEV retr sim: 0.0155
# MEAN c1s:  -0.0592, MEAN c2s: -0.0622
