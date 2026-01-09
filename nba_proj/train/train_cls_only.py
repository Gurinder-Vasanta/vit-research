import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from chromadb import PersistentClient

from dataset import load_samples, build_chunks, build_tf_dataset_chunks
from models.vit_backbone import VisionTransformer
from models.rag_head import RAGHead
# from models.projection_head import ProjectionHead
# from retrieval.frame_retriever import FrameRetriever
# from db_maintainence.db_rebuild import rebuild_db
import configs_cls_only

from transformers import ViTModel, ViTImageProcessor
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

# print("TRAIN CWD:", os.getcwd())
# print("TRAIN chroma path:", os.path.abspath("./chroma_store"))
# input('stop')

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
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
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
    def __init__(self, rag_head, accum_steps):
        self.accum_steps = accum_steps
        self.step = 0
        self.vars = rag_head.trainable_variables
        self.gradients = [tf.zeros_like(v) for v in self.vars]

    def accumulate(self, grads):
        # self.gradients = [g_old + g_new for g_old, g_new in zip(self.gradients, grads)]
        new_grads = []
        for g_old, g_new in zip(self.gradients, grads):
            if g_new is None:
                new_grads.append(g_old)
            else:
                new_grads.append(g_old + g_new)
        self.gradients = new_grads
        self.step += 1

    def apply(self, optimizer):
        if self.step == self.accum_steps:
            avg_grads = [g / self.accum_steps for g in self.gradients]
            optimizer.apply_gradients(zip(avg_grads, self.vars))
            self.gradients = [tf.zeros_like(v) for v in self.vars]
            self.step = 0


def train_step(rag_head, optimizer, loss_fn,
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
        chunk_embs = tf.nn.l2_normalize(raw_chunk, axis=-1)
        # chunk_embs = proj_head(raw_chunk, training=True)
        # chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)

        # ----- Retrieval (stop gradient) -----
        # retrieved_np = retriever(chunk_embs, metadata)
        # retrieved = tf.nn.l2_normalize(
        #     tf.stop_gradient(tf.convert_to_tensor(retrieved_np, tf.float32)),
        #     axis=2
        # )

        # this should prolly change, but just leave it as this for now
        # doesnt really matter what this is tho because its never actually used
        retrieved = chunk_embs

        logits, _ = rag_head(chunk_embs, retrieved, training=True, use_retrieval = False)
        loss_cls = loss_fn(labels, logits)
        # loss = loss_fn(labels, logits)
        # ----- NEW: simple retrieval-aware contrast -----
        # loss_contrast = simple_retrieval_contrastive_loss(chunk_embs, retrieved)

        # combine them
        loss = loss_cls # + contrastive_coefficient * loss_contrast        # λ = 0.1 to start

    # Get grads for BOTH heads
    grads = tape.gradient(loss,
            rag_head.trainable_variables)
             
             # + proj_head.trainable_variables)

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
def evaluate(val_ds, rag_head, loss_fn):
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
        chunk_embs = tf.nn.l2_normalize(raw_chunk)
        # chunk_embs = proj_head(raw_chunk, training=False)
        # chunk_embs = tf.nn.l2_normalize(chunk_embs, axis=-1)
      
        # retrieved_np = retriever(chunk_embs, metadata)
        # retrieved = tf.nn.l2_normalize(
        #     tf.convert_to_tensor(retrieved_np, dtype=tf.float32),
        #     axis=2
        # )

        # again, same thing, this should prolly be something else
        # but it doesnt get used so it doesnt matter
        retrieved = chunk_embs
        logits, _ = rag_head(chunk_embs, retrieved, training=False, use_retrieval = False)

        all_val_logits.append(logits.numpy())
        all_val_labels.append(labels.numpy())

        # print()
        # print('eval logits')
        # print(logits)
        # print()

        # --- retrieved similarity ---
        # retrieved shape: (B, top_k, 768)
        # r1 = retrieved[0]
        # r2 = retrieved[1]



        # REAL cosine similarity (correct)
        cos_cls = -tf.keras.losses.cosine_similarity(chunk_embs[0], chunk_embs[1])
        print()
        # print("CLS true cosine similarity:", float(cos_cls))

        # Retrieved similarity (REAL)
        # cos_retr = -tf.reduce_mean(tf.keras.losses.cosine_similarity(r1, r2))
        # print("Retrieved true cosine similarity:", float(cos_retr))
        # retr_sims.append(cos_retr)

        # # --- combined feature similarity (optional) ---
        # z1 = tf.concat([chunk_embs[0], tf.reduce_mean(r1, axis=0)], axis=0)
        # z2 = tf.concat([chunk_embs[1], tf.reduce_mean(r2, axis=0)], axis=0)

        # Combined similarity (REAL)
        # cos_comb = -tf.keras.losses.cosine_similarity(z1, z2)
        # print("Combined true cosine similarity:", float(cos_comb))
        # comb_sims.append(cos_comb)

        # q1 = chunk_embs[0]
        # q2 = chunk_embs[1]

        # cos_qr1 = -tf.reduce_mean(
        #     tf.keras.losses.cosine_similarity(q1[None, :], r1)
        # )
        # cos_qr2 = -tf.reduce_mean(
        #     tf.keras.losses.cosine_similarity(q2[None, :], r2)
        # )

        # print(f'C1 retrieval purity: {cos_qr1}, C2 retrieval purity: {cos_qr2}')
        # c1s.append(cos_qr1)
        # c2s.append(cos_qr2)
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
    # print(f'MEAN comb sim:  {np.mean(comb_sims):.4f}, MEAN retr sim: {np.mean(retr_sims):.4f}')
    # print(f'STDEV comb sim:  {np.std(comb_sims):.4f}, STDEV retr sim: {np.std(retr_sims):.4f}')
    # print(f'MEAN c1s:  {np.mean(c1s):.4f}, MEAN c2s: {np.mean(c2s):.4f}')
    print("----------------------------------------------------")


# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load samples -> chunkify
    # ---------------------------------------------
    train_vids = configs_cls_only.VIDS_TO_USE
    samples = load_samples(train_vids,stride=1)
    chunk_samples = build_chunks(samples, chunk_size=12)

    random.shuffle(chunk_samples)

    # split 95/5
    n = len(chunk_samples)
    print(n)
    # train_chunks = chunk_samples[:int(0.95*n)]
    # val_chunks = chunk_samples[int(0.95*n):]

    train_chunks = chunk_samples[configs_cls_only.START_CHUNK_TRAIN:configs_cls_only.END_CHUNK_TRAIN] #was 2000
    val_chunks = chunk_samples[configs_cls_only.START_CHUNK_VALID:configs_cls_only.END_CHUNK_VALID]
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")

    # ---------------------------------------------
    # 2. Build TF datasets
    # ---------------------------------------------
    train_dataset = build_tf_dataset_chunks(train_chunks, batch_size=configs_cls_only.CHUNK_BATCH_SIZE)
    val_dataset   = build_tf_dataset_chunks(val_chunks,   batch_size=configs_cls_only.CHUNK_BATCH_SIZE)

    # ---------------------------------------------
    # 3. Build models
    # ---------------------------------------------
    
    # RAG head (trainable)
    rag_head = RAGHead(hidden_size=768, num_queries=configs_cls_only.NUM_QUERIES, num_layers=configs_cls_only.NUM_LAYERS,num_heads=configs_cls_only.NUM_HEADS)

    # proj_head = ProjectionHead(input_dim=768, hidden_dim=768*8, proj_dim=768)

    # proj_head.load_weights("projection_head.weights.h5")
    
    # Retrieval DB
    # client = PersistentClient(path="./chroma_store")
    # collection = client.get_or_create_collection(
    #     name=configs_cls_only.CHROMADB_COLLECTION,
    #     metadata={"hnsw:space": "cosine"}
    # )

    
    # retriever = FrameRetriever(collection, top_k=config.TOP_K, search_k=config.SEARCH_K)

    dummy_chunk = tf.zeros((1, 768))
    dummy_retrieved = tf.zeros((1, 10, 768))
    _ = rag_head(dummy_chunk, dummy_retrieved, training=False)
    # rag_head.save_weights(config.RAG_WEIGHTS)
    # rag_head.load_weights('rag_head_5vid_new_v2.weights.h5')

    # dummy = tf.zeros((1, 768), dtype=tf.float32)
    # _ = proj_head(dummy)   # builds variables
    # proj_head.save_weights(config.PROJ_WEIGHTS)
    # proj_head.load_weights("projection_head_5vid_new_v2.weights.h5")

    # ---------------------------------------------
    # 4. Optimizer / loss
    # ---------------------------------------------
    optimizer = tf.keras.optimizers.Adam(configs_cls_only.PHASE_1_LEARNING_RATE)
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
    EPOCHS = configs_cls_only.EPOCHS

    # lower lr to 1e-5 and make contrastive coefficient 0.1 at epoch 24 (rebuild every 6)
    # stopped at epoch 39 (so if you were to start this run again, start it at epoch 40)
    # results look great, val loss (old treshold > 0.5 loss) is 0.2171, train loss is 0.2649, train acc is 0.75
    accum_steps = configs_cls_only.ACCUM_BATCH_SIZE  # effective batch = physical batch * accum_steps
    accum = Accumulator(rag_head, accum_steps)

    contrastive_coefficient = 0.1
    for epoch in range(1,EPOCHS+1):
        # if(epoch < 5):
        #     continue
        print(f"\n================= EPOCH {epoch} =================")
        print('collection count in training ')
        print(0)
        # print(collection.count())
        losses = []
        accs = []
        batch_counter = 0
        
        
        if(epoch >= int(EPOCHS/2)+1): 
            optimizer.learning_rate.assign(configs_cls_only.PHASE_2_LEARNING_RATE)
            # contrastive_coefficient = configs_cls_only.PHASE_2_CONTRASTIVE_LOSS
        else: 
            optimizer.learning_rate.assign(configs_cls_only.PHASE_1_LEARNING_RATE)
            # contrastive_coefficient = configs_cls_only.PHASE_1_CONTRASTIVE_LOSS
        
        for frames_batch, metadata_batch, labels_batch in train_dataset:
            curloss, curacc = train_step(
                rag_head,
                optimizer, bce,
                frames_batch, metadata_batch, labels_batch,
                accum,contrastive_coefficient
            )

            batch_counter += 1
            losses.append(curloss)
            accs.append(curacc)
            if(batch_counter % configs_cls_only.PRINT_EVERY == 0):
                print(f"EPOCH {epoch} BATCH {batch_counter} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f}")
        print(f"EPOCH {epoch} TRAIN loss: {np.mean(losses):.4f}, EPOCH {epoch} TRAIN acc: {np.mean(accs):.4f}")

        # w = proj_head.trainable_variables[0].numpy()
        # print(
        #     "[TRAIN] proj W0 mean/std:",
        #     float(w.mean()),
        #     float(w.std())
        # )

        # proj_head.save_weights(configs_cls_only.PROJ_WEIGHTS)
        rag_head.save_weights(configs_cls_only.RAG_WEIGHTS)
        # validation at end of every epoch
        evaluate(val_dataset, rag_head, bce)
        # rebuild_db()
        # if(epoch % config.ADJUST_CONTRASTIVE_LOSS_EVERY == 0 and epoch >= config.ADJUST_CONTRASTIVE_LOSS_EVERY): 
        #     contrastive_coefficient += config.INCREMENT_CONTRASTIVE_LOSS_BY
        # if((epoch) % config.REBUILD_EVERY == 0 and (epoch) >= config.REBUILD_EVERY): 
        #     rebuild_db()
