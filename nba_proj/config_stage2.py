import os 
from datetime import datetime
import uuid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
# rag_head_7_vids_bs4_cls_only_12_epochs

# ratt_w_ordered_new 4 toks
# ratt_w_ordered_new_32toj 32 tokens
CHROMADB_COLLECTION = "ratt_db_chunk_encoder_all_vids_overlap_chunks_chunk_size8_stride2" #ratt_db_chunk_encoder_all_vids_relcls_v2
# ratt_db_chunk_encoder_all_vids_v2
# ratt_db_cached_30_clips_ordered_fixed_rebuild_w_labels v26

DELTA_T_NORM = 0.05 #this was 0.05

STAGE1_WEIGHTS = "./chunk_encoder_ckpts_chunk8_stride2/chunk_encoder_best_v3.weights.h5"
BUILD_RET_C = 75 #this was a 100
TOP_K = 32
SEARCH_K = 100#750

EPOCHS = 3
REBUILD_EVERY = 3
CHUNK_BATCH_SIZE = 16
PRINT_EVERY = 1
ACCUM_BATCH_SIZE = 1

ADJUST_CONTRASTIVE_LOSS_EVERY = 2
# INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
# PHASE_1_CONTRASTIVE_LOSS = 0.0
# PHASE_2_CONTRASTIVE_LOSS = 0.1

PHASE_1_LEARNING_RATE = 1e-3 # was -5 and then -6
PHASE_2_LEARNING_RATE = 1e-3

CACHE_PATH = f"./cache/retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_chunk_encoded_temporal.pkl" #temporary_new_way
STAGE3_CACHE_PATH = f"./cache/stage3_retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_chunk_encoded_new.pkl"
STAGE2_CACHE_PATH = f"./cache/stage2_retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_all_train_videos_topk10.pkl"
# STAGE2_CACHE_PATH = f"./cache/stage2_retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_all_train_videos_faster_cache.pkl"
ENCODED_EMBEDDINGS = f"./cache/encoded_embeddings.pkl"

NUM_QUERIES = TOP_K #was 16
NUM_LAYERS = 2
NUM_HEADS = 8

CHUNK_SIZE = 8
CHUNK_STRIDE = 2

START_CHUNK_TRAIN = 0
END_CHUNK_TRAIN = 3750

START_CHUNK_VALID = 4000
END_CHUNK_VALID = 4100

NUM_CLIPS_PER_VID = 30
# VIDS_TO_USE = ['vid3','vid4','vid6']
VIDS_TO_USE = ["vid2", 'vid3','vid4','vid5','vid6','vid7','vid8','vid10']
TRAIN_VIDS = ['vid2','vid3','vid4','vid5','vid6','vid7','vid8']
# TRAIN_VIDS = ['vid2']
TEST_VIDS = ['vid10']


RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_UID = uuid.uuid4().hex[:4]

RUN_ID = (
    f"{RUN_STAMP}_"
    f"vtest-{'-'.join(TEST_VIDS)}_"
    f"db-{CHROMADB_COLLECTION}_"
    f"ret{BUILD_RET_C}_k{TOP_K}_sk{SEARCH_K}_"
    f"dt{str(DELTA_T_NORM).replace('.', '')}_"
    f"ch{CHUNK_SIZE}_L{NUM_LAYERS}H{NUM_HEADS}Q{NUM_QUERIES}_"
    f"bs{CHUNK_BATCH_SIZE}_acc{ACCUM_BATCH_SIZE}_"
    f"e{EPOCHS}_lr{PHASE_1_LEARNING_RATE:.0e}to{PHASE_2_LEARNING_RATE:.0e}_"
    f"reb{REBUILD_EVERY}_"
    f"{RUN_UID}"
)

RATT_WEIGHTS = f"rag_weights/{RUN_ID}.weights.h5" 
# RATT_WEIGHTS = "rag_weights/20260328-100733_vtest-vid10_db-ratt_db_chunk_encoder_all_vids_overlap_chunks_ret75_k32_sk100_dt005_ch12_L12H8Q32_bs16_acc1_e3_lr1e-03to1e-03_reb3_a8be.weights.h5"
PROJ_WEIGHTS = 'proj_weights/ratt_proj_ordered.weights.h5' 
RANKER_WEIGHTS = f'ranker_weights/{RUN_ID}.weights.h5'

TEMPORAL_POS_WINDOW = 0.50   # same-video close neighbor counts as positive
TEMPORAL_NEG_WINDOW = 0.05   # same-video far-away counts as hard negative
EXACT_SELF_EPS = 1e-6        # exclude exact self
TEMPORAL_EXPAND_WINDOW = 0.50

# -----------------------------
# config
# -----------------------------
K_SIM = 5
K_CONTRAST = 5
K_TEMPORAL = 5

SEARCH_K_CONTENT = 500
SEARCH_K_TEMPORAL = 500

FUTURE_CHUNK_STEP = 5
POS_WEIGHT = 2
# VIDS_TO_USE = ['vid3']